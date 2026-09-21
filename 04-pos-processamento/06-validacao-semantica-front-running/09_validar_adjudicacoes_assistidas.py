# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Recalcula evidências e valida adjudicações assistidas sem confiar no rótulo salvo."""

from __future__ import annotations

import argparse, csv, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import duckdb

V2_SWAP="0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
V3_SWAP="0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

def q(path: Path)->str: return "'"+path.as_posix().replace("'","''")+"'"
def low(v: Any)->str: return "" if v is None else str(v).lower()
def hx(v: Any)->int: return int(str(v),16) if v not in (None,"","0x") else 0
def rows(con: duckdb.DuckDBPyConnection,path: Path)->list[dict[str,Any]]:
    cur=con.execute(f"SELECT * FROM read_parquet({q(path)})"); cols=[x[0] for x in cur.description]
    return [dict(zip(cols,row)) for row in cur.fetchall()]
def logs(tx: dict[str,Any])->list[dict[str,Any]]:
    try:return json.loads(tx.get("logs_json") or "[]")
    except json.JSONDecodeError:return []
def words(data: Any)->list[int]:
    p=low(data).removeprefix("0x"); return [int(p[i:i+64],16) for i in range(0,len(p),64) if len(p[i:i+64])==64]
def signed(v:int)->int:return v-(1<<256) if v>=(1<<255) else v
def swaps(tx:dict[str,Any])->dict[tuple[str,str],tuple[int,int]]:
    result={}
    for item in logs(tx):
        topics=item.get("topics") or []; topic=low(topics[0]) if topics else ""; values=words(item.get("data")); address=low(item.get("address"))
        if topic==V2_SWAP and len(values)>=4: result[(address,"uniswap_v2")]=(values[0]-values[2],values[1]-values[3])
        elif topic==V3_SWAP and len(values)>=2: result[(address,"uniswap_v3")]=(signed(values[0]),signed(values[1]))
    return result
def direction(delta:tuple[int,int])->int|None:
    if delta[0]>0 and delta[1]<0:return 0
    if delta[1]>0 and delta[0]<0:return 1
    return None
def similarity(a:str,b:str)->float:
    if not a or not b or a=="0x" or b=="0x":return 0.0
    aa=a.lower().removeprefix("0x")[8:]; bb=b.lower().removeprefix("0x")[8:]
    aw=[aa[i:i+64] for i in range(0,len(aa),64)]; bw=[bb[i:i+64] for i in range(0,len(bb),64)]
    return sum(x==y for x,y in zip(aw,bw))/max(len(aw),len(bw),1)

def main()->int:
    here=Path(__file__).resolve().parent; default=here/"resultados"/"transicao_dencun_2024"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier",type=Path,default=default/"08_dossie_auditoria.parquet")
    parser.add_argument("--transactions",type=Path,default=default/"01_transacoes_rpc.parquet")
    parser.add_argument("--decisions",type=Path,default=default/"15_decisoes_dossie.csv")
    parser.add_argument("--expected-reviewer",default="Marcelo Corni Alves")
    parser.add_argument("--output-dir",type=Path,default=default)
    args=parser.parse_args(); con=duckdb.connect()
    dossier=rows(con,args.dossier.resolve()); tx={low(x["tx_hash"]):x for x in rows(con,args.transactions.resolve())}; con.close()
    decisions_path=args.decisions.resolve()
    with decisions_path.open(encoding="utf-8-sig",newline="") as stream: decisions={r["audit_id"]:r for r in csv.DictReader(stream)}
    results=[]; counts=Counter()
    for event in dossier:
        issues=[]; aid=str(event["audit_id"]); decision=decisions.get(aid)
        if not decision:
            results.append({"audit_id":aid,"detection_event_id":event["detection_event_id"],"detector":event["detector"],"saved_label":None,"recomputed_label":None,"validation_result":"invalid","issues":"decisão ausente"}); counts["invalid"]+=1; continue
        needed=[event.get("attacker_front_hash"),event.get("victim_hash")]
        if event["detector"]=="insertion":needed.append(event.get("attacker_back_hash"))
        if any(low(h) not in tx for h in needed):
            recomputed=None; issues.append("transação enriquecida ausente")
        else:
            front,victim=tx[low(needed[0])],tx[low(needed[1])]
            fi,vi=hx(front.get("transaction_index_hex")),hx(victim.get("transaction_index_hex"))
            if fi>=vi:issues.append("ordem atacante/vítima inválida")
            if event["detector"]=="insertion":
                back=tx[low(needed[2])]; bi=hx(back.get("transaction_index_hex"))
                if vi>=bi:issues.append("ordem vítima/back inválida")
                if low(front.get("from_address"))!=low(back.get("from_address")):issues.append("remetentes externos divergentes")
                if any(low(item.get("receipt_status_hex"))!="0x1" for item in (front,victim,back)):issues.append("uma perna não foi executada")
                fs,vs,bs=swaps(front),swaps(victim),swaps(back); matched=[]
                for key in set(fs)&set(vs)&set(bs):
                    fd,vd,bd=direction(fs[key]),direction(vs[key]),direction(bs[key])
                    if fd is not None and fd==vd and bd is not None and bd!=fd:matched.append(key)
                if not matched:issues.append("padrão sandwich não reproduzido")
                recomputed="confirmado" if not issues else None
            else:
                ai=low(front.get("input_data")); vi_data=low(victim.get("input_data")); sim=similarity(ai,vi_data)
                same_selector=len(ai)>=10 and ai[:10]==vi_data[:10]; common=set(swaps(front))&set(swaps(victim)); attacker_ok=low(front.get("receipt_status_hex"))=="0x1"; victim_ok=low(victim.get("receipt_status_hex"))=="0x1"
                if not same_selector or sim<0.80 or not attacker_ok:issues.append("equivalência mínima de displacement não reproduzida")
                if same_selector and sim>=0.80 and attacker_ok and not victim_ok:recomputed="confirmado"
                elif same_selector and sim>=0.80 and attacker_ok and common:recomputed="provável"
                else:recomputed=None
                if recomputed=="provável" and not common:issues.append("pool comum ausente")
        saved=decision["manual_label"]
        if recomputed and saved!=recomputed:issues.append(f"rótulo salvo={saved}, recalculado={recomputed}")
        if decision.get("reviewer")!=args.expected_reviewer:issues.append("responsável registrado inesperado")
        if not decision.get("notes","").strip():issues.append("justificativa vazia")
        result="valid" if not issues else ("warning" if recomputed and saved==recomputed else "invalid")
        counts[result]+=1
        results.append({"audit_id":aid,"detection_event_id":event["detection_event_id"],"detector":event["detector"],"saved_label":saved,"recomputed_label":recomputed,"validation_result":result,"issues":" | ".join(issues)})
    extras=sorted(set(decisions)-{str(x["audit_id"]) for x in dossier})
    for aid in extras: results.append({"audit_id":aid,"detection_event_id":"","detector":decisions[aid].get("detector"),"saved_label":decisions[aid].get("manual_label"),"recomputed_label":None,"validation_result":"invalid","issues":"decisão sem evento no dossiê"}); counts["invalid"]+=1
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); fields=list(results[0])
    with (out/"17_validacao_adjudicacoes.csv").open("w",encoding="utf-8-sig",newline="") as stream: w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(results)
    with (out/"18_resumo_validacao_adjudicacoes.csv").open("w",encoding="utf-8-sig",newline="") as stream: w=csv.writer(stream);w.writerow(["validation_result","events"]);w.writerows(sorted(counts.items()))
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"dossier":str(args.dossier.resolve()),"decisions":str(decisions_path),"decision_origin":"assisted_deterministic","expected_reviewer":args.expected_reviewer,"events":len(dossier),"decisions_found":len(decisions),"counts":dict(counts),"checks":["coverage","role hashes","block order","receipt status","outer sender","swap pool and direction","calldata selector and similarity","saved/recomputed label","reviewer","justification"]}
    (out/"19_manifest_validacao_adjudicacoes.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Validação das adjudicações: {dict(counts)}")
    return 1 if counts["invalid"] else 0

if __name__=="__main__":raise SystemExit(main())
