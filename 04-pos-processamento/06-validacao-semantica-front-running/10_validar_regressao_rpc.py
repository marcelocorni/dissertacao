# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Compara a coleta por bloco com o baseline por transação."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import duckdb

def q(p:Path)->str:return "'"+p.as_posix().replace("'","''")+"'"
def main()->int:
    here=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=here/"resultados"/"transicao_dencun_2024"/"01_transacoes_rpc.parquet")
    parser.add_argument("--candidate",type=Path,default=here/"resultados-blocos"/"transicao_dencun_2024"/"01_transacoes_rpc.parquet")
    parser.add_argument("--output",type=Path,default=here/"resultados-blocos"/"transicao_dencun_2024"/"04_regressao_rpc.json")
    args=parser.parse_args(); b=args.baseline.resolve(); c=args.candidate.resolve()
    if not b.is_file() or not c.is_file():parser.error("Baseline ou candidato não encontrado.")
    con=duckdb.connect(); columns=[x[0] for x in con.execute(f"DESCRIBE SELECT * FROM read_parquet({q(b)})").fetchall()]
    compare=[x for x in columns if x!="logs_json"]
    conditions=" OR ".join(f"a.{x} IS DISTINCT FROM b.{x}" for x in compare)
    stats=con.execute(f"""WITH a AS (SELECT * FROM read_parquet({q(b)})), b AS (SELECT * FROM read_parquet({q(c)}))
    SELECT (SELECT count(*) FROM a),(SELECT count(*) FROM b),(SELECT count(*) FROM a ANTI JOIN b ON lower(a.tx_hash)=lower(b.tx_hash)),(SELECT count(*) FROM b ANTI JOIN a ON lower(a.tx_hash)=lower(b.tx_hash)),(SELECT count(*) FROM a JOIN b ON lower(a.tx_hash)=lower(b.tx_hash) WHERE {conditions}),(SELECT count(*) FROM a JOIN b ON lower(a.tx_hash)=lower(b.tx_hash) WHERE json(a.logs_json) IS DISTINCT FROM json(b.logs_json))""").fetchone(); con.close()
    keys=["baseline_rows","candidate_rows","missing","extra","field_mismatches","logs_mismatches"]; result=dict(zip(keys,stats)); result.update({"created_at_utc":datetime.now(timezone.utc).isoformat(),"baseline":str(b),"candidate":str(c)}); result["passed"]=all(result[k]==0 for k in ["missing","extra","field_mismatches","logs_mismatches"]) and result["baseline_rows"]==result["candidate_rows"]
    args.output.resolve().parent.mkdir(parents=True,exist_ok=True);args.output.resolve().write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(result,ensure_ascii=False));return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
