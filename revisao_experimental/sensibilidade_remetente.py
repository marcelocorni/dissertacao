"""Reavalia escores congelados no teste estrito, sem treino, RPC ou alteração dos rótulos."""
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
import argparse
import json
import math
import duckdb

EXECUTIONS = Path(__file__).resolve().parents[1] / 'execucoes'
SPLIT = 'teste_final_2025'
MODELS = {'AE B-ST': ('AE', 'b_sem_taxas_absolutas', 'reconstruction_error'),
          'IF B completa': ('IF', 'b_completa', 'anomaly_score')}

def discrimination(rows):
    """AP não interpolada por limiar; ROC-AUC com meio crédito para empates."""
    npos = sum(y for y, s in rows)
    nneg = len(rows) - npos
    assert npos and nneg and all(math.isfinite(s) for _, s in rows)
    ap = auc = 0.0
    tp = fp = 0
    for _, tied in groupby(sorted(rows, key=lambda r: r[1], reverse=True), key=lambda r: r[1]):
        labels = [y for y, _ in tied]
        p = sum(labels)
        n = len(labels) - p
        auc += p * (nneg - fp - n / 2)
        tp += p
        fp += n
        ap += p / npos * tp / (tp + fp)
    return ap, auc / (npos * nneg)

def evaluate(events, metadata, scores):
    txs = {h for members in events.values() for h, _, _ in members}
    positive = sum(metadata[h][1] for h in txs)
    out = {'events': len(events), 'transactions': len(txs), 'positive': positive,
           'negative': len(txs)-positive, 'prevalence': positive/len(txs),
           'patterns': dict(Counter(m[0][2] for m in events.values())),
           'blocks': len({metadata[h][2] for h in txs}), 'models': {}}
    for model, by_hash in scores.items():
        higher = tied = pairs = all_higher = 0
        for members in events.values():
            victims = [h for h, role, _ in members if role == 'victim']
            attackers = [h for h, role, _ in members if role != 'victim']
            assert len(victims) == 1 and len(attackers) in (1, 2)
            assert len({metadata[h][2] for h, _, _ in members}) == 1
            v = by_hash[victims[0]]
            margins = [by_hash[h] - v for h in attackers]
            pairs += len(margins)
            higher += sum(m > 0 for m in margins)
            tied += sum(m == 0 for m in margins)
            all_higher += all(m > 0 for m in margins)
        ap, auc = discrimination([(metadata[h][1], by_hash[h]) for h in txs])
        out['models'][model] = {'ap': ap, 'roc_auc': auc, 'pairs': pairs,
                                'higher': higher, 'tied': tied, 'superiority': higher / pairs,
                                'complete_event_superiority': all_higher / len(events)}
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=EXECUTIONS / 'amostra-05pct')
    parser.add_argument('--scores-run', type=Path, default=EXECUTIONS / 'amostra-05pct-revisao-hpo')
    parser.add_argument('--output', type=Path, help='JSON novo, opcional; por padrão usa stdout.')
    args = parser.parse_args()
    sem = args.source_run / '04-pos-processamento/06-validacao-semantica-front-running/resultados/consolidado'
    meta = args.source_run / '03-machine-learning/01-features/resultados-splits/06-teste_final_2025/05_metadados.parquet'
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        labels = sem / '03_rotulos_transacao.parquet'
        roles = sem / '02_papeis_evento.parquet'
        con.execute("""
            CREATE TEMP TABLE cohort AS
            SELECT m.row_id, lower(m.hash) AS h, l.strict_label::INTEGER AS y,
                   m.block_number, lower(m.from_address) AS sender
            FROM read_parquet(?) l JOIN read_parquet(?) m ON lower(l.tx_hash)=lower(m.hash)
            WHERE l.split=? AND l.strict_label IS NOT NULL AND NOT l.strict_conflict
        """, [str(labels), str(meta), SPLIT])
        cohort = con.execute("SELECT * FROM cohort").fetchall()
        assert len(cohort) == len({r[0] for r in cohort}) == len({r[1] for r in cohort}) == 7118
        metadata = {h: (i,y,b,s) for i,h,y,b,s in cohort}
        role_rows = con.execute("""
            SELECT r.event_key, lower(r.tx_hash), r.role, r.adjudicated_type, r.strict_label, c.y
            FROM read_parquet(?) r LEFT JOIN cohort c ON c.h=lower(r.tx_hash)
            WHERE r.split=? AND r.event_key IN
                (SELECT event_key FROM read_parquet(?) WHERE split=? AND strict_label IS NOT NULL)
        """, [str(roles), SPLIT, str(roles), SPLIT]).fetchall()
        events = defaultdict(list)
        sender_events = defaultdict(set)
        for e,h,role,pattern,y,cohort_y in role_rows:
            assert y == cohort_y == (0 if role == 'victim' else 1), (e,role,y,cohort_y)
            assert role in {'victim','attacker','attacker_front','attacker_back'}
            events[e].append((h,role,pattern))
            if y == 1:
                sender = metadata[h][3]
                assert sender
                sender_events[sender].add(e)
        assert len(events)==2409 and len(sender_events)==120
        assert {h for members in events.values() for h,_,_ in members} == set(metadata)
        ranked = sorted(sender_events, key=lambda s: (-len(sender_events[s]),s))
        dominant = ranked[0]
        removed = sender_events[dominant]
        assert len(removed)==1515 and len(sender_events[ranked[1]]) < len(removed)
        remaining = {e: members for e,members in events.items() if e not in removed}
        scores = {}
        for label, (method, configuration, col) in MODELS.items():
            path = args.scores_run / 'scores' / SPLIT / method / configuration / 'scores.parquet'
            rows = con.execute(f'SELECT c.h, s.{col} FROM cohort c JOIN read_parquet(?) s USING(row_id)', [str(path)]).fetchall()
            assert len(rows)==len({h for h,s in rows})==len(cohort)
            scores[label] = dict(rows)
    full = evaluate(events,metadata,scores)
    restricted = evaluate(remaining,metadata,scores)
    expected = {'AE B-ST': (0.8636,0.8136,84.76), 'IF B completa': (0.8342,0.7308,75.27)}
    for model, (ap,auc,sup) in expected.items():
        observed = full['models'][model]
        assert observed['pairs']==4711
        assert (round(observed['ap'],4),round(observed['roc_auc'],4),round(observed['superiority']*100,2))==(ap,auc,sup)
    output = {'split': SPLIT, 'policy': 'strict', 'frozen_models': MODELS,
              'dominant_sender': dominant, 'excluded_events': len(removed),
              'rule': 'Exclude whole events in which the dominant sender has an attacker role. Keep only unique transactions represented by retained events, with original strict labels and scores.',
              'full':full, 'without_dominant':restricted}
    rendered = json.dumps(output, indent=2, ensure_ascii=False) + '\n'
    if args.output is None:
        print(rendered, end='')
    else:
        with args.output.open('x', encoding='utf-8') as result:
            result.write(rendered)

if __name__=='__main__': main()
