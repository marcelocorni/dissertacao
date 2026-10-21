"""Gera exclusivamente dados artificiais, dentro da pasta temporária do teste."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from revisao_experimental.core import atomic_json
from revisao_experimental.dados import RAW, SEMANTIC, SPLIT_FILE


def synthetic_source(root):
    root = Path(root)
    a = ["gas_used_log", "gas_used_ratio", "effective_gas_price_log", "max_fee_per_gas_log",
         "max_priority_fee_log", "fee_cap_utilization", "transaction_value_log", "nonce_log",
         "receipt_status", "tx_type_0", "tx_type_1", "tx_type_3", "transaction_index_pct"]
    b = a + ["block_tx_count_log", "sender_tx_count_block_log", "receiver_tx_count_block_log", "gas_price_rank_block",
             "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
    c = b + ["gas_limit_log", "transaction_fee_paid_log"]
    contract = dict(A=a, B=b, C=c)
    rng = np.random.default_rng(42)
    windows = [("estresse_pre_dencun_2024", "historical_stress"), ("transicao_dencun_2024", "protocol_transition"),
               ("treino_2024_pos_dencun", "train"), ("validacao_2025_pre_pectra", "validation"),
               ("transicao_pectra_2025", "protocol_transition"), ("teste_final_2025", "test"),
               ("estresse_fusaka_2025", "future_stress")]
    splits, labels, roles, external = [], [], [], []
    for index, (key, role) in enumerate(windows, 1):
        n = 2000 if role == "train" else 40
        blocks = np.repeat(np.arange(index * 10000, index * 10000 + n // 2) * 20, 2)
        ids = np.arange(n, dtype=np.uint64)
        hashes = [f"0x{int(index * 100000 + i):064x}" for i in ids]
        directory = root / "03-machine-learning/01-features/resultados-splits" / f"{index:02d}_{key}"
        directory.mkdir(parents=True, exist_ok=True)
        typ = np.where(ids % 7 == 0, 4, 2)
        pq.write_table(pa.table({"row_id": ids, "hash": hashes, "block_number": blocks,
                                "from_address": [f"sender_{i % 3}" for i in ids],
                                "transaction_type": typ, "is_type_4": typ == 4}), directory / "05_metadados.parquet")
        values = {f: rng.uniform(.1, 1., n) for f in c}
        values.update({"tx_type_0": np.zeros(n), "tx_type_1": np.zeros(n), "tx_type_3": np.zeros(n), "receipt_status": np.ones(n)})
        for matrix, features in contract.items():
            pq.write_table(pa.table({"row_id": ids, **{f: values[f] for f in features}}), directory / RAW[matrix])
        atomic_json(directory / "10_manifest_execucao.json", {f"matrix_{m}": fs for m, fs in contract.items()})
        splits.append(dict(order=index, key=key, role=role, output_directory=str(directory), sample_rows=n,
                           start_date="2024-01-01", end_date="2024-01-02"))
        for i in range(10):
            label = 1 if i % 2 == 0 else 0
            labels.append(dict(split=key, tx_hash=hashes[i], strict_label=label, sensitivity_label=label,
                               strict_conflict=False, sensitivity_conflict=False, attack_types="displacement"))
            roles.append(dict(split=key, tx_hash=hashes[i], strict_label=label, sensitivity_label=label,
                              event_key=f"{key}::{i // 2}", role="attacker" if label else "victim",
                              adjudicated_type="displacement"))
        external.extend(dict(split=key, row_id=int(i), label=int(i % 2)) for i in ids)
        semantic = root / "04-pos-processamento/06-validacao-semantica-front-running/resultados" / key
        atomic_json(semantic / "07_manifest_validacao.json", {"events": 5, "status_counts": {"displacement:confirmed": 5}})
    semantic = root / SEMANTIC
    semantic.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(labels), semantic / "03_rotulos_transacao.parquet")
    pq.write_table(pa.Table.from_pylist(roles), semantic / "02_papeis_evento.parquet")
    external_path = root / "04-pos-processamento/resultados-labelcloud/05_rotulos_binarios_hipoteticos_labelcloud.parquet"
    external_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(external), external_path)
    results = root / "03-machine-learning/01-features/resultados"
    results.mkdir(parents=True, exist_ok=True)
    for filename in ("09_correlacao_pearson.csv", "10_correlacao_spearman.csv"):
        (results / filename).write_text("feature,x,y\nx,1,0\ny,0,1\n", encoding="utf-8")
    atomic_json(results / "15_resultado_execucao.json", {"start_date": "2024-01-01", "end_date": "2024-01-02", "correlation_rows_used": 40})
    atomic_json(root / SPLIT_FILE, {"block_modulus": 20, "block_sampling_percent": 5, "splits": splits})
    return splits
