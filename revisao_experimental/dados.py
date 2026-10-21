"""Leitura local, contratos, joins e transformações isoladas da revisão."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa

from .core import CONFIGURATIONS, PROTOCOL, atomic_json, signature, partition_blocks

ROOT = Path(__file__).resolve().parents[1]
RAW = {"A": "06_matriz_A_baseline.parquet", "B": "07_matriz_B_contexto.parquet", "C": "08_matriz_C_estendida.parquet"}
PROCESSED = {"A": "12_matriz_A_preprocessada.parquet", "B": "13_matriz_B_preprocessada.parquet", "C": "14_matriz_C_preprocessada.parquet"}
ABSOLUTE = {"effective_gas_price_log", "max_fee_per_gas_log", "max_priority_fee_log", "transaction_fee_paid_log"}
SPLIT_FILE = "03-machine-learning/01-features/resultados-splits/00_manifest_splits_temporais.json"
SEMANTIC = "04-pos-processamento/06-validacao-semantica-front-running/resultados/consolidado"


def legacy(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("legacy_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def q(value):
    return "'" + str(value).replace("'", "''") + "'"


def ident(value):
    return '"' + value.replace('"', '""') + '"'


def parquet(path):
    return f"read_parquet({q(Path(path).as_posix())})"


def connect(temp):
    temp = Path(temp)
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory={q(temp.as_posix())}")
    return con


def load_source(source):
    source = Path(source).resolve()
    doc = json.loads((source / SPLIT_FILE).read_text(encoding="utf-8-sig"))
    if doc.get("block_modulus") != 20 or doc.get("block_sampling_percent") != 5:
        raise ValueError("A revisão exige os mesmos blocos da execução de 5%")
    splits = doc["splits"]
    if len(splits) != 7 or len({s["key"] for s in splits}) != 7:
        raise ValueError("Esperadas sete janelas distintas")
    for role in ("train", "validation", "test"):
        if sum(s["role"] == role for s in splits) != 1:
            raise ValueError(f"Papel temporal inválido: {role}")
    for split in splits:
        directory = Path(split["output_directory"]).resolve()
        if not directory.is_relative_to(source):
            raise ValueError("Metadados e matrizes devem pertencer à execução-fonte")
    return doc, splits


def metadata(split):
    return Path(split["output_directory"]) / "05_metadados.parquet"


def configurations(contract):
    return [{"key": key, "matrix": key[0].upper(), "family": "sem_taxas_absolutas" if "sem_taxas" in key else "completa",
             "features": [f for f in contract[key[0].upper()] if "sem_taxas" not in key or f not in ABSOLUTE]}
            for key in CONFIGURATIONS]


def assert_unique(con, relation, key):
    count, distinct, missing = con.execute(f"SELECT count(*), count(DISTINCT {key}), count(*) FILTER(WHERE {key} IS NULL) FROM {relation}").fetchone()
    if count != distinct or missing:
        raise ValueError(f"Chave ausente ou duplicada: {key}")
    return count


def assert_alignment(con, left, right):
    a = assert_unique(con, left, "row_id")
    b = assert_unique(con, right, "row_id")
    common = con.execute(f"SELECT count(*) FROM {left} a JOIN {right} b USING(row_id)").fetchone()[0]
    if a != b or common != a:
        raise ValueError("Cobertura de row_id divergente entre metadados e matriz")


def arrays(con, query, features):
    result = con.execute(query).fetchnumpy()
    values = np.column_stack([result[f] for f in features]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Valores não finitos nas entradas do modelo")
    return result, values


def semantic_cohort(con, source, split, policy):
    if policy not in ("strict", "sensitivity"):
        raise ValueError("Política desconhecida")
    labels = parquet(Path(source) / SEMANTIC / "03_rotulos_transacao.parquet")
    meta = parquet(metadata(split))
    rel = f"(SELECT *, lower(tx_hash) AS normalized_hash FROM {labels} WHERE split={q(split['key'])})"
    assert_unique(con, rel, "normalized_hash")
    assert_unique(con, meta, "row_id")
    assert_unique(con, f"(SELECT lower(hash) AS h FROM {meta})", "h")
    missing = con.execute(f"SELECT count(*) FROM {rel} l LEFT JOIN {meta} m ON lower(m.hash)=l.normalized_hash WHERE m.row_id IS NULL").fetchone()[0]
    if missing:
        raise ValueError(f"{missing} rótulos sem metadados; não reduzir população silenciosamente")
    label, conflict = f"{policy}_label", f"{policy}_conflict"
    return con.execute(f"""SELECT m.row_id, lower(m.hash) AS tx_hash, m.block_number,
               l.{label}::INTEGER AS label, l.attack_types
        FROM {rel} l JOIN {meta} m ON lower(m.hash)=l.normalized_hash
        WHERE l.{label} IS NOT NULL AND NOT l.{conflict}
        ORDER BY lower(m.hash)"""
    ).to_arrow_table()


def prepare(source, store, if_target=None):
    """Somente treino/validação: nenhum escore de teste é produzido na busca."""
    if store.complete("prepare"):
        return
    _, splits = load_source(source)
    train = next(s for s in splits if s["role"] == "train")
    selected = [s for s in splits if s["role"] in ("train", "validation")]
    pp = legacy("03-machine-learning/01-features/05_preprocessar_matrizes.py")
    contract = pp.validate_feature_contract(Path(train["output_directory"]), splits)
    with store.stage("prepare") as out, connect(out / ".tmp") as con:
        counts = con.execute(f"SELECT block_number, count(*) FROM {parquet(metadata(train))} GROUP BY 1").fetchall()
        if any(b is None or int(b) % 20 for b, _ in counts):
            raise ValueError("Blocos fora da seleção externa de 5%")
        blocks = partition_blocks(dict(counts), target=if_target or PROTOCOL["if_target_rows"])
        # A complete manifest records every block, not just a seed or a count.
        atomic_json(out / "blocks.json", blocks)
        block_set = set(blocks["internal"])
        search_set, if_set = set(blocks["ae_search"]), set(blocks["if_fit"])
        table = pa.table({"block_number": [int(b) for b, _ in counts],
                          "internal": [b in block_set for b, _ in counts],
                          "ae_search": [b in search_set for b, _ in counts],
                          "if_fit": [b in if_set for b, _ in counts]})
        con.register("partitions", table)
        raw_c = parquet(Path(train["output_directory"]) / RAW["C"])
        meta_train = parquet(metadata(train))
        assert_alignment(con, raw_c, meta_train)
        fit_c = out / "fit_c.parquet"
        con.execute(f"COPY (SELECT c.* FROM {raw_c} c JOIN {meta_train} m USING(row_id) JOIN partitions p USING(block_number) WHERE NOT p.internal ORDER BY c.row_id) TO {q(fit_c.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)")
        params = pp.fit_robust_parameters(con, fit_c)
        # The fit data are part of a resumable stage, and are never historical paths.
        config = {"matrices": contract, "parameters": params,
                  "preprocessing_signature": signature({"blocks": blocks, "parameters": params, "matrices": contract}),
                  "configurations": configurations(contract),
                  "block_counts": {key: len(value) for key, value in blocks.items()},
                  "row_counts": {key: sum(dict(counts)[b] for b in value) for key, value in blocks.items()}}
        atomic_json(out / "preprocessing.json", config)
        for split in selected:
            transform_split(con, split, out, config, pp)


def transform_split(con, split, out, config, pp=None):
    pp = pp or legacy("03-machine-learning/01-features/05_preprocessar_matrizes.py")
    directory = out / "splits" / split["key"]
    directory.mkdir(parents=True, exist_ok=True)
    for matrix in ("A", "B", "C"):
        path = Path(split["output_directory"]) / RAW[matrix]
        assert_alignment(con, parquet(path), parquet(metadata(split)))
        dest = directory / PROCESSED[matrix]
        # Retry writes only inside the isolated failed stage, never to inputs.
        if dest.exists():
            dest.unlink()
        pp.transform_matrix(con, path, dest, config["matrices"][matrix], config["parameters"])
        pp.validate_output(con, split["key"], matrix, path, dest, config["matrices"][matrix])


def training_arrays(con, split, directory, config, blocks, method, search):
    matrix = parquet(directory / "splits" / split["key"] / PROCESSED[config["matrix"]])
    meta = parquet(metadata(split))
    internal_set = set(blocks["internal"])
    rows = [(b, b in internal_set) for b in (blocks["ae_search"] if method == "AE" and search
                                                    else blocks["fit"] + blocks["internal"] if method == "AE" else blocks["if_fit"])]
    con.register("chosen_blocks", pa.table({"block_number": [r[0] for r in rows], "internal": [r[1] for r in rows]}))
    cols = ", ".join("x." + ident(f) for f in config["features"])
    result, values = arrays(con, f"SELECT x.row_id, p.internal, {cols} FROM {matrix} x JOIN {meta} m USING(row_id) JOIN chosen_blocks p USING(block_number) ORDER BY x.row_id", config["features"])
    return values, np.asarray(result["internal"], dtype=bool)


def validation_arrays(con, source, split, directory, config, policy="strict"):
    cohort = semantic_cohort(con, source, split, policy)
    con.register("cohort", cohort)
    matrix = parquet(directory / "splits" / split["key"] / PROCESSED[config["matrix"]])
    assert_unique(con, matrix, "row_id")
    columns = ", ".join("x." + ident(f) for f in config["features"])
    result, values = arrays(con, f"SELECT c.label, c.tx_hash, {columns} FROM cohort c JOIN {matrix} x USING(row_id) ORDER BY c.tx_hash", config["features"])
    if len(values) != len(cohort) or set(result["label"]) != {0, 1}:
        raise ValueError("Cobertura ou classes inválidas na validação")
    return np.asarray(result["label"], dtype=np.uint8), values
