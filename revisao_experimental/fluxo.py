"""Etapas explícitas da revisão: preparar, buscar, ajustar e pontuar."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .core import CONFIGURATIONS, PROTOCOL, atomic_json, grids, select_candidate
from .dados import (PROCESSED, configurations, connect, legacy, load_source, metadata,
                    parquet, q, training_arrays, transform_split, validation_arrays)
from .metricas import discrimination
from .modelos import Autoencoder, fit_model, score_model


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(store, name):
    if not store.complete(name):
        raise ValueError(f"Etapa prévia incompleta: {name}")


def search(source, store, device):
    require(store, "prepare")
    _, splits = load_source(source)
    train, validation = [next(s for s in splits if s["role"] == r) for r in ("train", "validation")]
    prep = store.destination("prepare")
    document, blocks = read_json(prep / "preprocessing.json"), read_json(prep / "blocks.json")
    for method, candidates in grids().items():
        for candidate in candidates:
            for config in document["configurations"]:
                name = f"search/{method}/{candidate['id']}/{config['key']}"
                if store.complete(name):
                    continue
                print(f"[busca] {method} {candidate['id']} {config['key']}", flush=True)
                with store.stage(name) as out, connect(out / ".tmp") as con:
                    start = time.perf_counter()
                    values, internal = training_arrays(con, train, prep, config, blocks, method, True)
                    load_seconds = time.perf_counter() - start
                    model, info = fit_model(method, candidate["parameters"], values, internal, device)
                    del values
                    labels, validation_values = validation_arrays(con, source, validation, prep, config)
                    start = time.perf_counter()
                    scores = score_model(method, model, validation_values)
                    infer_seconds = time.perf_counter() - start
                    np.savez_compressed(out / "validation.npz", labels=labels, scores=scores)
                    atomic_json(out / "result.json", {"method": method, "candidate": candidate,
                                "configuration": config, "rows": len(labels), "load_seconds": load_seconds,
                                "infer_seconds": infer_seconds, **discrimination(labels, scores), **info})
                    del model, validation_values
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    select_winners(store)


def select_winners(store):
    if store.complete("selection"):
        return
    records, winners = [], {}
    for method, candidates in grids().items():
        completed = {}
        for candidate in candidates:
            completed[candidate["id"]] = {}
            for key in CONFIGURATIONS:
                name = f"search/{method}/{candidate['id']}/{key}"
                require(store, name)
                record = read_json(store.destination(name) / "result.json")
                completed[candidate["id"]][key] = record["average_precision"]
                records.append({"method": method, "candidate": candidate["id"], "configuration": key,
                                "average_precision": record["average_precision"], "rows": record["rows"]})
        winners[method] = select_candidate(method, completed)
    with store.stage("selection") as out:
        atomic_json(out / "winners.json", winners)
        write_csv(out / "trials.csv", records)


def final_fit(source, store, device):
    require(store, "prepare")
    require(store, "selection")
    prep = store.destination("prepare")
    document, blocks = read_json(prep / "preprocessing.json"), read_json(prep / "blocks.json")
    winners = read_json(store.destination("selection") / "winners.json")
    _, splits = load_source(source)
    train = next(s for s in splits if s["role"] == "train")
    for method in ("AE", "IF"):
        for config in document["configurations"]:
            name = f"final/{method}/{config['key']}"
            if store.complete(name):
                continue
            with store.stage(name) as out, connect(out / ".tmp") as con:
                start = time.perf_counter()
                values, internal = training_arrays(con, train, prep, config, blocks, method, False)
                load_seconds = time.perf_counter() - start
                model, info = fit_model(method, winners[method]["parameters"], values, internal, device)
                if method == "AE":
                    torch.save(model.state_dict(), out / "model.pt")
                else:
                    joblib.dump(model, out / "model.joblib")
                atomic_json(out / "model.json", {"method": method, "configuration": config,
                            "parameters": winners[method]["parameters"], "load_seconds": load_seconds, **info})
                del values, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def load_model(store, method, config, device):
    name = f"final/{method}/{config['key']}"
    require(store, name)
    out = store.destination(name)
    document = read_json(out / "model.json")
    if method == "IF":
        return joblib.load(out / "model.joblib")
    parameters = document["parameters"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(len(config["features"]), parameters["hidden_dims"], parameters["latent_dim"])
    model.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval()


def score_split(source, store, split, method, config, device):
    name = f"scores/{split['key']}/{method}/{config['key']}"
    if store.complete(name):
        return
    prepared = store.destination("prepare")
    document = read_json(prepared / "preprocessing.json")
    if split["role"] in ("train", "validation"):
        matrix_dir = prepared
    else:
        transform_name = f"transform/{split['key']}"
        if not store.complete(transform_name):
            with store.stage(transform_name) as out, connect(out / ".tmp") as con:
                transform_split(con, split, out, document)
        matrix_dir = store.destination(transform_name)
    if split["role"] != "validation":
        _, splits = load_source(source)
        validation = next(s for s in splits if s["role"] == "validation")
        threshold_name = f"scores/{validation['key']}/{method}/{config['key']}"
        require(store, threshold_name)
        thresholds = read_json(store.destination(threshold_name) / "summary.json")["thresholds"]
    else:
        thresholds = None
    model = load_model(store, method, config, device)
    with store.stage(name) as out:
        start = time.perf_counter()
        ids, chunks = [], []
        path = matrix_dir / "splits" / split["key"] / PROCESSED[config["matrix"]]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=262144, columns=["row_id", *config["features"]]):
            values = np.column_stack([batch.column(i).to_numpy() for i in range(1, batch.num_columns)]).astype(np.float32)
            ids.append(batch.column(0).to_numpy())
            chunks.append(score_model(method, model, values))
        row_ids, scores = np.concatenate(ids), np.concatenate(chunks)
        if len(np.unique(row_ids)) != len(row_ids):
            raise ValueError("Escores com row_id duplicado")
        if thresholds is None:
            thresholds = {key: float(np.quantile(scores, quantile)) for key, quantile in PROTOCOL["quantiles"].items()}
        flags = {f"is_anomaly_{key}": scores >= value for key, value in thresholds.items()}
        column = "reconstruction_error" if method == "AE" else "anomaly_score"
        pq.write_table(pa.table({"row_id": row_ids, column: scores, **flags}), out / "scores.parquet", compression="zstd")
        statistics = {"score_min": float(scores.min()), "score_max": float(scores.max()),
                      "score_mean": float(scores.mean()), "score_std": float(scores.std())}
        statistics.update({key: float(np.quantile(scores, v)) for key, v in
                           {"score_q250": .25, "score_median": .5, "score_q750": .75, "score_q950": .95,
                            "score_q990": .99, "score_q995": .995, "score_q999": .999}.items()})
        atomic_json(out / "summary.json", {"split": split["key"], "configuration": config["key"],
                    "rows": len(scores), "thresholds": thresholds, "score_seconds": time.perf_counter() - start,
                    "statistics": statistics, "rates": {key.removeprefix("is_anomaly_"): float(v.mean()) for key, v in flags.items()},
                    "counts": {key.removeprefix("is_anomaly_"): int(v.sum()) for key, v in flags.items()}})


def score_all(source, store, device):
    require(store, "selection")
    doc = read_json(store.destination("prepare") / "preprocessing.json")
    _, splits = load_source(source)
    # All choices and ten final fits must exist before any new test scores.
    for method in ("AE", "IF"):
        for config in doc["configurations"]:
            require(store, f"final/{method}/{config['key']}")
    for split in sorted(splits, key=lambda s: (s["role"] != "validation", s["order"])):
        for method in ("AE", "IF"):
            for config in doc["configurations"]:
                score_split(source, store, split, method, config, device)
