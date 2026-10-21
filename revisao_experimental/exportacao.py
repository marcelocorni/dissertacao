"""Relatórios compatíveis e figuras; nenhum gerador RPC é chamado."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import csv
from pathlib import Path

from .core import CONFIGURATIONS, atomic_json, file_hash
from .dados import PROCESSED, ROOT, connect, legacy, load_source, metadata, parquet, q
from .fluxo import read_json, require, write_csv


def link_artifact(source, destination):
    """Somente saídas NOVAS: não criar hardlinks para as entradas históricas."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != file_hash(source):
            raise ValueError("Exportação existente diverge da saída assinada")
    else:
        os.link(source, destination)


def export(source, store):
    if store.complete("exports"):
        return
    _, splits = load_source(source)
    prep = read_json(store.destination("prepare") / "preprocessing.json")
    configs = prep["configurations"]
    for split in splits:
        for method in ("AE", "IF"):
            for config in configs:
                require(store, f"scores/{split['key']}/{method}/{config['key']}")
    with store.stage("exports") as out, connect(out / ".tmp") as con:
        revised_splits = []
        for split in splits:
            base = store.destination("prepare") if split["role"] in ("train", "validation") else store.destination(f"transform/{split['key']}")
            new_dir = out / "splits" / split["key"]
            new_dir.mkdir(parents=True, exist_ok=True)
            for filename in PROCESSED.values():
                link_artifact(base / "splits" / split["key"] / filename, new_dir / filename)
            atomic_json(new_dir / "15_manifest_preprocessamento.json", {"preprocessing_signature": prep["preprocessing_signature"]})
            revised_splits.append({**split, "output_directory": str(new_dir), "metadata_path": str(metadata(split))})
        atomic_json(out / "splits.json", {"splits": revised_splits})
        atomic_json(out / "preprocessing.json", prep)
        pp = legacy("03-machine-learning/01-features/05_preprocessar_matrizes.py")
        write_csv(out / "transformations.csv", [{"feature": f, "transformation":
                  "conditional_robust_scaler_train" if f in pp.CONDITIONAL_SCALING else "robust_scaler_train" if f in prep["parameters"] else "passthrough"}
                  for f in prep["matrices"]["C"]])
        for method in ("AE", "IF"):
            directory = out / method
            directory.mkdir(parents=True, exist_ok=True)
            rates, statistics, thresholds, training, histories, type4 = [], [], [], [], [], []
            for config in configs:
                fit = read_json(store.destination(f"final/{method}/{config['key']}") / "model.json")
                train_row = {"configuration": config["key"], "matrix": config["matrix"], "family": config["family"],
                             "features": len(config["features"]), "fit_seconds": fit["fit_seconds"],
                             "training_rows_fit": fit["fit_rows"], "internal_validation_rows": fit.get("internal_rows", 0)}
                train_row.update({"best_epoch": fit.get("checkpoint_epoch"), "epochs_executed": fit.get("epochs"),
                                  "best_internal_validation_loss": fit.get("checkpoint_loss"), "device": fit.get("device", "cpu"),
                                  "peak_gpu_memory_bytes": fit.get("peak_gpu_memory_bytes", 0)})
                training.append(train_row)
                histories.extend({"configuration": config["key"], **r} for r in fit["history"])
                for split in splits:
                    stage = store.destination(f"scores/{split['key']}/{method}/{config['key']}")
                    summary = read_json(stage / "summary.json")
                    path = directory / "escores" / config["key"] / f"{split['order']:02d}_{split['key']}.parquet"
                    link_artifact(stage / "scores.parquet", path)
                    base = {"split_order": split["order"], "split": split["key"], "role": split["role"],
                            "configuration": config["key"], "matrix": config["matrix"], "rows": summary["rows"]}
                    statistics.append({**base, **summary["statistics"]})
                    rates.append({**base, **{f"anomalies_{k}": v for k, v in summary["counts"].items()},
                                  **{f"anomaly_rate_{k}": v for k, v in summary["rates"].items()}})
                    if split["role"] == "validation":
                        thresholds.append({"configuration": config["key"], "calibration_split": split["key"],
                                           "validation_rows": summary["rows"], **summary["thresholds"]})
                    for typ, n, flagged in con.execute(f"SELECT m.is_type_4, count(*), sum(CAST(s.is_anomaly_q995 AS INTEGER)) FROM {parquet(path)} s JOIN {parquet(metadata(split))} m USING(row_id) GROUP BY 1 ORDER BY 1").fetchall():
                        type4.append({**{k: base[k] for k in ("split_order", "split", "role", "configuration")},
                                      "is_type_4": int(typ), "rows": n, "anomalies_q995": flagged, "anomaly_rate_q995": flagged / n})
            write_csv(directory / "02_resumo_treinamento.csv", training)
            if method == "AE":
                names = ("04_limiares_validacao.csv", "05_estatisticas_escores.csv", "06_taxas_anomalias_recortes.csv", "07_taxas_anomalias_tipo4.csv")
                write_csv(directory / "03_historico_treinamento.csv", histories)
            else:
                names = ("03_limiares_validacao.csv", "04_estatisticas_escores.csv", "05_taxas_anomalias_recortes.csv", "06_taxas_anomalias_tipo4.csv")
            for filename, records in zip(names, (thresholds, statistics, rates, type4)):
                write_csv(directory / filename, records)
            atomic_json(directory / ("13_manifest_autoencoder.json" if method == "AE" else "10_manifest_isolation_forest.json"),
                        {"configurations": list(CONFIGURATIONS), "preprocessing_signature": prep["preprocessing_signature"],
                         "experiment_signature": store.identity, "scores_directory": str(directory / "escores"),
                         "source_split_manifest": str(Path(source) / "03-machine-learning/01-features/resultados-splits/00_manifest_splits_temporais.json")})
            atomic_json(directory / "01_configuracoes_experimento.json", {"configurations": configs,
                        "selected_hyperparameters": read_json(store.destination("selection") / "winners.json")[method]})
        cross = []
        for config in CONFIGURATIONS:
            for split in splits:
                filename = f"{split['order']:02d}_{split['key']}.parquet"
                a, f = (parquet(out / method / "escores" / config / filename) for method in ("AE", "IF"))
                n, na, nf, both, union = con.execute(f"SELECT count(*), sum(CAST(a.is_anomaly_q995 AS INTEGER)), sum(CAST(f.is_anomaly_q995 AS INTEGER)), sum(CAST(a.is_anomaly_q995 AND f.is_anomaly_q995 AS INTEGER)), sum(CAST(a.is_anomaly_q995 OR f.is_anomaly_q995 AS INTEGER)) FROM {a} a JOIN {f} f USING(row_id)").fetchone()
                if n != split["sample_rows"]:
                    raise ValueError("Cobertura divergente na concordância")
                cross.append({"split_order": split["order"], "split": split["key"], "role": split["role"], "configuration": config,
                              "ae_anomalies": na, "if_anomalies": nf, "intersection": both, "union": union,
                              "jaccard": both / union if union else 1.})
        write_csv(out / "AE/09_concordancia_ae_if.csv", cross)


def legacy_reports(source, store):
    require(store, "exports")
    root = store.destination("exports")
    agreement_figures(source, store)
    stages = [
        ("external-labelcloud", "05-analise-resultados/03_avaliar_labelcloud_hipotetico.py",
         ["--labels", str(Path(source) / "04-pos-processamento/resultados-labelcloud/05_rotulos_binarios_hipoteticos_labelcloud.parquet"),
          "--split-manifest", str(Path(source) / "03-machine-learning/01-features/resultados-splits/00_manifest_splits_temporais.json"),
          "--ae-results-dir", str(root / "AE"), "--if-results-dir", str(root / "IF")]),
        ("drift", "03-machine-learning/01-features/06_analisar_drift_temporal.py",
         ["--split-manifest", str(root / "splits.json"), "--preprocessing-manifest", str(root / "preprocessing.json"),
          "--transformation-dictionary", str(root / "transformations.csv")]),
    ]
    for name, script, args in stages:
        if store.complete(name):
            continue
        with store.stage(name) as out:
            command = [sys.executable, str(ROOT / script), *args, "--output-dir", str(out),
                       "--temp-dir", str(out / ".tmp"), "--threads", "4"]
            subprocess.run(command, check=True, env={**os.environ, "MPLBACKEND": "Agg"})
            atomic_json(out / "invocation.json", {"script": script, "arguments": args,
                        "note": "Análise local revisada; teste histórico previamente consultado. Sem RPC."})
    if not store.complete("correlation-figures"):
        with store.stage("correlation-figures") as out:
            for filename in ("09_correlacao_pearson.csv", "10_correlacao_spearman.csv", "15_resultado_execucao.json"):
                shutil.copyfile(Path(source) / "03-machine-learning/01-features/resultados" / filename, out / filename)
            subprocess.run([sys.executable, str(ROOT / "03-machine-learning/01-features/04_gerar_figuras_correlacao_pdf.py"),
                            "--results-dir", str(out)], check=True, env={**os.environ, "MPLBACKEND": "Agg"})


def agreement_figures(source, store):
    if store.complete("figures-agreement"):
        return
    require(store, "exports")
    root = store.destination("exports")
    _, splits = load_source(source)
    module = legacy("05-analise-resultados/01_analisar_ae_if.py")
    def records(path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    a = {(r["configuration"], r["split"]): r for r in records(root / "AE/06_taxas_anomalias_recortes.csv")}
    f = {(r["configuration"], r["split"]): r for r in records(root / "IF/05_taxas_anomalias_recortes.csv")}
    combined = []
    for row in records(root / "AE/09_concordancia_ae_if.csv"):
        key = row["configuration"], row["split"]
        combined.append({**row, "ae_rate_q995": float(a[key]["anomaly_rate_q995"]),
                         "if_rate_q995": float(f[key]["anomaly_rate_q995"]), "jaccard": float(row["jaccard"])})
    with store.stage("figures-agreement") as out, connect(out / ".tmp") as con:
        write_csv(out / "01_resumo_integrado_ae_if.csv", combined)
        module.plot_if_quantiles(out / "if_quantiles.pdf", out / "if_quantiles.png",
                                 records(root / "IF/04_estatisticas_escores.csv"), list(CONFIGURATIONS), splits)
        module.plot_if_survival(con, out / "if_survival.pdf", out / "if_survival.png", root / "IF", list(CONFIGURATIONS), splits, 100000)
        module.plot_rate_heatmaps(out / "rates.pdf", out / "rates.png", combined, list(CONFIGURATIONS), splits)
        module.plot_agreement(out / "agreement.pdf", out / "agreement.png", combined, list(CONFIGURATIONS), splits)
        atomic_json(out / "scope.json", {"new_semantic_candidates": False,
                    "note": "Relatórios de escores não regeneram a referência semântica nem solicitam RPC."})
