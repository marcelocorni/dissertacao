# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
#   "matplotlib>=3.10,<4",
#   "numpy>=2.1,<3",
#   "scikit-learn>=1.6,<2",
# ]
# ///

"""Avalia AE e IF sob uma hipótese binária experimental derivada do Label Cloud."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


ASSUMPTION = (
    "Hipótese experimental: endereços MEV Bot do Label Cloud são positivos; "
    "todos os endereços ausentes são negativos."
)
TOP_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
THRESHOLDS = ("q990", "q995", "q999")


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def classification_metrics(y: np.ndarray, flag: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    balanced_accuracy = (recall + specificity) / 2
    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": precision, "recall_sensitivity": recall,
        "specificity": specificity, "npv": npv, "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": float(matthews_corrcoef(y, flag)),
        "false_positive_rate": 1 - specificity,
        "predicted_positive_rate": float(np.mean(flag)),
    }


def top_metrics(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float | int]:
    n = len(y)
    top_n = max(1, math.ceil(n * fraction))
    selected = np.argpartition(scores, -top_n)[-top_n:]
    positives = int(y[selected].sum())
    precision = positives / top_n
    recall = positives / int(y.sum()) if y.sum() else 0.0
    prevalence = float(y.mean())
    return {
        "top_fraction": fraction,
        "top_n": top_n,
        "positives_in_top": positives,
        "precision_at_top": precision,
        "recall_at_top": recall,
        "lift_at_top": precision / prevalence if prevalence else float("nan"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap(rows: list[dict[str, object]], splits: list[str], models: list[str], metric: str, title: str, stem: Path) -> None:
    lookup = {(str(r["model"]), str(r["split"])): float(r[metric]) for r in rows}
    matrix = np.array([[lookup.get((model, split), np.nan) for split in splits] for model in models])
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(splits)), [s.replace("_", "\n") for s in splits], fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if value < 0.35 or value > 0.75 else "black")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_figure(fig, stem)


def main() -> int:
    here = Path(__file__).resolve().parent
    src_root = here.parent
    parser = argparse.ArgumentParser(description="Avaliação hipotética AE/IF com Label Cloud.")
    parser.add_argument(
        "--labels", type=Path,
        default=here.parent / "04-pos-processamento" / "resultados-labelcloud" / "05_rotulos_binarios_hipoteticos_labelcloud.parquet",
    )
    parser.add_argument("--ae-results-dir", type=Path, default=here.parent / "03-machine-learning" / "02-ae" / "resultados")
    parser.add_argument("--if-results-dir", type=Path, default=here.parent / "03-machine-learning" / "03-if" / "resultados")
    parser.add_argument("--split-manifest", type=Path, default=here.parent / "03-machine-learning" / "01-features" / "resultados-splits" / "00_manifest_splits_temporais.json")
    parser.add_argument("--output-dir", type=Path, default=here / "resultados-labelcloud-hipotetico")
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    labels = args.labels.resolve()
    if not labels.is_file():
        parser.error(f"Rótulos não encontrados: {labels}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    split_records = json.loads(args.split_manifest.read_text(encoding="utf-8"))["splits"]
    configurations = list(json.loads((args.ae_results_dir / "13_manifest_autoencoder.json").read_text(encoding="utf-8"))["configurations"])
    split_keys = [str(item["key"]) for item in split_records]
    model_names = [f"{method}:{config}" for config in configurations for method in ("AE", "IF")]

    con = duckdb.connect()
    con.execute(f"SET memory_limit={sql_quote(args.memory_limit)}")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory={sql_quote(args.temp_dir.resolve())}")
    labels_relation = f"read_parquet({sql_quote(labels.as_posix())})"

    metrics_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    test_confusions: dict[tuple[str, str], dict[str, float | int]] = {}
    test_percentiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rng = np.random.default_rng(20260918)

    for split_record in split_records:
        split = str(split_record["key"])
        prefix = f"{int(split_record['order']):02d}_{split}.parquet"
        curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]] = []
        print(f"Avaliando {split}...")
        for configuration in configurations:
            for method, root, score_column in (
                ("AE", args.ae_results_dir, "reconstruction_error"),
                ("IF", args.if_results_dir, "anomaly_score"),
            ):
                model = f"{method}:{configuration}"
                score_file = root / "escores" / configuration / prefix
                query = f"""
                    SELECT l.label::UTINYINT AS label,
                           s.{score_column}::FLOAT AS score,
                           s.is_anomaly_q990, s.is_anomaly_q995, s.is_anomaly_q999
                    FROM {labels_relation} l
                    INNER JOIN read_parquet({sql_quote(score_file.as_posix())}) s USING (row_id)
                    WHERE l.split = {sql_quote(split)}
                """
                arrays = con.execute(query).fetchnumpy()
                y = np.asarray(arrays["label"], dtype=np.uint8)
                scores = np.asarray(arrays["score"], dtype=np.float64)
                if len(y) != int(split_record["sample_rows"]):
                    raise RuntimeError(f"Cobertura incompleta em {split}/{model}: {len(y)}")
                positives = int(y.sum())
                roc_auc = float(roc_auc_score(y, scores))
                average_precision = float(average_precision_score(y, scores))
                fpr, tpr, _ = roc_curve(y, scores)
                pr_precision, pr_recall, _ = precision_recall_curve(y, scores)
                curves.append((model, fpr, tpr, pr_recall, pr_precision, roc_auc, average_precision))
                metrics_rows.append({
                    "split": split,
                    "split_role": split_record["role"],
                    "configuration": configuration,
                    "method": method,
                    "model": model,
                    "rows": len(y),
                    "positives": positives,
                    "negatives": len(y) - positives,
                    "prevalence": positives / len(y),
                    "roc_auc": roc_auc,
                    "pr_auc_average_precision": average_precision,
                    "assumption": ASSUMPTION,
                })
                for threshold in THRESHOLDS:
                    values = classification_metrics(y, np.asarray(arrays[f"is_anomaly_{threshold}"], dtype=bool))
                    row = {"split": split, "configuration": configuration, "method": method, "model": model, "threshold": threshold, **values}
                    threshold_rows.append(row)
                    if split == "teste_final_2025":
                        test_confusions[(threshold, model)] = values
                for fraction in TOP_FRACTIONS:
                    top_rows.append({"split": split, "configuration": configuration, "method": method, "model": model, **top_metrics(y, scores, fraction)})

                if split == "teste_final_2025":
                    negative_scores = scores[y == 0]
                    positive_scores = scores[y == 1]
                    if len(negative_scores) > 200_000:
                        negative_scores = rng.choice(negative_scores, 200_000, replace=False)
                    if len(positive_scores) > 200_000:
                        positive_scores = rng.choice(positive_scores, 200_000, replace=False)
                    reference = np.sort(negative_scores)
                    negative_percentiles = np.searchsorted(reference, negative_scores, side="right") / len(reference)
                    positive_percentiles = np.searchsorted(reference, positive_scores, side="right") / len(reference)
                    test_percentiles[model] = (negative_percentiles, positive_percentiles)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for model, fpr, tpr, recall, precision, roc_auc, average_precision in curves:
            axes[0].plot(fpr, tpr, linewidth=1, label=f"{model} (AUC={roc_auc:.3f})")
            axes[1].plot(recall, precision, linewidth=1, label=f"{model} (AP={average_precision:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[0].set(title="Curvas ROC", xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos")
        axes[1].set(title="Curvas Precision–Recall", xlabel="Recall", ylabel="Precisão")
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=6, loc="lower right")
        axes[1].legend(fontsize=6, loc="upper right")
        fig.suptitle(f"Avaliação hipotética — {split}")
        save_figure(fig, output / f"05_curvas_roc_pr_{split}")

    con.close()
    write_csv(output / "01_metricas_discriminacao.csv", metrics_rows)
    write_csv(output / "02_metricas_limiares.csv", threshold_rows)
    write_csv(output / "03_metricas_top_k.csv", top_rows)

    heatmap(metrics_rows, split_keys, model_names, "roc_auc", "ROC-AUC por recorte e modelo", output / "06_heatmap_roc_auc")
    heatmap(metrics_rows, split_keys, model_names, "pr_auc_average_precision", "PR-AUC (Average Precision) por recorte e modelo", output / "07_heatmap_pr_auc")

    for threshold in THRESHOLDS:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
        for axis, model in zip(axes.flat, model_names):
            values = test_confusions[(threshold, model)]
            matrix = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
            axis.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    axis.text(
                        j, i, f"{matrix[i, j]:,}".replace(",", "."),
                        ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() * 0.5 else "black",
                    )
            axis.set(title=model, xticks=[0, 1], xticklabels=["Pred. 0", "Pred. 1"], yticks=[0, 1], yticklabels=["Real 0", "Real 1"])
        fig.suptitle(f"Matrizes de confusão — teste_final_2025 — limiar {threshold}")
        save_figure(fig, output / f"08_matrizes_confusao_teste_{threshold}")

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    bins = np.linspace(0, 1, 31)
    for axis, model in zip(axes.flat, model_names):
        negative, positive = test_percentiles[model]
        axis.hist(negative, bins=bins, density=True, alpha=0.45, label="Classe 0")
        axis.hist(positive, bins=bins, density=True, alpha=0.65, label="MEV Bot")
        axis.set(title=model, xlabel="Percentil do escore vs. classe 0", ylabel="Densidade")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle("Distribuição dos escores — teste_final_2025")
    save_figure(fig, output / "09_distribuicao_escores_teste")

    test_top = [row for row in top_rows if row["split"] == "teste_final_2025"]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for model in model_names:
        selected = sorted((row for row in test_top if row["model"] == model), key=lambda r: float(r["top_fraction"]))
        ax.plot([100 * float(r["top_fraction"]) for r in selected], [float(r["lift_at_top"]) for r in selected], marker="o", label=model)
    ax.set(title="Lift no topo dos escores — teste_final_2025", xlabel="Topo selecionado (%)", ylabel="Lift sobre a prevalência")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    save_figure(fig, output / "10_lift_top_k_teste")

    validation_rows = [row for row in metrics_rows if row["split"] == "validacao_2025_pre_pectra"]
    test_rows = [row for row in metrics_rows if row["split"] == "teste_final_2025"]
    test_threshold = [row for row in threshold_rows if row["split"] == "teste_final_2025" and row["threshold"] == "q995"]
    test_top_one = [row for row in top_rows if row["split"] == "teste_final_2025" and float(row["top_fraction"]) == 0.01]
    best_validation_roc = max(validation_rows, key=lambda row: float(row["roc_auc"]))
    best_test_roc = max(test_rows, key=lambda row: float(row["roc_auc"]))
    best_test_ap = max(test_rows, key=lambda row: float(row["pr_auc_average_precision"]))
    best_test_f1 = max(test_threshold, key=lambda row: float(row["f1"]))
    best_test_fixed_threshold = max(
        (row for row in threshold_rows if row["split"] == "teste_final_2025"),
        key=lambda row: float(row["f1"]),
    )
    best_test_lift = max(test_top_one, key=lambda row: float(row["lift_at_top"]))
    test_prevalence = float(test_rows[0]["prevalence"])
    report = f"""# Síntese da avaliação hipotética com Label Cloud

## Hipótese do experimento

{ASSUMPTION} Esta hipótese viabiliza as métricas supervisionadas abaixo, mas não
transforma a ausência de tag em evidência empírica de ausência de front-running.

## População avaliada

- sete recortes cronológicos;
- cinco configurações de features;
- Autoencoder e Isolation Forest;
- avaliação sobre todos os 9.494.450 registros amostrados;
- prevalência hipotética no teste final: {100 * test_prevalence:.4f}%.

## Principais resultados

- melhor ROC-AUC na validação: **{best_validation_roc['model']}**, {float(best_validation_roc['roc_auc']):.4f};
- melhor ROC-AUC no teste final: **{best_test_roc['model']}**, {float(best_test_roc['roc_auc']):.4f};
- melhor PR-AUC no teste final: **{best_test_ap['model']}**, {float(best_test_ap['pr_auc_average_precision']):.4f};
- melhor F1 no teste final com limiar q995: **{best_test_f1['model']}**, {float(best_test_f1['f1']):.4f}
  (precisão {float(best_test_f1['precision']):.4f}, recall {float(best_test_f1['recall_sensitivity']):.4f},
  especificidade {float(best_test_f1['specificity']):.4f}, MCC {float(best_test_f1['mcc']):.4f});
- melhor F1 entre os três limiares predefinidos: **{best_test_fixed_threshold['model']}** em
  **{best_test_fixed_threshold['threshold']}**, {float(best_test_fixed_threshold['f1']):.4f}
  (precisão {float(best_test_fixed_threshold['precision']):.4f},
  recall {float(best_test_fixed_threshold['recall_sensitivity']):.4f},
  MCC {float(best_test_fixed_threshold['mcc']):.4f});
- maior lift no top 1% do teste: **{best_test_lift['model']}**, {float(best_test_lift['lift_at_top']):.2f}×
  (precisão {float(best_test_lift['precision_at_top']):.4f}, recall {float(best_test_lift['recall_at_top']):.4f}).

## Leitura metodológica

ROC-AUC mede ordenação global e pode parecer elevada em bases desbalanceadas.
PR-AUC, precisão, recall, F1, MCC e lift devem receber maior peso na discussão.
Os limiares q990, q995 e q999 foram calibrados anteriormente sem esses rótulos;
portanto, as matrizes de confusão avaliam a política original e não um limiar
otimizado retrospectivamente para o Label Cloud.

Os resultados do teste final devem permanecer como avaliação final. A validação
pode orientar a escolha de configuração, mas o teste não deve ser usado para
reajustar o modelo ou o limiar.
"""
    (output / "11_sintese_resultados.md").write_text(report, encoding="utf-8")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "hypothetical_supervised_evaluation",
        "assumption": ASSUMPTION,
        "labels_file": str(labels),
        "splits": split_keys,
        "configurations": configurations,
        "methods": ["AE", "IF"],
        "thresholds": list(THRESHOLDS),
        "top_fractions": list(TOP_FRACTIONS),
        "primary_test_split": "teste_final_2025",
        "metric_note": "ROC/PR and confusion metrics are valid only under the stated hypothetical class-0 assumption.",
        "artifacts": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "04_manifest_avaliacao.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "metric_rows": len(metrics_rows), "threshold_rows": len(threshold_rows), "top_k_rows": len(top_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
