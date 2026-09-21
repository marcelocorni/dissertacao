# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
#   "matplotlib>=3.10,<4",
#   "numpy>=2.1,<3",
#   "scikit-learn>=1.6,<2",
# ]
# ///

"""Avalia AE e IF contra os rótulos semânticos consolidados.

A população é restrita aos papéis dos eventos adjudicados. Transações sem
rótulo semântico não são convertidas em negativos. São avaliadas duas políticas:
estrita (somente eventos confirmados) e sensibilidade (confirmados ou prováveis).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib

matplotlib.use("Agg")
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


POLICIES = {
    "strict": {
        "label": "strict_label",
        "conflict": "strict_conflict",
        "description": "Somente eventos adjudicados como confirmado.",
    },
    "sensitivity": {
        "label": "sensitivity_label",
        "conflict": "sensitivity_conflict",
        "description": "Eventos adjudicados como confirmado ou provável.",
    },
}
THRESHOLDS = ("q990", "q995", "q999")
TOP_FRACTIONS = (0.01, 0.05, 0.10)
ATTACK_SCOPES = ("all", "insertion", "displacement")
PRIMARY_VALIDATION_SPLIT = "validacao_2025_pre_pectra"
PRIMARY_TEST_SPLIT = "teste_final_2025"


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Nenhuma linha produzida para {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def classification_metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": float(matthews_corrcoef(y, predicted)),
        "false_positive_rate": 1 - specificity,
        "predicted_positive_rate": float(np.mean(predicted)),
    }


def top_metrics(y: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float | int]:
    top_n = max(1, math.ceil(len(y) * fraction))
    selected = np.argsort(scores, kind="stable")[-top_n:]
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


def bootstrap_intervals(
    y: np.ndarray,
    scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    if repetitions <= 0:
        return {
            "roc_auc_ci_low": float("nan"),
            "roc_auc_ci_high": float("nan"),
            "pr_auc_ci_low": float("nan"),
            "pr_auc_ci_high": float("nan"),
        }
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    roc_values = np.empty(repetitions, dtype=np.float64)
    pr_values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = np.concatenate(
            [
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            ]
        )
        roc_values[index] = roc_auc_score(y[sample], scores[sample])
        pr_values[index] = average_precision_score(y[sample], scores[sample])
    return {
        "roc_auc_ci_low": float(np.quantile(roc_values, 0.025)),
        "roc_auc_ci_high": float(np.quantile(roc_values, 0.975)),
        "pr_auc_ci_low": float(np.quantile(pr_values, 0.025)),
        "pr_auc_ci_high": float(np.quantile(pr_values, 0.975)),
    }


def metric_seed(base: int, *parts: str) -> int:
    text = "::".join(parts).encode("utf-8")
    return (base + int.from_bytes(hashlib.sha256(text).digest()[:4], "big")) % (2**32)


def subset_for_scope(
    attack_types: np.ndarray,
    scope: str,
) -> np.ndarray:
    if scope == "all":
        return np.ones(len(attack_types), dtype=bool)
    return attack_types == scope


def resolve_metadata(split_record: dict[str, object], split_manifest: Path) -> Path:
    recorded = Path(str(split_record["output_directory"])) / "05_metadados.parquet"
    if recorded.is_file():
        return recorded.resolve()
    features_root = split_manifest.resolve().parent.parent
    if str(split_record["role"]) == "train":
        fallback = features_root / "resultados-matrizes" / "05_metadados.parquet"
    else:
        fallback = split_manifest.resolve().parent / str(split_record["directory_name"]) / "05_metadados.parquet"
    if not fallback.is_file():
        raise FileNotFoundError(f"Metadados não encontrados para {split_record['key']}: {fallback}")
    return fallback


def score_path(root: Path, configuration: str, split_record: dict[str, object]) -> Path:
    path = root / "escores" / configuration / (
        f"{int(split_record['order']):02d}_{split_record['key']}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de escores não encontrado: {path}")
    return path.resolve()


def arrays_from_query(connection: duckdb.DuckDBPyConnection, query: str) -> dict[str, np.ndarray]:
    result = connection.execute(query).fetchnumpy()
    return {name: np.asarray(values) for name, values in result.items()}


def build_coverage(
    connection: duckdb.DuckDBPyConnection,
    split: str,
    expected_source_rows: int,
) -> list[dict[str, object]]:
    row = connection.execute(
        """
        SELECT
            count(*)::BIGINT,
            count(*) FILTER (WHERE strict_label = 1 AND strict_conflict = 0)::BIGINT,
            count(*) FILTER (WHERE strict_label = 0 AND strict_conflict = 0)::BIGINT,
            count(*) FILTER (WHERE strict_label IS NULL AND strict_conflict = 0)::BIGINT,
            count(*) FILTER (WHERE strict_conflict = 1)::BIGINT,
            count(*) FILTER (WHERE sensitivity_label = 1 AND sensitivity_conflict = 0)::BIGINT,
            count(*) FILTER (WHERE sensitivity_label = 0 AND sensitivity_conflict = 0)::BIGINT,
            count(*) FILTER (WHERE sensitivity_label IS NULL AND sensitivity_conflict = 0)::BIGINT,
            count(*) FILTER (WHERE sensitivity_conflict = 1)::BIGINT
        FROM label_base
        """
    ).fetchone()
    total = int(row[0])
    if total != expected_source_rows:
        raise RuntimeError(
            f"Cobertura de metadados incompleta em {split}: "
            f"rótulos={expected_source_rows}, vinculados={total}"
        )
    rows: list[dict[str, object]] = []
    values = {
        "strict": (row[1], row[2], row[3], row[4]),
        "sensitivity": (row[5], row[6], row[7], row[8]),
    }
    for policy, (positive, negative, unlabeled, conflicts) in values.items():
        evaluable = int(positive) + int(negative)
        rows.append(
            {
                "split": split,
                "policy": policy,
                "semantic_transactions": total,
                "evaluable_transactions": evaluable,
                "positive_attackers": int(positive),
                "contextual_negative_victims": int(negative),
                "unlabeled_in_policy": int(unlabeled),
                "conflicting_excluded": int(conflicts),
                "evaluable_share": evaluable / total if total else 0.0,
            }
        )
    return rows


def paired_metrics(
    event_keys: np.ndarray,
    attack_types: np.ndarray,
    roles: np.ndarray,
    scores: np.ndarray,
    scope: str,
) -> dict[str, float | int]:
    events: dict[str, dict[str, object]] = defaultdict(
        lambda: {"attackers": [], "victims": [], "attack_type": ""}
    )
    for event_key, attack_type, role, score in zip(
        event_keys, attack_types, roles, scores, strict=True
    ):
        item = events[str(event_key)]
        item["attack_type"] = str(attack_type)
        target = "victims" if str(role) == "victim" else "attackers"
        item[target].append(float(score))

    pair_margins: list[float] = []
    all_attackers_above: list[bool] = []
    valid_events = 0
    for item in events.values():
        if scope != "all" and item["attack_type"] != scope:
            continue
        attackers = list(item["attackers"])
        victims = list(item["victims"])
        if not attackers or not victims:
            continue
        valid_events += 1
        margins = [attacker - victim for attacker in attackers for victim in victims]
        pair_margins.extend(margins)
        all_attackers_above.append(all(value > 0 for value in margins))

    margins_array = np.asarray(pair_margins, dtype=np.float64)
    if not valid_events or not len(margins_array):
        return {
            "events": 0,
            "attacker_victim_pairs": 0,
            "attacker_score_higher": 0,
            "equal_scores": 0,
            "pairwise_superiority_rate": float("nan"),
            "events_all_attackers_above_victim": 0,
            "event_superiority_rate": float("nan"),
            "median_score_margin": float("nan"),
            "mean_score_margin": float("nan"),
        }
    higher = int(np.sum(margins_array > 0))
    ties = int(np.sum(margins_array == 0))
    all_count = int(np.sum(all_attackers_above))
    return {
        "events": valid_events,
        "attacker_victim_pairs": int(len(margins_array)),
        "attacker_score_higher": higher,
        "equal_scores": ties,
        "pairwise_superiority_rate": higher / len(margins_array),
        "events_all_attackers_above_victim": all_count,
        "event_superiority_rate": all_count / valid_events,
        "median_score_margin": float(np.median(margins_array)),
        "mean_score_margin": float(np.mean(margins_array)),
    }


def plot_curves(
    curves: dict[tuple[str, str], list[dict[str, object]]],
    output: Path,
) -> None:
    for (policy, split), records in curves.items():
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
        for record in records:
            model = str(record["model"])
            axes[0].plot(
                record["fpr"], record["tpr"], linewidth=1.2,
                label=f"{model} ({float(record['roc_auc']):.3f})",
            )
            axes[1].plot(
                record["recall"], record["precision"], linewidth=1.2,
                label=f"{model} ({float(record['pr_auc']):.3f})",
            )
        axes[0].plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1)
        axes[0].set(
            title="Curvas ROC",
            xlabel="Taxa de falsos positivos",
            ylabel="Taxa de verdadeiros positivos",
        )
        axes[1].set(
            title="Curvas Precision–Recall",
            xlabel="Recall",
            ylabel="Precisão",
        )
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        fig.suptitle(f"Rótulos semânticos {policy} — {split}")
        save_figure(fig, output / f"07_curvas_roc_pr_{policy}_{split}")


def heatmap(
    rows: Iterable[dict[str, object]],
    splits: list[str],
    models: list[str],
    metric: str,
    title: str,
    stem: Path,
    row_key: str = "model",
) -> None:
    rows_list = list(rows)
    lookup = {
        (str(row[row_key]), str(row["split"])): float(row[metric])
        for row in rows_list
    }
    matrix = np.array(
        [[lookup.get((model, split), np.nan) for split in splits] for model in models]
    )
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(
        range(len(splits)), [item.replace("_", "\n") for item in splits], fontsize=8
    )
    ax.set_yticks(range(len(models)), models, fontsize=8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                ax.text(
                    column_index, row_index, f"{value:.3f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if value < 0.35 or value > 0.75 else "black",
                )
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_figure(fig, stem)


def plot_confusions(
    threshold_rows: list[dict[str, object]],
    policy: str,
    models: list[str],
    output: Path,
) -> None:
    lookup = {
        str(row["model"]): row
        for row in threshold_rows
        if row["policy"] == policy
        and row["split"] == PRIMARY_TEST_SPLIT
        and row["attack_scope"] == "all"
        and row["threshold"] == "q995"
    }
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for axis, model in zip(axes.flat, models, strict=True):
        values = lookup[model]
        matrix = np.array(
            [[values["tn"], values["fp"]], [values["fn"], values["tp"]]],
            dtype=np.int64,
        )
        axis.imshow(matrix, cmap="Blues")
        maximum = max(1, int(matrix.max()))
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:,}".replace(",", "."),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[row_index, column_index] > maximum * 0.5 else "black",
                )
        axis.set(
            title=model,
            xticks=[0, 1],
            xticklabels=["Pred. vítima", "Pred. atacante"],
            yticks=[0, 1],
            yticklabels=["Vítima", "Atacante"],
        )
    fig.suptitle(f"Teste final — política {policy} — limiar q995")
    save_figure(fig, output / f"10_matrizes_confusao_teste_q995_{policy}")


def select_by_validation(
    discrimination_rows: list[dict[str, object]],
    threshold_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected_rows: list[dict[str, object]] = []
    for policy in POLICIES:
        for method in ("AE", "IF"):
            candidates = [
                row
                for row in discrimination_rows
                if row["policy"] == policy
                and row["method"] == method
                and row["split"] == PRIMARY_VALIDATION_SPLIT
                and row["attack_scope"] == "all"
            ]
            selected = max(candidates, key=lambda row: float(row["pr_auc_average_precision"]))
            model = str(selected["model"])
            test = next(
                row
                for row in discrimination_rows
                if row["policy"] == policy
                and row["model"] == model
                and row["split"] == PRIMARY_TEST_SPLIT
                and row["attack_scope"] == "all"
            )
            q995 = next(
                row
                for row in threshold_rows
                if row["policy"] == policy
                and row["model"] == model
                and row["split"] == PRIMARY_TEST_SPLIT
                and row["attack_scope"] == "all"
                and row["threshold"] == "q995"
            )
            validation_threshold = max(
                (
                    row
                    for row in threshold_rows
                    if row["policy"] == policy
                    and row["model"] == model
                    and row["split"] == PRIMARY_VALIDATION_SPLIT
                    and row["attack_scope"] == "all"
                ),
                key=lambda row: float(row["f1"]),
            )
            selected_threshold_test = next(
                row
                for row in threshold_rows
                if row["policy"] == policy
                and row["model"] == model
                and row["split"] == PRIMARY_TEST_SPLIT
                and row["attack_scope"] == "all"
                and row["threshold"] == validation_threshold["threshold"]
            )
            paired = next(
                row
                for row in paired_rows
                if row["policy"] == policy
                and row["model"] == model
                and row["split"] == PRIMARY_TEST_SPLIT
                and row["attack_scope"] == "all"
            )
            selected_rows.append(
                {
                    "policy": policy,
                    "method": method,
                    "selected_configuration": selected["configuration"],
                    "selection_rule": "maior PR-AUC na validação pré-Pectra",
                    "validation_rows": selected["rows"],
                    "validation_prevalence": selected["prevalence"],
                    "validation_roc_auc": selected["roc_auc"],
                    "validation_pr_auc": selected["pr_auc_average_precision"],
                    "test_rows": test["rows"],
                    "test_prevalence": test["prevalence"],
                    "test_roc_auc": test["roc_auc"],
                    "test_roc_auc_ci_low": test["roc_auc_ci_low"],
                    "test_roc_auc_ci_high": test["roc_auc_ci_high"],
                    "test_pr_auc": test["pr_auc_average_precision"],
                    "test_pr_auc_ci_low": test["pr_auc_ci_low"],
                    "test_pr_auc_ci_high": test["pr_auc_ci_high"],
                    "test_q995_precision": q995["precision"],
                    "test_q995_recall": q995["recall_sensitivity"],
                    "test_q995_specificity": q995["specificity"],
                    "test_q995_f1": q995["f1"],
                    "test_q995_mcc": q995["mcc"],
                    "selected_threshold": validation_threshold["threshold"],
                    "threshold_selection_status": (
                        "selected_by_validation_f1"
                        if float(validation_threshold["f1"]) > 0
                        else "no_positive_predictions_in_validation"
                    ),
                    "validation_selected_threshold_f1": validation_threshold["f1"],
                    "test_selected_threshold_precision": selected_threshold_test["precision"],
                    "test_selected_threshold_recall": selected_threshold_test["recall_sensitivity"],
                    "test_selected_threshold_specificity": selected_threshold_test["specificity"],
                    "test_selected_threshold_f1": selected_threshold_test["f1"],
                    "test_selected_threshold_mcc": selected_threshold_test["mcc"],
                    "test_selected_threshold_tn": selected_threshold_test["tn"],
                    "test_selected_threshold_fp": selected_threshold_test["fp"],
                    "test_selected_threshold_fn": selected_threshold_test["fn"],
                    "test_selected_threshold_tp": selected_threshold_test["tp"],
                    "test_pairwise_superiority": paired["pairwise_superiority_rate"],
                    "test_event_superiority": paired["event_superiority_rate"],
                }
            )
    return selected_rows


def plot_selected_models(selected: list[dict[str, object]], output: Path) -> None:
    labels = [f"{row['policy']}\n{row['method']}:{row['selected_configuration']}" for row in selected]
    metrics = [
        ("test_roc_auc", "ROC-AUC"),
        ("test_pr_auc", "PR-AUC"),
        ("test_pairwise_superiority", "Superioridade pareada"),
        ("test_selected_threshold_f1", "F1 no limiar selecionado"),
    ]
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    for index, (key, label) in enumerate(metrics):
        values = [float(row[key]) for row in selected]
        ax.bar(x + (index - 1.5) * width, values, width, label=label)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Métrica")
    ax.set_title("Modelos selecionados na validação e avaliados no teste final")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    save_figure(fig, output / "12_modelos_selecionados_teste")


def plot_selected_confusions(selected: list[dict[str, object]], output: Path) -> None:
    for policy in POLICIES:
        rows = [row for row in selected if row["policy"] == policy]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for axis, row in zip(axes, rows, strict=True):
            tn = int(row["test_selected_threshold_tn"])
            fp = int(row["test_selected_threshold_fp"])
            fn = int(row["test_selected_threshold_fn"])
            tp = int(row["test_selected_threshold_tp"])
            matrix = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
            axis.imshow(matrix, cmap="Blues")
            maximum = max(1, int(matrix.max()))
            for row_index in range(2):
                for column_index in range(2):
                    axis.text(
                        column_index,
                        row_index,
                        f"{matrix[row_index, column_index]:,}".replace(",", "."),
                        ha="center",
                        va="center",
                        color=(
                            "white"
                            if matrix[row_index, column_index] > maximum * 0.5
                            else "black"
                        ),
                    )
            axis.set(
                title=(
                    f"{row['method']}:{row['selected_configuration']} — "
                    f"{row['selected_threshold']}"
                ),
                xticks=[0, 1],
                xticklabels=["Pred. vítima", "Pred. atacante"],
                yticks=[0, 1],
                yticklabels=["Vítima", "Atacante"],
            )
        fig.suptitle(f"Teste final — modelos e limiares selecionados — {policy}")
        save_figure(fig, output / f"10_matrizes_confusao_teste_selecionado_{policy}")


def main() -> int:
    here = Path(__file__).resolve().parent
    src_root = here.parent
    parser = argparse.ArgumentParser(
        description="Avaliação final de AE e IF com rótulos semânticos adjudicados."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=src_root
        / "04-pos-processamento"
        / "06-validacao-semantica-front-running"
        / "resultados"
        / "consolidado"
        / "03_rotulos_transacao.parquet",
    )
    parser.add_argument(
        "--event-roles",
        type=Path,
        default=src_root
        / "04-pos-processamento"
        / "06-validacao-semantica-front-running"
        / "resultados"
        / "consolidado"
        / "02_papeis_evento.parquet",
    )
    parser.add_argument(
        "--ae-results-dir",
        type=Path,
        default=src_root / "03-machine-learning" / "02-ae" / "resultados",
    )
    parser.add_argument(
        "--if-results-dir",
        type=Path,
        default=src_root / "03-machine-learning" / "03-if" / "resultados",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=src_root
        / "03-machine-learning"
        / "01-features"
        / "resultados-splits"
        / "00_manifest_splits_temporais.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=here / "resultados-rotulos-semanticos"
    )
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite substituir uma avaliação semântica já concluída.",
    )
    args = parser.parse_args()

    required = [args.labels, args.event_roles, args.split_manifest]
    for path in required:
        if not path.resolve().is_file():
            parser.error(f"Entrada não encontrada: {path.resolve()}")
    if args.threads < 1:
        parser.error("--threads deve ser positivo")
    if args.bootstrap_repetitions < 0:
        parser.error("--bootstrap-repetitions não pode ser negativo")

    output = args.output_dir.resolve()
    completed_manifest = output / "15_manifest_avaliacao.json"
    if completed_manifest.exists() and not args.overwrite:
        parser.error(
            f"A avaliação já existe em {output}. Use --overwrite para regenerá-la."
        )
    output.mkdir(parents=True, exist_ok=True)
    args.temp_dir.resolve().mkdir(parents=True, exist_ok=True)

    split_document = json.loads(args.split_manifest.resolve().read_text(encoding="utf-8"))
    split_records = list(split_document["splits"])
    split_keys = [str(item["key"]) for item in split_records]
    ae_manifest = json.loads(
        (args.ae_results_dir.resolve() / "13_manifest_autoencoder.json").read_text(
            encoding="utf-8"
        )
    )
    if_manifest = json.loads(
        (args.if_results_dir.resolve() / "10_manifest_isolation_forest.json").read_text(
            encoding="utf-8"
        )
    )
    configurations = list(ae_manifest["configurations"])
    if configurations != list(if_manifest["configurations"]):
        raise RuntimeError("As configurações AE e IF não coincidem.")
    models = [f"{method}:{configuration}" for configuration in configurations for method in ("AE", "IF")]

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit={sql_quote(args.memory_limit)}")
    connection.execute(f"SET threads={args.threads}")
    connection.execute(f"SET temp_directory={sql_quote(args.temp_dir.resolve())}")
    labels_relation = f"read_parquet({sql_quote(args.labels.resolve().as_posix())})"
    roles_relation = f"read_parquet({sql_quote(args.event_roles.resolve().as_posix())})"

    coverage_rows: list[dict[str, object]] = []
    discrimination_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)

    try:
        for split_record in split_records:
            split = str(split_record["key"])
            metadata = resolve_metadata(split_record, args.split_manifest)
            expected_labels = int(
                connection.execute(
                    f"SELECT count(*) FROM {labels_relation} WHERE split = {sql_quote(split)}"
                ).fetchone()[0]
            )
            connection.execute("DROP TABLE IF EXISTS label_base")
            connection.execute("DROP TABLE IF EXISTS role_base")
            connection.execute(
                f"""
                CREATE TEMP TABLE label_base AS
                SELECT l.*, m.row_id::UBIGINT AS row_id
                FROM {labels_relation} l
                INNER JOIN read_parquet({sql_quote(metadata.as_posix())}) m
                    ON lower(m.hash) = lower(l.tx_hash)
                WHERE l.split = {sql_quote(split)}
                """
            )
            connection.execute(
                f"""
                CREATE TEMP TABLE role_base AS
                SELECT r.*, b.row_id
                FROM {roles_relation} r
                INNER JOIN label_base b USING (tx_hash)
                WHERE r.split = {sql_quote(split)}
                """
            )
            coverage_rows.extend(build_coverage(connection, split, expected_labels))
            print(f"Avaliando {split}...")

            for configuration in configurations:
                for method, root, score_column in (
                    ("AE", args.ae_results_dir.resolve(), "reconstruction_error"),
                    ("IF", args.if_results_dir.resolve(), "anomaly_score"),
                ):
                    model = f"{method}:{configuration}"
                    scores = score_path(root, configuration, split_record)
                    score_relation = f"read_parquet({sql_quote(scores.as_posix())})"

                    for policy, definition in POLICIES.items():
                        label_column = str(definition["label"])
                        conflict_column = str(definition["conflict"])
                        arrays = arrays_from_query(
                            connection,
                            f"""
                            SELECT
                                l.{label_column}::UTINYINT AS label,
                                l.attack_types::VARCHAR AS attack_types,
                                s.{score_column}::DOUBLE AS score,
                                s.is_anomaly_q990::BOOLEAN AS q990,
                                s.is_anomaly_q995::BOOLEAN AS q995,
                                s.is_anomaly_q999::BOOLEAN AS q999
                            FROM label_base l
                            INNER JOIN {score_relation} s USING (row_id)
                            WHERE l.{conflict_column} = 0
                              AND l.{label_column} IS NOT NULL
                            ORDER BY l.tx_hash
                            """,
                        )
                        y_all = np.asarray(arrays["label"], dtype=np.uint8)
                        score_all = np.asarray(arrays["score"], dtype=np.float64)
                        attack_types = np.asarray(arrays["attack_types"], dtype=str)
                        expected_evaluable = next(
                            int(row["evaluable_transactions"])
                            for row in coverage_rows
                            if row["split"] == split and row["policy"] == policy
                        )
                        if len(y_all) != expected_evaluable:
                            raise RuntimeError(
                                f"Cobertura de escores incompleta em {split}/{policy}/{model}: "
                                f"esperado={expected_evaluable}, obtido={len(y_all)}"
                            )

                        for attack_scope in ATTACK_SCOPES:
                            mask = subset_for_scope(attack_types, attack_scope)
                            y = y_all[mask]
                            model_scores = score_all[mask]
                            if len(y) == 0 or len(np.unique(y)) != 2:
                                continue
                            roc_auc = float(roc_auc_score(y, model_scores))
                            pr_auc = float(average_precision_score(y, model_scores))
                            intervals = (
                                bootstrap_intervals(
                                    y,
                                    model_scores,
                                    args.bootstrap_repetitions,
                                    metric_seed(args.seed, split, policy, model),
                                )
                                if attack_scope == "all"
                                else {
                                    "roc_auc_ci_low": float("nan"),
                                    "roc_auc_ci_high": float("nan"),
                                    "pr_auc_ci_low": float("nan"),
                                    "pr_auc_ci_high": float("nan"),
                                }
                            )
                            metric_row = {
                                "split": split,
                                "split_role": split_record["role"],
                                "policy": policy,
                                "attack_scope": attack_scope,
                                "configuration": configuration,
                                "method": method,
                                "model": model,
                                "rows": len(y),
                                "positives": int(y.sum()),
                                "contextual_negatives": int(len(y) - y.sum()),
                                "prevalence": float(y.mean()),
                                "roc_auc": roc_auc,
                                "pr_auc_average_precision": pr_auc,
                                **intervals,
                            }
                            discrimination_rows.append(metric_row)

                            for threshold in THRESHOLDS:
                                predicted = np.asarray(arrays[threshold], dtype=bool)[mask]
                                threshold_rows.append(
                                    {
                                        "split": split,
                                        "policy": policy,
                                        "attack_scope": attack_scope,
                                        "configuration": configuration,
                                        "method": method,
                                        "model": model,
                                        "threshold": threshold,
                                        "rows": len(y),
                                        "positives": int(y.sum()),
                                        "contextual_negatives": int(len(y) - y.sum()),
                                        **classification_metrics(y, predicted),
                                    }
                                )
                            for fraction in TOP_FRACTIONS:
                                top_rows.append(
                                    {
                                        "split": split,
                                        "policy": policy,
                                        "attack_scope": attack_scope,
                                        "configuration": configuration,
                                        "method": method,
                                        "model": model,
                                        **top_metrics(y, model_scores, fraction),
                                    }
                                )

                            if (
                                attack_scope == "all"
                                and split in (PRIMARY_VALIDATION_SPLIT, PRIMARY_TEST_SPLIT)
                            ):
                                fpr, tpr, _ = roc_curve(y, model_scores)
                                precision, recall, _ = precision_recall_curve(y, model_scores)
                                curves[(policy, split)].append(
                                    {
                                        "model": model,
                                        "fpr": fpr,
                                        "tpr": tpr,
                                        "recall": recall,
                                        "precision": precision,
                                        "roc_auc": roc_auc,
                                        "pr_auc": pr_auc,
                                    }
                                )

                        role_arrays = arrays_from_query(
                            connection,
                            f"""
                            SELECT
                                r.event_key::VARCHAR AS event_key,
                                r.adjudicated_type::VARCHAR AS attack_type,
                                r.role::VARCHAR AS role,
                                s.{score_column}::DOUBLE AS score
                            FROM role_base r
                            INNER JOIN {score_relation} s USING (row_id)
                            WHERE r.{label_column} IS NOT NULL
                            ORDER BY r.event_key, r.role, r.tx_hash
                            """,
                        )
                        for attack_scope in ATTACK_SCOPES:
                            paired_rows.append(
                                {
                                    "split": split,
                                    "policy": policy,
                                    "attack_scope": attack_scope,
                                    "configuration": configuration,
                                    "method": method,
                                    "model": model,
                                    **paired_metrics(
                                        np.asarray(role_arrays["event_key"], dtype=str),
                                        np.asarray(role_arrays["attack_type"], dtype=str),
                                        np.asarray(role_arrays["role"], dtype=str),
                                        np.asarray(role_arrays["score"], dtype=np.float64),
                                        attack_scope,
                                    ),
                                }
                            )
    finally:
        connection.close()

    selected_rows = select_by_validation(
        discrimination_rows, threshold_rows, paired_rows
    )
    write_csv(output / "01_cobertura_rotulos.csv", coverage_rows)
    write_csv(output / "02_metricas_discriminacao.csv", discrimination_rows)
    write_csv(output / "03_metricas_limiares.csv", threshold_rows)
    write_csv(output / "04_metricas_top_k.csv", top_rows)
    write_csv(output / "05_metricas_pareadas_eventos.csv", paired_rows)
    write_csv(output / "06_selecao_validacao_teste.csv", selected_rows)

    plot_curves(curves, output)
    overall_metrics = [
        row for row in discrimination_rows if row["attack_scope"] == "all"
    ]
    overall_paired = [row for row in paired_rows if row["attack_scope"] == "all"]
    for policy in POLICIES:
        policy_metrics = [row for row in overall_metrics if row["policy"] == policy]
        heatmap(
            policy_metrics,
            split_keys,
            models,
            "roc_auc",
            f"ROC-AUC por janela — política {policy}",
            output / f"08_heatmap_roc_auc_{policy}",
        )
        heatmap(
            policy_metrics,
            split_keys,
            models,
            "pr_auc_average_precision",
            f"PR-AUC por janela — política {policy}",
            output / f"09_heatmap_pr_auc_{policy}",
        )
        plot_confusions(threshold_rows, policy, models, output)
        heatmap(
            [row for row in overall_paired if row["policy"] == policy],
            split_keys,
            models,
            "pairwise_superiority_rate",
            f"Atacante com escore maior que a vítima — política {policy}",
            output / f"11_heatmap_superioridade_pareada_{policy}",
        )
    plot_selected_models(selected_rows, output)
    plot_selected_confusions(selected_rows, output)

    strict_test = next(
        row
        for row in selected_rows
        if row["policy"] == "strict" and row["method"] == "AE"
    )
    strict_if_test = next(
        row
        for row in selected_rows
        if row["policy"] == "strict" and row["method"] == "IF"
    )
    sensitivity_ae = next(
        row
        for row in selected_rows
        if row["policy"] == "sensitivity" and row["method"] == "AE"
    )
    sensitivity_if = next(
        row
        for row in selected_rows
        if row["policy"] == "sensitivity" and row["method"] == "IF"
    )
    selected_models = {
        "AE": str(strict_test["selected_configuration"]),
        "IF": str(strict_if_test["selected_configuration"]),
    }
    strict_type_metrics = {
        (method, attack_scope): next(
            row
            for row in discrimination_rows
            if row["policy"] == "strict"
            and row["split"] == PRIMARY_TEST_SPLIT
            and row["method"] == method
            and row["configuration"] == configuration
            and row["attack_scope"] == attack_scope
        )
        for method, configuration in selected_models.items()
        for attack_scope in ("insertion", "displacement")
    }
    strict_type_paired = {
        (method, attack_scope): next(
            row
            for row in paired_rows
            if row["policy"] == "strict"
            and row["split"] == PRIMARY_TEST_SPLIT
            and row["method"] == method
            and row["configuration"] == configuration
            and row["attack_scope"] == attack_scope
        )
        for method, configuration in selected_models.items()
        for attack_scope in ("insertion", "displacement")
    }
    report = f"""# Síntese da avaliação com rótulos semânticos

## Desenho

A avaliação usa somente transações com papel conhecido em eventos adjudicados.
Atacantes são positivos e vítimas do mesmo conjunto de eventos são negativos
contextuais. Transações fora desses papéis não são consideradas negativas.

A política estrita inclui somente eventos confirmados. A política de
sensibilidade inclui eventos confirmados ou prováveis. Transações conflitantes
são excluídas da avaliação correspondente.

As configurações abaixo foram escolhidas pela maior PR-AUC em
`{PRIMARY_VALIDATION_SPLIT}` e aplicadas sem nova seleção a
`{PRIMARY_TEST_SPLIT}`.

## Resultado estrito no teste final

- Autoencoder `{strict_test['selected_configuration']}`: ROC-AUC
  {float(strict_test['test_roc_auc']):.4f} (IC95%
  {float(strict_test['test_roc_auc_ci_low']):.4f}–{float(strict_test['test_roc_auc_ci_high']):.4f}),
  PR-AUC {float(strict_test['test_pr_auc']):.4f}, F1 em
  `{strict_test['selected_threshold']}` {float(strict_test['test_selected_threshold_f1']):.4f}
  e superioridade pareada
  {float(strict_test['test_pairwise_superiority']):.4f}.
- Isolation Forest `{strict_if_test['selected_configuration']}`: ROC-AUC
  {float(strict_if_test['test_roc_auc']):.4f} (IC95%
  {float(strict_if_test['test_roc_auc_ci_low']):.4f}–{float(strict_if_test['test_roc_auc_ci_high']):.4f}),
  PR-AUC {float(strict_if_test['test_pr_auc']):.4f}, F1 em
  `{strict_if_test['selected_threshold']}` {float(strict_if_test['test_selected_threshold_f1']):.4f}
  e superioridade pareada
  {float(strict_if_test['test_pairwise_superiority']):.4f}.

## Análise de sensibilidade no teste final

- Autoencoder `{sensitivity_ae['selected_configuration']}`: ROC-AUC
  {float(sensitivity_ae['test_roc_auc']):.4f}, PR-AUC
  {float(sensitivity_ae['test_pr_auc']):.4f}, F1 em
  `{sensitivity_ae['selected_threshold']}` {float(sensitivity_ae['test_selected_threshold_f1']):.4f}
  e superioridade pareada
  {float(sensitivity_ae['test_pairwise_superiority']):.4f}.
- Isolation Forest `{sensitivity_if['selected_configuration']}`: ROC-AUC
  {float(sensitivity_if['test_roc_auc']):.4f}, PR-AUC
  {float(sensitivity_if['test_pr_auc']):.4f}, F1 em
  `{sensitivity_if['selected_threshold']}` {float(sensitivity_if['test_selected_threshold_f1']):.4f}
  e superioridade pareada
  {float(sensitivity_if['test_pairwise_superiority']):.4f}.

## Resultado estrito por tipo de ataque

- Em `insertion`, o AE obteve ROC-AUC
  {float(strict_type_metrics[('AE', 'insertion')]['roc_auc']):.4f} e
  superioridade pareada
  {float(strict_type_paired[('AE', 'insertion')]['pairwise_superiority_rate']):.4f};
  o IF obteve ROC-AUC
  {float(strict_type_metrics[('IF', 'insertion')]['roc_auc']):.4f} e
  superioridade pareada
  {float(strict_type_paired[('IF', 'insertion')]['pairwise_superiority_rate']):.4f}.
- Em `displacement`, com
  {int(strict_type_paired[('AE', 'displacement')]['attacker_victim_pairs'])}
  pares no teste, o AE obteve ROC-AUC
  {float(strict_type_metrics[('AE', 'displacement')]['roc_auc']):.4f} e o IF
  {float(strict_type_metrics[('IF', 'displacement')]['roc_auc']):.4f}. A
  superioridade pareada foi
  {float(strict_type_paired[('AE', 'displacement')]['pairwise_superiority_rate']):.4f}
  nos dois modelos, indicando ordenação inversa nessa subpopulação pequena.

## Interpretação

ROC-AUC e PR-AUC medem a ordenação entre atacantes e vítimas dentro da
população condicionada aos candidatos estruturais. As métricas em q990, q995 e
q999 avaliam os limiares calibrados anteriormente, sem reajuste pelos rótulos.
A superioridade pareada mede a proporção de pares do mesmo evento nos quais o
atacante recebeu escore maior que a vítima.

Esses resultados não estimam prevalência, precisão ou recall em toda a rede.
Os rótulos cobrem apenas eventos encontrados pelas regras C# e adjudicados pelo
pipeline semântico. Essa seleção condicionada deve acompanhar qualquer uso das
métricas na dissertação.

Nos dois cenários do Autoencoder, nenhum dos limiares q990, q995 ou q999 gerou
predições positivas na validação. O q990 é exibido apenas como o menos
conservador entre os três, sem caracterizar um limiar operacional válido. A
discriminação contínua e a superioridade pareada permanecem informativas.

O resultado agregado não deve ocultar a heterogeneidade entre regras. A grande
maioria dos papéis estritos do teste pertence a `insertion`; a evidência para
`displacement` é pequena e apresenta direção oposta à esperada.
"""
    (output / "13_sintese_resultados.md").write_text(report, encoding="utf-8")

    methodology = f"""# Metodologia da avaliação semântica

## População e classes

- Fonte: `03_rotulos_transacao.parquet` e `02_papeis_evento.parquet`.
- Positivo: `attacker_front` ou `attacker_back`.
- Negativo contextual: `victim` em evento positivo da mesma política.
- Não rotulado: excluído; não é presumido negativo.
- Conflito: excluído da política em que ocorre.

## Políticas

- Estrita: {POLICIES['strict']['description']}
- Sensibilidade: {POLICIES['sensitivity']['description']}

## Modelos e separação temporal

São avaliadas cinco configurações de features para Autoencoder e Isolation
Forest em sete janelas. A escolha apresentada como resultado final usa a maior
PR-AUC na validação pré-Pectra, separadamente por método e política. O teste
final não participa dessa seleção. Após escolher a configuração, o limiar entre
q990, q995 e q999 é escolhido pelo maior F1 na mesma validação e aplicado sem
reajuste ao teste final.

## Métricas

- ROC-AUC e PR-AUC sobre escores contínuos;
- IC95% por bootstrap estratificado com {args.bootstrap_repetitions} repetições;
- precisão, recall, especificidade, F1, acurácia balanceada e MCC nos limiares
  q990, q995 e q999 previamente calibrados;
- precisão, recall e lift nos top 1%, 5% e 10% dos escores;
- superioridade pareada atacante–vítima dentro do mesmo evento.

## Limitação central

A referência semântica é condicionada aos candidatos produzidos pelas regras
estruturais C#. Portanto, as métricas caracterizam a capacidade de ordenar e
separar papéis dentro dessa população e não o desempenho supervisionado sobre
todas as transações Ethereum.
"""
    (output / "14_metodologia_avaliacao.md").write_text(
        methodology, encoding="utf-8"
    )

    manifest_path = output / "15_manifest_avaliacao.json"
    artifacts = sorted(
        path.name
        for path in output.iterdir()
        if path.is_file() and path.name != manifest_path.name
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "script": Path(__file__).name,
        "evaluation_population": "semantic_event_roles_only",
        "unlabeled_policy": "excluded_not_assumed_negative",
        "conflict_policy": "excluded_per_policy",
        "validation_selection_metric": "pr_auc_average_precision",
        "threshold_selection_metric": "f1_among_q990_q995_q999_on_validation",
        "validation_split": PRIMARY_VALIDATION_SPLIT,
        "primary_test_split": PRIMARY_TEST_SPLIT,
        "policies": POLICIES,
        "thresholds": list(THRESHOLDS),
        "top_fractions": list(TOP_FRACTIONS),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "seed": args.seed,
        "splits": split_keys,
        "configurations": configurations,
        "methods": ["AE", "IF"],
        "counts": {
            "coverage_rows": len(coverage_rows),
            "discrimination_rows": len(discrimination_rows),
            "threshold_rows": len(threshold_rows),
            "top_k_rows": len(top_rows),
            "paired_rows": len(paired_rows),
            "selected_rows": len(selected_rows),
        },
        "sources": [
            {
                "path": str(path.resolve()),
                "bytes": path.resolve().stat().st_size,
                "sha256": sha256(path.resolve()),
            }
            for path in (args.labels, args.event_roles, args.split_manifest)
        ],
        "artifacts": artifacts + [manifest_path.name],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output),
                "discrimination_rows": len(discrimination_rows),
                "selected_models": selected_rows,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
