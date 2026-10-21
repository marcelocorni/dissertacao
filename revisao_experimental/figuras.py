"""Figuras dos modelos finais e avaliações complementares efetivamente calculadas."""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .complementos import get_scores
from .core import atomic_json
from .dados import connect, legacy, load_source
from .fluxo import read_json, require


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def generate(source, store):
    require(store, "complements")
    require(store, "exports")
    if store.complete("figures-semantic"):
        return
    _, splits = load_source(source)
    test = next(s for s in splits if s["role"] == "test")
    comp = store.destination("complements")
    choices = csv_rows(comp / "selection.csv")
    configs = read_json(store.destination("prepare") / "preprocessing.json")["configurations"]
    with store.stage("figures-semantic") as out, connect(out / ".tmp") as con:
        # Reuse only plot functions, never the historical fitting entry point.
        ae = legacy("03-machine-learning/02-ae/01_treinar_autoencoder.py")
        history = csv_rows(store.destination("exports") / "AE/03_historico_treinamento.csv")
        numeric_history = [{**r, "epoch": int(r["epoch"]), "train_loss": float(r["train_loss"]),
                            "internal_validation_loss": float(r["internal_validation_loss"])} for r in history]
        ae.plot_history(out / "ae_training.pdf", out / "ae_training.png", numeric_history, configs)
        for policy in ("strict", "sensitivity"):
            curves, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            confusion, matrix_axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
            for method, axis in zip(("AE", "IF"), matrix_axes):
                choice = next(c for c in choices if c["policy"] == policy and c["method"] == method)
                values = get_scores(con, source, store, test, choice["configuration"], method, policy)
                y, score = values["label"], values["score"]
                fpr, tpr, _ = roc_curve(y, score)
                precision, recall, _ = precision_recall_curve(y, score)
                label = f"{method}: {choice['configuration']}"
                axes[0].plot(fpr, tpr, label=label)
                axes[1].plot(recall, precision, label=label)
                matrix = confusion_matrix(y, score >= float(choice["cut"]), labels=[0, 1])
                axis.imshow(matrix, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        axis.text(j, i, f"{matrix[i,j]:,}", ha="center", va="center")
                axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Pred. vítima", "Pred. atacante"],
                         yticklabels=["Vítima", "Atacante"], title=f"{method} — {choice['threshold']}")
            axes[0].plot([0, 1], [0, 1], "--", color="gray")
            axes[0].set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos")
            axes[1].axhline(float(y.mean()), linestyle="--", color="gray", label="Prevalência contextual")
            axes[1].set(xlabel="Recall", ylabel="Precision")
            for axis in axes:
                axis.legend(fontsize=8)
            curves.suptitle(f"Teste final — modelos selecionados na validação — {policy}")
            confusion.suptitle(f"Teste final — população semântica — {policy}")
            save(curves, out / f"curves_{policy}")
            save(confusion, out / f"confusion_{policy}")
        metrics = csv_rows(comp / "discrimination.csv")
        paired = csv_rows(comp / "paired.csv")
        fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        strict = [r for r in paired if r["policy"] == "strict" and r["scope"] == "all"]
        split_keys = [s["key"] for s in splits]
        lookup = {(r["split"], r["method"]): float(r["superiority"]) for r in strict}
        values = np.array([[lookup[(s, method)] for s in split_keys] for method in ("AE", "IF")])
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(2):
            for j in range(len(split_keys)):
                axis.text(j, i, f"{values[i,j]:.1%}", ha="center", va="center", color="white")
        axis.set(yticks=[0, 1], yticklabels=["AE selecionado", "IF selecionado"],
                 xticks=range(len(split_keys)), xticklabels=[s["order"] for s in splits],
                 xlabel="Janela (ordem cronológica do manifesto)", title="Superioridade por par — política estrita")
        fig.colorbar(image, ax=axis, label="Fração de pares com atacante acima da vítima")
        save(fig, out / "paired_strict")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for method, axis in zip(("AE", "IF"), axes):
            subset = [r for r in metrics if r["method"] == method and r["policy"] == "strict" and r["scope"] == "all"]
            keys = [c["key"] for c in configs]
            for offset, role, label in ((-.18, "validation", "Validação"), (.18, "test", "Teste")):
                split_key = next(s["key"] for s in splits if s["role"] == role)
                lookup = {r["configuration"]: float(r["average_precision"]) for r in subset if r["split"] == split_key}
                axis.bar(np.arange(len(keys)) + offset, [lookup[k] for k in keys], width=.36, label=label)
            axis.set(xticks=range(len(keys)), xticklabels=["A", "B", "C", "B-ST", "C-ST"], ylabel="AP", title=f"{method} — política estrita")
            axis.legend()
        save(fig, out / "ap_positions")
        atomic_json(out / "context.json", {"execution": str(store.root), "semantic_scope": "contextual_attacker_victim",
                    "test_previously_inspected": True, "quantiles": "calibrated_validation_only",
                    "paired_conflicts": "whole_event_excluded", "intervals": "separate clustered_intervals.json"})
