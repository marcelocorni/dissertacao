"""Métricas, baselines e bootstrap por blocos, sem acesso à rede."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

QUANTILES = (("q990", .99), ("q995", .995), ("q999", .999))
HEURISTICS = ("gas_price_rank_block", "transaction_index_pct", "gas_used_ratio")


def discrimination(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if y.shape != scores.shape or set(y) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Métrica exige duas classes, cobertura completa e escores finitos")
    return {"roc_auc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores))}


def classification(y, flags):
    y, flags = np.asarray(y), np.asarray(flags, dtype=bool)
    if y.shape != flags.shape:
        raise ValueError("Cobertura divergente")
    tp = int(np.sum((y == 1) & flags))
    fp = int(np.sum((y == 0) & flags))
    fn = int(np.sum((y == 1) & ~flags))
    tn = int(np.sum((y == 0) & ~flags))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "precision": precision,
            "recall": recall, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "flagged_share": float(flags.mean())}


def select_threshold(y, scores, calibration):
    if not np.isfinite(calibration).all() or not len(calibration):
        raise ValueError("Calibração vazia ou não finita")
    candidates = [{"threshold": name, "value": float(np.quantile(calibration, quantile)),
                   **classification(y, np.asarray(scores) >= np.quantile(calibration, quantile))}
                  for name, quantile in QUANTILES]
    # max preserves Q99 priority in exact F1 ties; tied scores are all flagged.
    return max(candidates, key=lambda r: r["f1"]), candidates


def orient_baseline(y, values):
    plus = discrimination(y, values)["average_precision"]
    minus = discrimination(y, -np.asarray(values))["average_precision"]
    return (1 if plus >= minus else -1), {"positive_ap": plus, "negative_ap": minus}


def top_metrics(y, scores, fraction):
    # Inputs must already be ordered by lower(tx_hash), matching the historical tie rule.
    n = max(1, math.ceil(len(y) * fraction))
    indices = np.argsort(scores, kind="stable")[-n:]
    hits = int(np.asarray(y)[indices].sum())
    precision = hits / n
    return {"top_fraction": fraction, "top_n": n, "positives": hits,
            "precision": precision, "recall": hits / int(np.sum(y)),
            "lift": precision / float(np.mean(y))}


def paired_values(events, roles, scores, blocks):
    margins, pair_blocks, event_success, event_blocks = [], [], [], []
    events, roles, scores, blocks = map(np.asarray, (events, roles, scores, blocks))
    if set(roles) - {"victim", "attacker", "attacker_front", "attacker_back"}:
        raise ValueError("Papel desconhecido na comparação pareada")
    for key in sorted(set(events)):
        indices = np.flatnonzero(events == key)
        if len(set(blocks[indices])) != 1:
            raise ValueError("Evento abrange múltiplos blocos: revisar agrupamento")
        attackers = scores[indices][roles[indices] != "victim"]
        victims = scores[indices][roles[indices] == "victim"]
        if not len(attackers) or not len(victims):
            continue
        if len(victims) != 1:
            raise ValueError("Evento com múltiplas vítimas: revisar definição dos pares")
        current = (attackers[:, None] - victims[None, :]).ravel()
        margins.extend(current)
        pair_blocks.extend([blocks[indices[0]]] * len(current))
        event_success.append(bool(np.all(current > 0)))
        event_blocks.append(blocks[indices[0]])
    return {"margins": np.asarray(margins), "pair_blocks": np.asarray(pair_blocks),
            "event_success": np.asarray(event_success), "event_blocks": np.asarray(event_blocks)}


def clustered_contrast(blocks, statistic, repetitions=2000, seed=20261018):
    """statistic(indices) returns all methods on the SAME replicated rows."""
    blocks = np.asarray(blocks)
    unique = np.unique(blocks)
    if not len(unique):
        raise ValueError("Nenhum bloco avaliável")
    positions = {b: np.flatnonzero(blocks == b) for b in unique}
    rng = np.random.default_rng(seed)
    values, invalid = [], 0
    for _ in range(repetitions):
        draws = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[b] for b in draws])
        result = statistic(indices)
        if result is None or not all(np.isfinite(x) for x in result.values()):
            invalid += 1
        else:
            values.append(result)
    if not repetitions:
        raise ValueError("Bootstrap sem repetições")
    review = invalid / repetitions > .05 or not values
    intervals = {} if review else {key: [float(x) for x in np.quantile([v[key] for v in values], [.025, .975])]
                                  for key in values[0]}
    return {"status": "requires_review" if review else "completed", "replications": repetitions,
            "invalid": invalid, "seed": seed, "blocks": len(unique), "intervals": intervals,
            "estimand": "conditional_on_fitted_models_not_training_variability"}


def transaction_intervals(y, ae, forest, blocks, repetitions=2000):
    y, ae, forest = map(np.asarray, (y, ae, forest))
    def statistic(indices):
        if len(set(y[indices])) != 2:
            return None
        a, f = discrimination(y[indices], ae[indices]), discrimination(y[indices], forest[indices])
        return {"ae_roc_auc": a["roc_auc"], "if_roc_auc": f["roc_auc"],
                "if_minus_ae_roc_auc": f["roc_auc"] - a["roc_auc"],
                "ae_ap": a["average_precision"], "if_ap": f["average_precision"],
                "if_minus_ae_ap": f["average_precision"] - a["average_precision"]}
    return clustered_contrast(blocks, statistic, repetitions)
