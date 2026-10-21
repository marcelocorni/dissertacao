"""Populações explícitas, baselines, reamostragem e contagens da revisão."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa

from .core import CONFIGURATIONS, PROTOCOL, atomic_json
from .dados import (RAW, SEMANTIC, connect, ident, load_source, metadata, parquet, q,
                    semantic_cohort)
from .fluxo import read_json, require, write_csv
from .metricas import (HEURISTICS, classification, clustered_contrast, discrimination,
                       orient_baseline, paired_values, select_threshold, top_metrics,
                       transaction_intervals)


def get_scores(con, source, store, split, config, method, policy):
    cohort = semantic_cohort(con, source, split, policy)
    con.register("eligible", cohort)
    name = f"scores/{split['key']}/{method}/{config}"
    require(store, name)
    path = store.destination(name) / "scores.parquet"
    column = "reconstruction_error" if method == "AE" else "anomaly_score"
    relation = parquet(path)
    total, distinct = con.execute(f"SELECT count(*), count(DISTINCT row_id) FROM {relation}").fetchone()
    if total != distinct:
        raise ValueError("row_id duplicado nos escores")
    table = con.execute(f"SELECT e.*, s.{column} AS score FROM eligible e JOIN {relation} s USING(row_id) ORDER BY e.tx_hash").to_arrow_table()
    if len(table) != len(cohort):
        raise ValueError("Cobertura incompleta da população semântica")
    values = table.to_pydict()
    values = {key: np.asarray(value) for key, value in values.items()}
    discrimination(values["label"], values["score"])
    return values


def role_population(con, source, split, policy, population, conflict_policy="exclude_event"):
    con.register("scored", pa.table({k: v for k, v in population.items()}))
    roles = parquet(Path(source) / SEMANTIC / "02_papeis_evento.parquet")
    labels = parquet(Path(source) / SEMANTIC / "03_rotulos_transacao.parquet")
    label = f"{policy}_label"
    # Mark complete events FIRST, then join scores; never evaluate partial sandwiches.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE role_audit AS
        SELECT r.*, s.block_number, s.score, s.label AS global_label,
               l.{policy}_conflict AS participant_conflict,
               s.tx_hash IS NOT NULL AND r.{label}=s.label AS eligible
        FROM {roles} r LEFT JOIN scored s ON lower(r.tx_hash)=s.tx_hash
        LEFT JOIN {labels} l ON l.split=r.split AND lower(l.tx_hash)=lower(r.tx_hash)
        WHERE r.split={q(split['key'])} AND r.event_key IN
              (SELECT event_key FROM {roles} WHERE split={q(split['key'])} AND {label} IS NOT NULL)""")
    unexplained = con.execute("SELECT count(*) FROM role_audit WHERE NOT coalesce(eligible,false) AND NOT coalesce(participant_conflict,false)").fetchone()[0]
    if unexplained:
        raise ValueError("Papéis sem cobertura ou rótulo coerente, sem conflito declarado; investigar dados")
    total, excluded = con.execute("SELECT count(*), count(*) FILTER(WHERE NOT complete) FROM (SELECT event_key, bool_and(coalesce(eligible,false)) AS complete FROM role_audit GROUP BY 1)").fetchone()
    if excluded and conflict_policy != "exclude_event":
        raise ValueError(f"{split['key']}/{policy}: {excluded} eventos com conflito/perda de cobertura. Regra pendente de aprovação; não gerar pares silenciosamente.")
    result = con.execute("""SELECT r.event_key, r.role, r.score, r.block_number, r.adjudicated_type
        FROM role_audit r JOIN (SELECT event_key FROM role_audit GROUP BY 1
        HAVING bool_and(coalesce(eligible,false))) e USING(event_key)
        ORDER BY r.event_key,r.role,lower(r.tx_hash)""").fetchnumpy()
    return result, {"events_before": total, "events_excluded_incomplete_or_conflicting": excluded,
                    "conflict_policy": conflict_policy}


def evaluate(source, store, conflict_policy="exclude_event"):
    if store.complete("complements"):
        return
    _, splits = load_source(source)
    validation = next(s for s in splits if s["role"] == "validation")
    configs = read_json(store.destination("prepare") / "preprocessing.json")["configurations"]
    rows, binary, tops, paired, choices, intervals, baselines, audit = [], [], [], [], [], [], [], []
    with store.stage("complements") as out, connect(out / ".tmp") as con:
        for policy in ("strict", "sensitivity"):
            selected = {}
            for method in ("AE", "IF"):
                candidates = []
                for order, config in enumerate(configs):
                    values = get_scores(con, source, store, validation, config["key"], method, policy)
                    candidates.append((discrimination(values["label"], values["score"])["average_precision"], -order, config))
                selected[method] = max(candidates, key=lambda x: x[:2])[2]["key"]
                # Threshold candidates use the ENTIRE validation distribution, not semantic rows.
                name = f"scores/{validation['key']}/{method}/{selected[method]}"
                score_column = "reconstruction_error" if method == "AE" else "anomaly_score"
                full = con.execute(f"SELECT {score_column} FROM {parquet(store.destination(name) / 'scores.parquet')}").fetchnumpy()[score_column]
                values = get_scores(con, source, store, validation, selected[method], method, policy)
                chosen, _ = select_threshold(values["label"], values["score"], full)
                choices.append({"policy": policy, "method": method, "configuration": selected[method],
                                "threshold": chosen["threshold"], "cut": chosen["value"]})
            baseline_specs = {}
            cohort = semantic_cohort(con, source, validation, policy)
            con.register("baseline_cohort", cohort)
            for feature in HEURISTICS:
                raw = parquet(Path(validation["output_directory"]) / RAW["B"])
                full = con.execute(f"SELECT {ident(feature)} FROM {raw} ORDER BY row_id").fetchnumpy()[feature]
                vals = con.execute(f"SELECT e.label, x.{ident(feature)} AS value FROM baseline_cohort e JOIN {raw} x USING(row_id) ORDER BY e.tx_hash").fetchnumpy()
                if len(vals["label"]) != len(cohort):
                    raise ValueError("Cobertura incompleta do baseline")
                direction, orientation = orient_baseline(vals["label"], vals["value"])
                chosen, cuts = select_threshold(vals["label"], direction * vals["value"], direction * full)
                baseline_specs[feature] = (direction, chosen)
                baselines.append({"policy": policy, "feature": feature, "direction": direction,
                                  "threshold": chosen["threshold"], "cut": chosen["value"], **orientation})
            for split in splits:
                populations = {}
                for method in ("AE", "IF"):
                    for config in configs:
                        values = get_scores(con, source, store, split, config["key"], method, policy)
                        prefix = {"split": split["key"], "policy": policy, "method": method,
                                  "configuration": config["key"], "rows": len(values["label"]),
                                  "prevalence": float(values["label"].mean())}
                        for scope in ("all", "insertion", "displacement"):
                            mask = np.ones(len(values["label"]), dtype=bool) if scope == "all" else values["attack_types"] == scope
                            if set(values["label"][mask]) == {0, 1}:
                                rows.append({**prefix, "rows": int(mask.sum()), "prevalence": float(values["label"][mask].mean()),
                                             "scope": scope, **discrimination(values["label"][mask], values["score"][mask])})
                        if config["key"] == selected[method]:
                            populations[method] = values
                            choice = next(c for c in choices if c["policy"] == policy and c["method"] == method)
                            binary.append({**prefix, "scope": "all", "threshold": choice["threshold"], "cut": choice["cut"],
                                           **classification(values["label"], values["score"] >= choice["cut"])})
                            for fraction in (.01, .05, .1):
                                tops.append({**prefix, "scope": "all", **top_metrics(values["label"], values["score"], fraction)})
                            roles, role_audit = role_population(con, source, split, policy, values, conflict_policy)
                            audit.append({**prefix, **role_audit})
                            pairs = paired_values(roles["event_key"], roles["role"], roles["score"], roles["block_number"])
                            for scope in ("all", "insertion", "displacement"):
                                mask = np.ones(len(roles["event_key"]), dtype=bool) if scope == "all" else roles["adjudicated_type"] == scope
                                scoped = paired_values(roles["event_key"][mask], roles["role"][mask], roles["score"][mask], roles["block_number"][mask])
                                paired.append({**prefix, "scope": scope, "pairs": len(scoped["margins"]), "events": len(scoped["event_success"]),
                                               "attacker_score_higher": int(np.sum(scoped["margins"] > 0)),
                                               "equal_scores": int(np.sum(scoped["margins"] == 0)),
                                               "events_all_attackers_above_victim": int(np.sum(scoped["event_success"])),
                                               "median_score_margin": float(np.median(scoped["margins"])) if len(scoped["margins"]) else None,
                                               "mean_score_margin": float(np.mean(scoped["margins"])) if len(scoped["margins"]) else None,
                                               "superiority": float((scoped["margins"] > 0).mean()) if len(scoped["margins"]) else None,
                                               "event_superiority": float(scoped["event_success"].mean()) if len(scoped["event_success"]) else None})
                                if scope != "all":
                                    tx_mask = values["attack_types"] == scope
                                    if set(values["label"][tx_mask]) == {0, 1}:
                                        subprefix = {**prefix, "rows": int(tx_mask.sum()), "prevalence": float(values["label"][tx_mask].mean())}
                                        binary.append({**subprefix, "scope": scope, "threshold": choice["threshold"], "cut": choice["cut"],
                                                       **classification(values["label"][tx_mask], values["score"][tx_mask] >= choice["cut"])})
                                        for fraction in (.01, .05, .1):
                                            tops.append({**subprefix, "scope": scope, **top_metrics(values["label"][tx_mask], values["score"][tx_mask], fraction)})
                            populations[method + "_pairs"] = pairs
                cohort = semantic_cohort(con, source, split, policy)
                con.register("baseline_cohort", cohort)
                for feature, (direction, cut) in baseline_specs.items():
                    raw = parquet(Path(split["output_directory"]) / RAW["B"])
                    vals = con.execute(f"SELECT e.label, x.{ident(feature)} AS value FROM baseline_cohort e JOIN {raw} x USING(row_id) ORDER BY e.tx_hash").fetchnumpy()
                    score = direction * vals["value"]
                    prefix = {"split": split["key"], "policy": policy, "method": "heuristic", "configuration": feature,
                              "rows": len(score), "prevalence": float(np.mean(vals["label"]))}
                    rows.append({**prefix, "scope": "all", **discrimination(vals["label"], score)})
                    binary.append({**prefix, "scope": "all", "threshold": cut["threshold"], "cut": cut["value"], **classification(vals["label"], score >= cut["value"])})
                    for fraction in (.01, .05, .1):
                        tops.append({**prefix, "scope": "all", **top_metrics(vals["label"], score, fraction)})
                if split["role"] in ("validation", "test"):
                    ae, forest = populations["AE"], populations["IF"]
                    if not np.array_equal(ae["tx_hash"], forest["tx_hash"]):
                        raise ValueError("Populações AE e IF divergem")
                    intervals.append({"split": split["key"], "policy": policy, "unit": "unique_transaction",
                                      **transaction_intervals(ae["label"], ae["score"], forest["score"], ae["block_number"])})
                    a, f = populations["AE_pairs"], populations["IF_pairs"]
                    for unit, block_key, value_key in (("pair", "pair_blocks", "margins"), ("event", "event_blocks", "event_success")):
                        if not np.array_equal(a[block_key], f[block_key]):
                            raise ValueError("Pares/eventos dos métodos divergem")
                        av = a[value_key] > 0 if unit == "pair" else a[value_key]
                        fv = f[value_key] > 0 if unit == "pair" else f[value_key]
                        def statistic(indices):
                            aa, ff = float(av[indices].mean()), float(fv[indices].mean())
                            return {"ae": aa, "if": ff, "if_minus_ae": ff - aa}
                        intervals.append({"split": split["key"], "policy": policy, "unit": unit,
                                          **clustered_contrast(a[block_key], statistic)})
        write_csv(out / "discrimination.csv", rows)
        write_csv(out / "thresholds.csv", binary)
        write_csv(out / "top.csv", tops)
        write_csv(out / "paired.csv", paired)
        write_csv(out / "selection.csv", choices)
        write_csv(out / "baselines.csv", baselines)
        write_csv(out / "role_population_audit.csv", audit)
        atomic_json(out / "clustered_intervals.json", intervals)


def count_reuse_and_types(source, store):
    if store.complete("descriptive"):
        return
    _, splits = load_source(source)
    reuse, types, funnels = [], [], []
    with store.stage("descriptive") as out, connect(out / ".tmp") as con:
        roles = parquet(Path(source) / SEMANTIC / "02_papeis_evento.parquet")
        labels = parquet(Path(source) / SEMANTIC / "03_rotulos_transacao.parquet")
        for split in splits:
            meta = parquet(metadata(split))
            manifest = Path(source) / "04-pos-processamento/06-validacao-semantica-front-running/resultados" / split["key"] / "07_manifest_validacao.json"
            semantic = read_json(manifest)
            if sum(semantic["status_counts"].values()) != semantic["events"]:
                raise ValueError("Funil semântico inconsistente")
            for status, n in semantic["status_counts"].items():
                pattern, state = status.split(":")
                funnels.append({"split": split["key"], "pattern": pattern, "status": state, "events": n})
            for policy in ("strict", "sensitivity"):
                query = f"""SELECT r.event_key, lower(r.tx_hash) AS tx_hash, lower(m.from_address) AS sender
                    FROM {roles} r JOIN {labels} l ON l.split=r.split AND lower(l.tx_hash)=lower(r.tx_hash)
                    JOIN {meta} m ON lower(m.hash)=lower(r.tx_hash)
                    WHERE r.split={q(split['key'])} AND r.{policy}_label=1
                    AND l.{policy}_label=1 AND NOT l.{policy}_conflict"""
                records = con.execute(query).fetchall()
                for unit, index in (("transaction_hash", 1), ("sender_address", 2)):
                    if any(not r[index] for r in records):
                        raise ValueError("Metadados ausentes na análise de reutilização")
                    counts = Counter(r[index] for r in records)
                    event_counts = Counter()
                    for event, value in {(r[0], r[index]) for r in records}:
                        event_counts[value] += 1
                    reuse.append({"split": split["key"], "policy": policy, "unit": unit,
                                  "population_scope": "eligible_positive_roles_before_paired_event_exclusion",
                                  "occurrences": len(records), "unique": len(counts),
                                  "repeated_in_multiple_events": sum(n > 1 for n in event_counts.values()),
                                  "max_event_count": max(event_counts.values(), default=0)})
            for method in ("AE", "IF"):
                for config in CONFIGURATIONS:
                    name = f"scores/{split['key']}/{method}/{config}"
                    require(store, name)
                    scores = parquet(store.destination(name) / "scores.parquet")
                    for grouping, expression in (("type4_vs_other", "CASE WHEN m.transaction_type=4 THEN 'type_4' ELSE 'not_type_4' END"),
                                                  ("type2_vs_type4", "CAST(m.transaction_type AS VARCHAR)")):
                        where = "WHERE m.transaction_type IN (2,4)" if grouping == "type2_vs_type4" else ""
                        for group, n, anomalies in con.execute(f"SELECT {expression}, count(*), sum(CAST(s.is_anomaly_q995 AS INTEGER)) FROM {scores} s JOIN {meta} m USING(row_id) {where} GROUP BY 1").fetchall():
                            types.append({"split": split["key"], "method": method, "configuration": config,
                                          "grouping": grouping, "group": group, "rows": n,
                                          "anomalies_q995": anomalies, "rate": anomalies / n})
        write_csv(out / "funnel.csv", funnels)
        write_csv(out / "reuse.csv", reuse)
        write_csv(out / "transaction_types.csv", types)
