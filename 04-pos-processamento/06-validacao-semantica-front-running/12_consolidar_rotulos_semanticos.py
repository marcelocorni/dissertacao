# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Consolida as adjudicações semânticas das sete janelas temporais.

Produz referências estrita (somente confirmado) e de sensibilidade
(confirmado + provável). O conjunto contém somente papéis de eventos
adjudicados; transações ausentes não são convertidas implicitamente em classe
negativa.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb


SPLIT_ORDER = (
    "estresse_pre_dencun_2024",
    "transicao_dencun_2024",
    "treino_2024_pos_dencun",
    "validacao_2025_pre_pectra",
    "transicao_pectra_2025",
    "teste_final_2025",
    "estresse_fusaka_2025",
)
POSITIVE_TYPES = {"insertion", "displacement"}
CONFIRMED = "confirmado"
PROBABLE = "provável"

EVENT_FIELDS = (
    "event_key", "audit_id", "detection_event_id", "split",
    "source_detector", "adjudicated_type", "semantic_status",
    "decision", "evidence_quality", "reason_codes", "notes", "reviewer",
    "reviewed_at_utc", "decision_source", "validation_result",
    "attacker_front_hash", "attacker_back_hash", "victim_hash",
    "protocols", "pool_addresses", "economic_status",
    "strict_positive_event", "sensitivity_positive_event",
)
ROLE_FIELDS = (
    "event_key", "audit_id", "detection_event_id", "split",
    "source_detector", "adjudicated_type", "decision", "decision_source",
    "tx_hash", "role", "strict_label", "sensitivity_label",
)
TRANSACTION_FIELDS = (
    "split", "tx_hash", "roles", "attack_types", "event_keys",
    "event_count", "strict_label", "strict_conflict",
    "sensitivity_label", "sensitivity_conflict", "reference_scope",
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=here / "resultados",
        help="Raiz que contém uma subpasta para cada janela.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=here / "resultados" / "consolidado",
        help="Diretório dos artefatos globais.",
    )
    return parser.parse_args()


def low(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_parquet(connection: duckdb.DuckDBPyConnection, path: Path) -> list[dict[str, Any]]:
    cursor = connection.execute("SELECT * FROM read_parquet(?)", [str(path)])
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sql_literal(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def csv_to_parquet(connection: duckdb.DuckDBPyConnection, csv_path: Path, parquet_path: Path) -> None:
    connection.execute(
        "COPY (SELECT * FROM read_csv_auto("
        f"{sql_literal(csv_path)}, header = true, all_varchar = false, "
        "sample_size = -1, nullstr = '')) TO "
        f"{sql_literal(parquet_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def load_decisions(window: Path) -> dict[str, dict[str, str]]:
    assisted_path = window / "15_decisoes_dossie.csv"
    return {row["audit_id"]: row for row in read_csv(assisted_path)}


def label_for(role: str, eligible: bool) -> int | None:
    if not eligible:
        return None
    return 0 if role == "victim" else 1


def build_roles(event: dict[str, Any]) -> list[tuple[str, str]]:
    roles: list[tuple[str, str]] = []
    front = low(event.get("attacker_front_hash"))
    back = low(event.get("attacker_back_hash"))
    victim = low(event.get("victim_hash"))
    if front:
        roles.append((front, "attacker_front"))
    if back:
        roles.append((back, "attacker_back"))
    if victim:
        roles.append((victim, "victim"))
    return roles


def collapse_label(values: set[int]) -> tuple[int | None, int]:
    if not values:
        return None, 0
    if len(values) > 1:
        return None, 1
    return next(iter(values)), 0


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(":memory:")
    event_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for split in SPLIT_ORDER:
        window = results_root / split
        required = (
            window / "08_dossie_auditoria.parquet",
            window / "15_decisoes_dossie.csv",
            window / "17_validacao_adjudicacoes.csv",
            window / "19_manifest_validacao_adjudicacoes.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: artefatos ausentes: {missing}")

        dossier = read_parquet(connection, required[0])
        decisions = load_decisions(window)
        validations = {
            row["audit_id"]: row for row in read_csv(required[2])
        }

        dossier_ids = [str(row["audit_id"]) for row in dossier]
        if len(dossier_ids) != len(set(dossier_ids)):
            raise RuntimeError(f"{split}: audit_id duplicado no dossiê")
        if set(dossier_ids) != set(decisions):
            raise RuntimeError(
                f"{split}: cobertura de decisões divergente: "
                f"dossiê={len(dossier_ids)}, decisões={len(decisions)}"
            )

        counters: Counter[str] = Counter()
        for event in dossier:
            audit_id = str(event["audit_id"])
            decision = decisions[audit_id]
            validation = validations.get(audit_id, {})
            validation_result = validation.get("validation_result", "missing")
            if validation_result != "valid":
                raise RuntimeError(
                    f"{split}/{audit_id}: adjudicação não aprovada: {validation_result}"
                )

            label = low(decision.get("manual_label"))
            adjudicated_type = low(decision.get("manual_type"))
            strict_positive = label == CONFIRMED and adjudicated_type in POSITIVE_TYPES
            sensitivity_positive = (
                label in {CONFIRMED, PROBABLE} and adjudicated_type in POSITIVE_TYPES
            )
            event_key = f"{split}::{audit_id}"
            decision_source = "assisted_deterministic"
            row = {
                "event_key": event_key,
                "audit_id": audit_id,
                "detection_event_id": int(event["detection_event_id"]),
                "split": split,
                "source_detector": low(event.get("detector")),
                "adjudicated_type": adjudicated_type,
                "semantic_status": low(event.get("validation_status")),
                "decision": label,
                "evidence_quality": decision.get("evidence_quality", ""),
                "reason_codes": decision.get("reason_codes", ""),
                "notes": decision.get("notes", ""),
                "reviewer": decision.get("reviewer", ""),
                "reviewed_at_utc": decision.get("reviewed_at_utc", ""),
                "decision_source": decision_source,
                "validation_result": validation_result,
                "attacker_front_hash": low(event.get("attacker_front_hash")),
                "attacker_back_hash": low(event.get("attacker_back_hash")),
                "victim_hash": low(event.get("victim_hash")),
                "protocols": event.get("protocols", ""),
                "pool_addresses": event.get("pool_addresses", ""),
                "economic_status": event.get("economic_status", ""),
                "strict_positive_event": int(strict_positive),
                "sensitivity_positive_event": int(sensitivity_positive),
            }
            event_rows.append(row)
            counters[f"decision:{label}"] += 1
            counters[f"type:{adjudicated_type}"] += 1
            counters[f"source:{decision_source}"] += 1

            roles = build_roles(event)
            if strict_positive and adjudicated_type == "insertion" and len(roles) != 3:
                raise RuntimeError(f"{event_key}: insertion confirmado sem três papéis")
            if strict_positive and adjudicated_type == "displacement" and len(roles) < 2:
                raise RuntimeError(f"{event_key}: displacement confirmado sem dois papéis")
            for tx_hash, role in roles:
                role_rows.append(
                    {
                        "event_key": event_key,
                        "audit_id": audit_id,
                        "detection_event_id": int(event["detection_event_id"]),
                        "split": split,
                        "source_detector": low(event.get("detector")),
                        "adjudicated_type": adjudicated_type,
                        "decision": label,
                        "decision_source": decision_source,
                        "tx_hash": tx_hash,
                        "role": role,
                        "strict_label": label_for(role, strict_positive),
                        "sensitivity_label": label_for(role, sensitivity_positive),
                    }
                )

        split_counts[split] = dict(sorted(counters.items()))
        for path in required:
            sources.append(
                {
                    "path": relative(path, here),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )

    aggregate: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "roles": set(), "attack_types": set(), "event_keys": set(),
            "strict": set(), "sensitivity": set(),
        }
    )
    for row in role_rows:
        item = aggregate[(row["split"], row["tx_hash"])]
        item["roles"].add(row["role"])
        item["attack_types"].add(row["adjudicated_type"])
        item["event_keys"].add(row["event_key"])
        if row["strict_label"] is not None:
            item["strict"].add(row["strict_label"])
        if row["sensitivity_label"] is not None:
            item["sensitivity"].add(row["sensitivity_label"])

    transaction_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for (split, tx_hash), item in aggregate.items():
        strict_label, strict_conflict = collapse_label(item["strict"])
        sensitivity_label, sensitivity_conflict = collapse_label(item["sensitivity"])
        row = {
            "split": split,
            "tx_hash": tx_hash,
            "roles": "|".join(sorted(item["roles"])),
            "attack_types": "|".join(sorted(item["attack_types"])),
            "event_keys": "|".join(sorted(item["event_keys"])),
            "event_count": len(item["event_keys"]),
            "strict_label": strict_label,
            "strict_conflict": strict_conflict,
            "sensitivity_label": sensitivity_label,
            "sensitivity_conflict": sensitivity_conflict,
            "reference_scope": "semantic_event_roles_only",
        }
        transaction_rows.append(row)
        if strict_conflict or sensitivity_conflict:
            conflict_rows.append(row)

    split_index = {name: index for index, name in enumerate(SPLIT_ORDER)}
    event_rows.sort(key=lambda row: (split_index[row["split"]], row["detection_event_id"], row["audit_id"]))
    role_rows.sort(key=lambda row: (split_index[row["split"]], row["detection_event_id"], row["role"], row["tx_hash"]))
    transaction_rows.sort(key=lambda row: (split_index[row["split"]], row["tx_hash"]))
    conflict_rows.sort(key=lambda row: (split_index[row["split"]], row["tx_hash"]))

    paths = {
        "events_csv": output_dir / "01_eventos_adjudicados.csv",
        "events_parquet": output_dir / "01_eventos_adjudicados.parquet",
        "roles_csv": output_dir / "02_papeis_evento.csv",
        "roles_parquet": output_dir / "02_papeis_evento.parquet",
        "transactions_csv": output_dir / "03_rotulos_transacao.csv",
        "transactions_parquet": output_dir / "03_rotulos_transacao.parquet",
        "conflicts_csv": output_dir / "04_conflitos_rotulos.csv",
        "summary_csv": output_dir / "05_resumo_consolidacao.csv",
        "manifest": output_dir / "06_manifest_consolidacao.json",
        "methodology": output_dir / "07_metodologia_rotulos_semanticos.md",
    }
    write_csv(paths["events_csv"], EVENT_FIELDS, event_rows)
    write_csv(paths["roles_csv"], ROLE_FIELDS, role_rows)
    write_csv(paths["transactions_csv"], TRANSACTION_FIELDS, transaction_rows)
    write_csv(paths["conflicts_csv"], TRANSACTION_FIELDS, conflict_rows)
    csv_to_parquet(connection, paths["events_csv"], paths["events_parquet"])
    csv_to_parquet(connection, paths["roles_csv"], paths["roles_parquet"])
    csv_to_parquet(connection, paths["transactions_csv"], paths["transactions_parquet"])
    connection.close()

    summary_rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        selected_events = [row for row in event_rows if row["split"] == split]
        selected_transactions = [row for row in transaction_rows if row["split"] == split]
        summary_rows.append(
            {
                "split": split,
                "events": len(selected_events),
                "confirmed_insertion": sum(row["decision"] == CONFIRMED and row["adjudicated_type"] == "insertion" for row in selected_events),
                "confirmed_displacement": sum(row["decision"] == CONFIRMED and row["adjudicated_type"] == "displacement" for row in selected_events),
                "probable_displacement": sum(row["decision"] == PROBABLE and row["adjudicated_type"] == "displacement" for row in selected_events),
                "assisted_decisions": sum(row["decision_source"] == "assisted_deterministic" for row in selected_events),
                "strict_positive_transactions": sum(row["strict_label"] == 1 and not row["strict_conflict"] for row in selected_transactions),
                "strict_contextual_negatives": sum(row["strict_label"] == 0 and not row["strict_conflict"] for row in selected_transactions),
                "sensitivity_positive_transactions": sum(row["sensitivity_label"] == 1 and not row["sensitivity_conflict"] for row in selected_transactions),
                "sensitivity_contextual_negatives": sum(row["sensitivity_label"] == 0 and not row["sensitivity_conflict"] for row in selected_transactions),
                "strict_conflicting_transactions": sum(row["strict_conflict"] for row in selected_transactions),
                "sensitivity_conflicting_transactions": sum(row["sensitivity_conflict"] for row in selected_transactions),
            }
        )
    summary_fields = tuple(summary_rows[0])
    write_csv(paths["summary_csv"], summary_fields, summary_rows)

    total_confirmed = sum(row["strict_positive_event"] for row in event_rows)
    total_sensitivity = sum(row["sensitivity_positive_event"] for row in event_rows)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "splits": list(SPLIT_ORDER),
        "counts": {
            "events": len(event_rows),
            "strict_positive_events": total_confirmed,
            "sensitivity_positive_events": total_sensitivity,
            "role_rows": len(role_rows),
            "unique_split_transactions": len(transaction_rows),
            "strict_positive_transactions": sum(row["strict_label"] == 1 for row in transaction_rows),
            "strict_contextual_negatives": sum(row["strict_label"] == 0 for row in transaction_rows),
            "strict_conflicting_transactions": sum(row["strict_conflict"] for row in transaction_rows),
            "sensitivity_positive_transactions": sum(row["sensitivity_label"] == 1 for row in transaction_rows),
            "sensitivity_contextual_negatives": sum(row["sensitivity_label"] == 0 for row in transaction_rows),
            "sensitivity_conflicting_transactions": sum(row["sensitivity_conflict"] for row in transaction_rows),
            "assisted_decisions": sum(row["decision_source"] == "assisted_deterministic" for row in event_rows),
        },
        "definitions": {
            "strict": "somente eventos adjudicados como confirmado",
            "sensitivity": "eventos adjudicados como confirmado ou provável",
            "positive": "papel attacker_front ou attacker_back",
            "contextual_negative": "papel victim no mesmo evento positivo",
            "unlabeled": "qualquer transação fora dos papéis adjudicados; não presumida negativa",
            "suppression": "excluído por exigir evidência de mempool",
        },
        "split_counts": split_counts,
        "sources": sources,
        "outputs": {key: relative(path, here) for key, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["methodology"].write_text(
        "# Consolidação dos rótulos semânticos\n\n"
        "A referência **strict** contém somente eventos confirmados. A referência "
        "**sensitivity** acrescenta os casos prováveis de displacement. Insertion "
        "e displacement confirmados são mantidos separadamente pelo campo "
        "`adjudicated_type`. Suppression não recebe rótulo positivo sem evidência "
        "de mempool.\n\n"
        "Os negativos deste artefato são exclusivamente vítimas contextuais dos "
        "eventos positivos. Transações ausentes permanecem não rotuladas; portanto, "
        "o artefato não afirma que toda a população restante seja negativa. Quando "
        "um mesmo hash recebe papéis incompatíveis, o rótulo agregado fica nulo e "
        "o caso é registrado em `04_conflitos_rotulos.csv`.\n\n"
        "Conflitos no cenário de sensibilidade podem representar cadeias de "
        "displacement provável, nas quais a mesma transação é vítima de uma "
        "anterior e candidata a atacante de uma posterior. Esses casos devem ser "
        "avaliados no nível do evento ou excluídos de métricas binárias por "
        "transação; não se deve escolher um papel por precedência.\n\n"
        "A coluna `decision_source` registra `assisted_deterministic` em todas "
        "as decisões. A interface é somente leitura e não altera os rótulos "
        "produzidos pelo protocolo.\n",
        encoding="utf-8",
    )

    print(
        "Consolidação concluída: "
        f"{len(event_rows)} eventos, {len(role_rows)} papéis, "
        f"{len(transaction_rows)} transações, {len(conflict_rows)} conflitos."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
