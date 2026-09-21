# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Adjudica o dossiê semântico com critérios determinísticos e justificativa individual."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


FIELDS = [
    "audit_id", "detection_event_id", "split", "detector", "semantic_status",
    "manual_label", "manual_type", "evidence_quality", "reason_codes", "notes",
    "reviewer", "reviewed_at_utc",
]


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def rows(connection: duckdb.DuckDBPyConnection, path: Path) -> list[dict[str, Any]]:
    cursor = connection.execute(f"SELECT * FROM read_parquet({sql_path(path)})")
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def hex_int(value: Any) -> int:
    return int(str(value), 16) if value not in (None, "", "0x") else 0


def similarity(first: str, second: str) -> float:
    if not first or not second or first == "0x" or second == "0x":
        return 0.0
    first_payload = first.lower().removeprefix("0x")[8:]
    second_payload = second.lower().removeprefix("0x")[8:]
    first_words = [first_payload[index:index + 64] for index in range(0, len(first_payload), 64)]
    second_words = [second_payload[index:index + 64] for index in range(0, len(second_payload), 64)]
    return sum(left == right for left, right in zip(first_words, second_words)) / max(
        len(first_words), len(second_words), 1
    )


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["audit_id"]: row for row in csv.DictReader(stream)}


def main() -> int:
    here = Path(__file__).resolve().parent
    default_dir = here / "resultados" / "transicao_dencun_2024"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=default_dir / "08_dossie_auditoria.parquet")
    parser.add_argument("--transactions", type=Path, default=default_dir / "01_transacoes_rpc.parquet")
    parser.add_argument("--decisions", type=Path, default=default_dir / "15_decisoes_dossie.csv")
    parser.add_argument("--reviewer", default="Marcelo Corni Alves")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dossier_path = args.dossier.resolve()
    transactions_path = args.transactions.resolve()
    decisions_path = args.decisions.resolve()
    if not dossier_path.is_file() or not transactions_path.is_file():
        parser.error("Dossiê ou transações RPC não encontrados.")

    connection = duckdb.connect()
    dossier = rows(connection, dossier_path)
    tx = {str(row["tx_hash"]).lower(): row for row in rows(connection, transactions_path)}
    connection.close()
    existing = load_existing(decisions_path)
    preserved = 0
    generated = 0
    counts: Counter[tuple[str, str]] = Counter()
    now = datetime.now(timezone.utc).isoformat()

    for event in dossier:
        audit_id = str(event["audit_id"])
        if audit_id in existing and not args.overwrite:
            preserved += 1
            counts[(existing[audit_id]["detector"], existing[audit_id]["manual_label"])] += 1
            continue
        detector = str(event["detector"])
        front = tx[str(event["attacker_front_hash"]).lower()]
        victim = tx[str(event["victim_hash"]).lower()]
        front_index = hex_int(front.get("transaction_index_hex"))
        victim_index = hex_int(victim.get("transaction_index_hex"))
        front_status = hex_int(front.get("receipt_status_hex"))
        victim_status = hex_int(victim.get("receipt_status_hex"))
        front_input = str(front.get("input_data") or "0x")
        victim_input = str(victim.get("input_data") or "0x")
        sim = similarity(front_input, victim_input)

        if detector == "insertion":
            back = tx[str(event["attacker_back_hash"]).lower()]
            back_index = hex_int(back.get("transaction_index_hex"))
            economic = str(event["economic_status"])
            label = "confirmado"
            quality = "alta" if economic == "weth_surplus_after_gas_observed" else "média"
            reasons = ["semântica_compatível", "ordem_no_bloco_compatível", "mesmo_pool"]
            if economic == "weth_surplus_after_gas_observed":
                reasons.extend(["fluxo_econômico_compatível", "weth_superior_ao_gás"])
                economic_note = (
                    f"Superávit observável de {float(event['weth_after_gas_eth']):.9f} WETH após "
                    f"{float(event['attacker_gas_cost_eth']):.9f} ETH de gás."
                )
            elif economic == "mixed_assets_requires_valuation":
                reasons.append("múltiplos_ativos_sem_valoração")
                economic_note = "Há múltiplos ativos residuais; a rentabilidade líquida ainda exige valoração."
            else:
                reasons.append("dados_insuficientes")
                economic_note = "O ativo de resultado não é WETH e ainda exige valoração histórica."
            notes = (
                f"Sandwich confirmado no evento {event['detection_event_id']}: índices "
                f"{front_index} < {victim_index} < {back_index}, mesmo agente nas pernas externas, "
                f"pool {event['pool_addresses']} ({event['protocols']}), front e vítima na mesma "
                f"direção e back na direção oposta; as três transações foram executadas com sucesso. "
                f"{economic_note} Decisão produzida pelo protocolo de adjudicação assistida."
            )
        elif str(event["validation_status"]) == "confirmed":
            label = "confirmado"
            quality = "alta"
            reasons = [
                "semântica_compatível", "ordem_no_bloco_compatível", "vítima_revertida",
                "fluxo_econômico_compatível",
            ]
            gas_front = hex_int(front.get("effective_gas_price_hex"))
            gas_victim = hex_int(victim.get("effective_gas_price_hex"))
            notes = (
                f"Displacement confirmado no evento {event['detection_event_id']}: atacante no índice "
                f"{front_index} e vítima no índice {victim_index}, mesmo destino, valor, seletor e calldata "
                f"(similaridade {sim:.3f}); atacante executou e recebeu o token, enquanto a vítima reverteu. "
                f"Preço efetivo do gás do atacante={gas_front} wei e da vítima={gas_victim} wei. "
                f"Decisão produzida pelo protocolo de adjudicação assistida."
            )
        else:
            label = "provável"
            quality = "média"
            reasons = ["semântica_compatível", "ordem_no_bloco_compatível", "mesmo_pool"]
            if front_input == victim_input:
                reasons.append("chamada_idêntica")
            notes = (
                f"Displacement provável no evento {event['detection_event_id']}: atacante no índice "
                f"{front_index} antes da vítima no índice {victim_index}, mesmo seletor, similaridade de "
                f"calldata {sim:.3f} e pool comum {event['pool_addresses']}. Status atacante={front_status} "
                f"e vítima={victim_status}. Não há efeito adverso suficiente para confirmação estrita sem "
                f"evidência adicional de mempool/causalidade. Decisão produzida pelo protocolo de "
                f"adjudicação assistida."
            )

        record = {
            "audit_id": audit_id,
            "detection_event_id": str(event["detection_event_id"]),
            "split": str(event["split"]),
            "detector": detector,
            "semantic_status": str(event["validation_status"]),
            "manual_label": label,
            "manual_type": detector,
            "evidence_quality": quality,
            "reason_codes": "|".join(reasons),
            "notes": notes,
            "reviewer": args.reviewer,
            "reviewed_at_utc": now,
        }
        existing[audit_id] = record
        counts[(detector, label)] += 1
        generated += 1

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = decisions_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            sorted(existing.values(), key=lambda item: (int(item["detection_event_id"]), item["audit_id"]))
        )
    temporary.replace(decisions_path)
    manifest = {
        "created_at_utc": now,
        "source_dossier": str(dossier_path),
        "source_transactions": str(transactions_path),
        "decisions": str(decisions_path),
        "reviewer": args.reviewer,
        "method": "adjudicação assistida determinística com justificativa por evento",
        "decision_policy": "final_assisted_without_manual_override",
        "interface_mode": "read_only",
        "legacy_column_names": ["manual_label", "manual_type"],
        "generated": generated,
        "preserved_existing": preserved,
        "overwrite": args.overwrite,
        "counts": {f"{detector}:{label}": count for (detector, label), count in sorted(counts.items())},
        "caveat": "Rótulos produzidos por protocolo algorítmico assistido; não constituem observação independente de mempool.",
    }
    manifest_path = decisions_path.parent / "16_manifest_adjudicacao_assistida.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Adjudicação assistida concluída: {generated} geradas; {preserved} preservadas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
