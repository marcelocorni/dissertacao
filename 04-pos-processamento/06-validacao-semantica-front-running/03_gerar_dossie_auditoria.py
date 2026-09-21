# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Gera dossiê auditável e fluxos econômicos dos eventos semanticamente selecionados."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def load_rows(connection: duckdb.DuckDBPyConnection, path: Path) -> list[dict[str, Any]]:
    cursor = connection.execute(f"SELECT * FROM read_parquet({sql_path(path)})")
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def hex_int(value: Any) -> int:
    if value in (None, "", "0x"):
        return 0
    return int(str(value), 16)


def lower(value: Any) -> str | None:
    return str(value).lower() if value else None


def selector(transaction: dict[str, Any] | None) -> str | None:
    if not transaction:
        return None
    calldata = str(transaction.get("input_data") or "0x").lower()
    return calldata[:10] if len(calldata) >= 10 else None


def transaction_index(transaction: dict[str, Any] | None) -> int | None:
    return hex_int(transaction.get("transaction_index_hex")) if transaction else None


def transaction_status(transaction: dict[str, Any] | None) -> int | None:
    return hex_int(transaction.get("receipt_status_hex")) if transaction else None


def gas_cost(transaction: dict[str, Any] | None) -> int:
    if not transaction:
        return 0
    return hex_int(transaction.get("gas_used_hex")) * hex_int(
        transaction.get("effective_gas_price_hex")
    )


def transfer_rows(transaction: dict[str, Any] | None, role: str) -> list[dict[str, Any]]:
    if not transaction:
        return []
    try:
        logs = json.loads(transaction.get("logs_json") or "[]")
    except json.JSONDecodeError:
        return []
    rows: list[dict[str, Any]] = []
    for log in logs:
        topics = log.get("topics") or []
        # ERC-20 Transfer possui exatamente três tópicos. Quatro tópicos indicam ERC-721.
        if len(topics) != 3 or lower(topics[0]) != TRANSFER_TOPIC:
            continue
        data = log.get("data")
        if not data or data == "0x":
            continue
        rows.append(
            {
                "transaction_hash": lower(transaction.get("tx_hash")),
                "transaction_role": role,
                "block_number": hex_int(transaction.get("block_number_hex")),
                "token_address": lower(log.get("address")),
                "from_address": "0x" + str(topics[1])[-40:].lower(),
                "to_address": "0x" + str(topics[2])[-40:].lower(),
                "amount_raw": str(hex_int(data)),
                "log_index": hex_int(log.get("logIndex")),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    table: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    types: dict[str, str] | None = None,
) -> None:
    types = types or {}
    connection.execute(
        f"CREATE OR REPLACE TABLE {table} ("
        + ",".join(f'"{column}" {types.get(column, "VARCHAR")}' for column in columns)
        + ")"
    )
    if rows:
        placeholders = ",".join("?" for _ in columns)
        connection.executemany(
            f"INSERT INTO {table} VALUES ({placeholders})",
            [[row.get(column) for column in columns] for row in rows],
        )
    connection.execute(
        f"COPY {table} TO {sql_path(path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def main() -> int:
    here = Path(__file__).resolve().parent
    default_dir = here / "resultados" / "transicao_dencun_2024"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-events", type=Path, default=default_dir / "04_eventos_validados.parquet")
    parser.add_argument("--transactions", type=Path, default=default_dir / "01_transacoes_rpc.parquet")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()

    events_path = args.validated_events.resolve()
    transactions_path = args.transactions.resolve()
    if not events_path.is_file() or not transactions_path.is_file():
        parser.error("Informe os artefatos de eventos validados e transações RPC.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    events = load_rows(connection, events_path)
    transactions = {
        lower(row["tx_hash"]): row for row in load_rows(connection, transactions_path)
    }
    selected = [
        event
        for event in events
        if event["validation_status"] in {"confirmed", "probable"}
    ]

    dossier_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    summary = Counter()
    for event in selected:
        front = transactions.get(lower(event.get("attacker_front_hash")))
        back = transactions.get(lower(event.get("attacker_back_hash")))
        victim = transactions.get(lower(event.get("victim_hash")))
        attacker = lower(front.get("from_address")) if front else None
        executor = None
        if (
            event["detector"] == "insertion"
            and front
            and back
            and lower(front.get("to_address"))
            and lower(front.get("to_address")) == lower(back.get("to_address"))
        ):
            executor = lower(front.get("to_address"))
        economic_entity = {address for address in (attacker, executor) if address}
        transfers: list[dict[str, Any]] = []
        transfers.extend(transfer_rows(front, "attacker_front"))
        transfers.extend(transfer_rows(back, "attacker_back"))

        token_deltas: defaultdict[str, int] = defaultdict(int)
        attacker_transfer_count = 0
        for transfer in transfers:
            delta = 0
            if transfer["from_address"] in economic_entity:
                delta -= int(transfer["amount_raw"])
            if transfer["to_address"] in economic_entity:
                delta += int(transfer["amount_raw"])
            if delta:
                attacker_transfer_count += 1
                token_deltas[str(transfer["token_address"])] += delta
            flow_rows.append(
                {
                    "detection_event_id": event["detection_event_id"],
                    "detector": event["detector"],
                    "validation_status": event["validation_status"],
                    "attacker_address": attacker,
                    "attacker_executor_address": executor,
                    **transfer,
                    "attacker_entity_delta_raw": str(delta),
                    "is_weth": str(transfer["token_address"] == WETH).lower(),
                }
            )

        total_gas = gas_cost(front) + gas_cost(back)
        native_value_out = sum(
            hex_int(tx.get("value_hex"))
            for tx in (front, back)
            if tx and lower(tx.get("from_address")) == attacker
        )
        weth_delta = token_deltas.get(WETH)
        non_weth_nonzero = {
            token: amount for token, amount in token_deltas.items() if token != WETH and amount != 0
        }
        if not attacker_transfer_count:
            economic_status = "insufficient_transfer_visibility"
        elif WETH not in token_deltas:
            economic_status = "requires_token_valuation"
        elif non_weth_nonzero:
            economic_status = "mixed_assets_requires_valuation"
        elif weth_delta - total_gas > 0:
            economic_status = "weth_surplus_after_gas_observed"
        else:
            economic_status = "no_weth_surplus_after_gas"

        if event["detector"] == "displacement" and event["validation_status"] == "confirmed":
            priority = 1
        elif event["detector"] == "insertion" and economic_status != "weth_surplus_after_gas_observed":
            priority = 2
        elif event["detector"] == "insertion":
            priority = 3
        else:
            priority = 4

        row = {
            "audit_id": f"semantic__{event['detector']}__{event['detection_event_id']}",
            "audit_priority": priority,
            "detection_event_id": event["detection_event_id"],
            "dataset_id": event["dataset_id"],
            "split": event["split"],
            "detector": event["detector"],
            "validation_status": event["validation_status"],
            "semantic_evidence": event["semantic_evidence"],
            "protocols": event["protocols"],
            "pool_addresses": event["pool_addresses"],
            "attacker_address": attacker,
            "attacker_executor_address": executor,
            "economic_entity_addresses": "|".join(sorted(economic_entity)),
            "attacker_front_hash": event["attacker_front_hash"],
            "attacker_front_index": transaction_index(front),
            "attacker_front_status": transaction_status(front),
            "attacker_front_selector": selector(front),
            "attacker_back_hash": event.get("attacker_back_hash"),
            "attacker_back_index": transaction_index(back),
            "attacker_back_status": transaction_status(back),
            "attacker_back_selector": selector(back),
            "victim_hash": event.get("victim_hash"),
            "victim_index": transaction_index(victim),
            "victim_status": transaction_status(victim),
            "victim_selector": selector(victim),
            "attacker_gas_cost_wei": str(total_gas),
            "attacker_gas_cost_eth": f"{total_gas / 10**18:.18f}",
            "attacker_native_value_out_wei": str(native_value_out),
            "attacker_transfer_events": attacker_transfer_count,
            "attacker_token_deltas_raw_json": json.dumps(
                dict(sorted(token_deltas.items())), ensure_ascii=False, separators=(",", ":")
            ),
            "weth_delta_wei": str(weth_delta) if WETH in token_deltas else None,
            "weth_delta_eth": f"{weth_delta / 10**18:.18f}" if WETH in token_deltas else None,
            "weth_after_gas_wei": str(weth_delta - total_gas) if WETH in token_deltas else None,
            "weth_after_gas_eth": f"{(weth_delta - total_gas) / 10**18:.18f}" if WETH in token_deltas else None,
            "economic_status": economic_status,
            "front_etherscan_url": f"https://etherscan.io/tx/{event['attacker_front_hash']}",
            "back_etherscan_url": f"https://etherscan.io/tx/{event['attacker_back_hash']}" if event.get("attacker_back_hash") else None,
            "victim_etherscan_url": f"https://etherscan.io/tx/{event['victim_hash']}" if event.get("victim_hash") else None,
        }
        dossier_rows.append(row)
        summary[(event["detector"], event["validation_status"], economic_status)] += 1

    dossier_rows.sort(key=lambda row: (int(row["audit_priority"]), int(row["detection_event_id"])))
    flow_rows.sort(
        key=lambda row: (
            int(row["detection_event_id"]),
            str(row["transaction_role"]),
            int(row["log_index"]),
        )
    )
    dossier_columns = list(dossier_rows[0]) if dossier_rows else []
    flow_columns = list(flow_rows[0]) if flow_rows else [
        "detection_event_id", "detector", "validation_status", "attacker_address",
        "attacker_executor_address", "transaction_hash", "transaction_role", "block_number",
        "token_address", "from_address", "to_address", "amount_raw", "log_index",
        "attacker_entity_delta_raw", "is_weth",
    ]
    dossier_types = {
        "audit_priority": "UTINYINT",
        "detection_event_id": "UBIGINT",
        "attacker_front_index": "BIGINT",
        "attacker_front_status": "UTINYINT",
        "attacker_back_index": "BIGINT",
        "attacker_back_status": "UTINYINT",
        "victim_index": "BIGINT",
        "victim_status": "UTINYINT",
        "attacker_gas_cost_wei": "DECIMAL(38,0)",
        "attacker_gas_cost_eth": "DOUBLE",
        "attacker_native_value_out_wei": "DECIMAL(38,0)",
        "attacker_transfer_events": "INTEGER",
        "weth_delta_wei": "DECIMAL(38,0)",
        "weth_delta_eth": "DOUBLE",
        "weth_after_gas_wei": "DECIMAL(38,0)",
        "weth_after_gas_eth": "DOUBLE",
    }
    flow_types = {
        "detection_event_id": "UBIGINT",
        "block_number": "UBIGINT",
        "log_index": "BIGINT",
        "is_weth": "BOOLEAN",
    }
    write_parquet(
        connection, output / "08_dossie_auditoria.parquet", "dossier",
        dossier_columns, dossier_rows, dossier_types,
    )
    write_csv(output / "08_dossie_auditoria.csv", dossier_rows, dossier_columns)
    write_parquet(
        connection, output / "09_fluxos_tokens_atacante.parquet", "flows",
        flow_columns, flow_rows, flow_types,
    )
    write_csv(output / "09_fluxos_tokens_atacante.csv", flow_rows, flow_columns)

    summary_rows = [
        {
            "detector": detector,
            "validation_status": status,
            "economic_status": economic,
            "events": count,
        }
        for (detector, status, economic), count in sorted(summary.items())
    ]
    write_csv(
        output / "10_resumo_dossie.csv",
        summary_rows,
        ["detector", "validation_status", "economic_status", "events"],
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_validated_events": str(events_path),
        "source_transactions": str(transactions_path),
        "selected_statuses": ["confirmed", "probable"],
        "events_selected": len(dossier_rows),
        "token_transfer_rows": len(flow_rows),
        "method": {
            "erc20_transfer": "topic Transfer(address,address,uint256), exatamente 3 tópicos",
            "gas_cost": "gasUsed * effectiveGasPrice",
            "attacker": "EOA remetente e, somente em insertion, contrato executor comum às duas pernas externas",
            "weth": WETH,
            "limitation": "Fluxos internos de ETH e ativos sem preço não permitem afirmar lucro líquido completo.",
        },
        "outputs": [
            "08_dossie_auditoria.parquet", "08_dossie_auditoria.csv",
            "09_fluxos_tokens_atacante.parquet", "09_fluxos_tokens_atacante.csv",
            "10_resumo_dossie.csv",
        ],
    }
    (output / "11_manifest_dossie.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    connection.close()
    print(
        f"Dossiê concluído: {len(dossier_rows)} eventos; "
        f"{len(flow_rows)} transferências ERC-20 extraídas."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
