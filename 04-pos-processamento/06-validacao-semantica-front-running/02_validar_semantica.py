# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Valida semanticamente candidatos C# v3 usando calldata, receipts e logs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


V2_SWAP = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
V3_SWAP = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def discover(path: Path, filename: str) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        files = sorted(path.rglob(filename))
        if files:
            return [item.resolve() for item in files]
    raise FileNotFoundError(f"Nenhum {filename} encontrado em {path}")


def lower(value: Any) -> str:
    return "" if value is None else str(value).lower()


def parse_logs(tx: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return json.loads(str(tx.get("logs_json") or "[]"))
    except json.JSONDecodeError:
        return []


def uint_words(data: Any) -> list[int]:
    payload = lower(data).removeprefix("0x")
    return [
        int(payload[index:index + 64], 16)
        for index in range(0, len(payload), 64)
        if len(payload[index:index + 64]) == 64
    ]


def signed(value: int) -> int:
    return value - (1 << 256) if value >= (1 << 255) else value


def swap_deltas(tx: dict[str, Any]) -> dict[tuple[str, str], tuple[int, int]]:
    result: dict[tuple[str, str], tuple[int, int]] = {}
    for log in parse_logs(tx):
        topics = log.get("topics") or []
        topic0 = lower(topics[0]) if topics else ""
        address = lower(log.get("address"))
        values = uint_words(log.get("data"))
        if topic0 == V2_SWAP and len(values) >= 4:
            result[(address, "uniswap_v2")] = (
                values[0] - values[2], values[1] - values[3]
            )
        elif topic0 == V3_SWAP and len(values) >= 2:
            result[(address, "uniswap_v3")] = (signed(values[0]), signed(values[1]))
    return result


def direction(delta: tuple[int, int]) -> int | None:
    if delta[0] > 0 and delta[1] < 0:
        return 0
    if delta[1] > 0 and delta[0] < 0:
        return 1
    return None


def calldata_similarity(first: str, second: str) -> float:
    if not first or not second or first == "0x" or second == "0x":
        return 0.0
    first_payload = first.lower().removeprefix("0x")[8:]
    second_payload = second.lower().removeprefix("0x")[8:]
    first_words = [first_payload[index:index + 64] for index in range(0, len(first_payload), 64)]
    second_words = [second_payload[index:index + 64] for index in range(0, len(second_payload), 64)]
    denominator = max(len(first_words), len(second_words), 1)
    matches = sum(left == right for left, right in zip(first_words, second_words))
    return matches / denominator


def successful(tx: dict[str, Any]) -> bool:
    return lower(tx.get("receipt_status_hex")) == "0x1"


def load_rows(connection: duckdb.DuckDBPyConnection, files: list[Path]) -> list[dict[str, Any]]:
    source = "[" + ",".join(sql_path(path) for path in files) + "]"
    cursor = connection.execute(f"SELECT * FROM read_parquet({source}, union_by_name=true)")
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def validate_insertion(
    event: dict[str, Any], transactions: dict[str, dict[str, Any]]
) -> tuple[str, str, list[str], list[str]]:
    hashes = [event.get("attacker_front_hash"), event.get("victim_hash"), event.get("attacker_back_hash")]
    if any(lower(item) not in transactions for item in hashes):
        return "pending_enrichment", "Transações necessárias ainda não foram enriquecidas.", [], []
    front, victim, back = (transactions[lower(item)] for item in hashes)
    if not all(successful(item) for item in (front, victim, back)):
        return "rejected", "Ao menos uma das três transações não foi executada com sucesso.", [], []
    if lower(front.get("from_address")) != lower(back.get("from_address")):
        return "rejected", "As pernas externas não possuem o mesmo remetente.", [], []

    front_swaps, victim_swaps, back_swaps = (
        swap_deltas(front), swap_deltas(victim), swap_deltas(back)
    )
    matched: list[tuple[str, str]] = []
    for key in sorted(set(front_swaps) & set(victim_swaps) & set(back_swaps)):
        front_direction = direction(front_swaps[key])
        victim_direction = direction(victim_swaps[key])
        back_direction = direction(back_swaps[key])
        if (
            front_direction is not None
            and front_direction == victim_direction
            and back_direction is not None
            and back_direction != front_direction
        ):
            matched.append(key)
    if not matched:
        return (
            "rejected",
            "Não há swap V2/V3 no mesmo pool com front e vítima na mesma direção e back na direção oposta.",
            [],
            [],
        )
    pools = sorted({key[0] for key in matched})
    protocols = sorted({key[1] for key in matched})
    return (
        "confirmed",
        "Estrutura sandwich confirmada: transações consecutivas, mesmo agente nas pernas externas, mesmo pool, front e vítima na mesma direção e back na direção oposta. O resultado líquido após custos será calculado em etapa própria.",
        pools,
        protocols,
    )


def validate_displacement(
    event: dict[str, Any], transactions: dict[str, dict[str, Any]]
) -> tuple[str, str, list[str], list[str]]:
    attacker_hash, victim_hash = event.get("attacker_front_hash"), event.get("victim_hash")
    if lower(attacker_hash) not in transactions or lower(victim_hash) not in transactions:
        return "pending_enrichment", "Transações necessárias ainda não foram enriquecidas.", [], []
    attacker, victim = transactions[lower(attacker_hash)], transactions[lower(victim_hash)]
    attacker_input = lower(attacker.get("input_data"))
    victim_input = lower(victim.get("input_data"))
    same_selector = len(attacker_input) >= 10 and attacker_input[:10] == victim_input[:10]
    similarity = calldata_similarity(attacker_input, victim_input)
    common_swaps = sorted(set(swap_deltas(attacker)) & set(swap_deltas(victim)))
    if same_selector and similarity >= 0.80 and successful(attacker) and not successful(victim):
        return (
            "confirmed",
            f"Atacante anterior executou com sucesso e a vítima falhou; mesmo seletor e similaridade de calldata {similarity:.3f}.",
            sorted({key[0] for key in common_swaps}),
            sorted({key[1] for key in common_swaps}),
        )
    if same_selector and similarity >= 0.80 and common_swaps and successful(attacker):
        return (
            "probable",
            f"Ordem, seletor, calldata e pool são compatíveis, mas não foi comprovado efeito adverso suficiente; similaridade {similarity:.3f}.",
            sorted({key[0] for key in common_swaps}),
            sorted({key[1] for key in common_swaps}),
        )
    return (
        "rejected",
        f"Sem equivalência semântica suficiente: mesmo seletor={same_selector}; similaridade de calldata={similarity:.3f}; pools comuns={len(common_swaps)}.",
        [],
        [],
    )


def validate_event(
    event: dict[str, Any], transactions: dict[str, dict[str, Any]]
) -> tuple[str, str, list[str], list[str]]:
    detector = str(event["detector"])
    if detector == "insertion":
        return validate_insertion(event, transactions)
    if detector == "displacement":
        return validate_displacement(event, transactions)
    return (
        "requires_mempool_evidence",
        "Suppression não pode ser confirmado somente por blocos minerados; é necessária evidência temporal de mempool.",
        [],
        [],
    )


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events",
        type=Path,
        default=here.parent / "05-rotulador-front-running-csharp" / "resultados",
    )
    parser.add_argument("--transactions", type=Path, default=here / "resultados" / "01_transacoes_rpc.parquet")
    parser.add_argument("--output-dir", type=Path, default=here / "resultados")
    args = parser.parse_args()

    event_files = discover(args.events.resolve(), "01_detection_events.parquet")
    transaction_files = discover(args.transactions.resolve(), "01_transacoes_rpc.parquet") if args.transactions.is_dir() else [args.transactions.resolve()]
    if not all(path.is_file() for path in transaction_files):
        parser.error("Arquivo de transações enriquecidas não encontrado.")

    connection = duckdb.connect()
    events = load_rows(connection, event_files)
    transaction_rows = load_rows(connection, transaction_files)
    transactions = {lower(row["tx_hash"]): row for row in transaction_rows}

    validations: list[tuple[Any, ...]] = []
    roles: list[tuple[Any, ...]] = []
    status_counts: Counter[tuple[str, str]] = Counter()
    now = datetime.now(timezone.utc).isoformat()
    for event in events:
        status, evidence, pools, protocols = validate_event(event, transactions)
        status_counts[(str(event["detector"]), status)] += 1
        validations.append(
            (
                event["detection_event_id"], event["dataset_id"], event["split"],
                event["detector"], event["candidate_hash"], event["attacker_front_hash"],
                event.get("attacker_back_hash"), event.get("victim_hash"), status,
                "|".join(pools), "|".join(protocols), evidence, now,
            )
        )
        if status == "confirmed":
            roles.append((event["dataset_id"], event["split"], event["detector"], event["attacker_front_hash"], "attacker_front", 1, event["detection_event_id"]))
            if event.get("attacker_back_hash"):
                roles.append((event["dataset_id"], event["split"], event["detector"], event["attacker_back_hash"], "attacker_back", 1, event["detection_event_id"]))
            if event.get("victim_hash"):
                roles.append((event["dataset_id"], event["split"], event["detector"], event["victim_hash"], "victim", 0, event["detection_event_id"]))

    connection.execute("""
        CREATE TABLE validated_events (
            detection_event_id UBIGINT, dataset_id VARCHAR, split VARCHAR,
            detector VARCHAR, candidate_hash VARCHAR, attacker_front_hash VARCHAR,
            attacker_back_hash VARCHAR, victim_hash VARCHAR, validation_status VARCHAR,
            pool_addresses VARCHAR, protocols VARCHAR, semantic_evidence VARCHAR,
            validated_at_utc VARCHAR
        )
    """)
    if validations:
        connection.executemany("INSERT INTO validated_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", validations)
    connection.execute("""
        CREATE TABLE confirmed_roles (
            dataset_id VARCHAR, split VARCHAR, detector VARCHAR, hash VARCHAR,
            confirmed_role VARCHAR, label UTINYINT, detection_event_id UBIGINT
        )
    """)
    if roles:
        connection.executemany("INSERT INTO confirmed_roles VALUES (?,?,?,?,?,?,?)", roles)

    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY validated_events TO {sql_path(output / '04_eventos_validados.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
    connection.execute(f"COPY confirmed_roles TO {sql_path(output / '05_papeis_confirmados.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
    connection.execute(f"""COPY (
        SELECT detector, validation_status, count(*)::BIGINT AS events
        FROM validated_events GROUP BY ALL ORDER BY detector, validation_status
    ) TO {sql_path(output / '06_resumo_validacao.csv')} (HEADER, DELIMITER ',')""")
    connection.close()

    manifest = {
        "created_at_utc": now,
        "source_event_files": [str(path) for path in event_files],
        "source_transaction_files": [str(path) for path in transaction_files],
        "events": len(events),
        "enriched_transactions_available": len(transactions),
        "confirmed_role_rows": len(roles),
        "status_counts": {f"{detector}:{status}": count for (detector, status), count in sorted(status_counts.items())},
        "methodological_note": "Confirmed exige evidência semântica. Suppression permanece sem ground truth sem mempool. Resultado líquido do sandwich é etapa posterior.",
    }
    (output / "07_manifest_validacao.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Validação concluída: {len(events)} eventos; {sum(count for (detector, status), count in status_counts.items() if status == 'confirmed')} confirmados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
