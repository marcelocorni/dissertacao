# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Reconstrói caches reutilizáveis a partir dos Parquets RPC já validados."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb


RESULT_COLUMNS = [
    "tx_hash", "block_hash", "block_number_hex", "transaction_index_hex",
    "from_address", "to_address", "value_hex", "gas_hex", "gas_price_hex",
    "max_fee_per_gas_hex", "max_priority_fee_per_gas_hex", "input_data",
    "receipt_status_hex", "gas_used_hex", "effective_gas_price_hex",
    "contract_address", "logs_json",
]


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def payload_from_row(row: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    values = dict(zip(RESULT_COLUMNS, row, strict=True))
    transaction = {
        "hash": values["tx_hash"],
        "blockHash": values["block_hash"],
        "blockNumber": values["block_number_hex"],
        "transactionIndex": values["transaction_index_hex"],
        "from": values["from_address"],
        "to": values["to_address"],
        "value": values["value_hex"],
        "gas": values["gas_hex"],
        "gasPrice": values["gas_price_hex"],
        "maxFeePerGas": values["max_fee_per_gas_hex"],
        "maxPriorityFeePerGas": values["max_priority_fee_per_gas_hex"],
        "input": values["input_data"],
    }
    receipt = {
        "transactionHash": values["tx_hash"],
        "blockHash": values["block_hash"],
        "blockNumber": values["block_number_hex"],
        "transactionIndex": values["transaction_index_hex"],
        "from": values["from_address"],
        "to": values["to_address"],
        "status": values["receipt_status_hex"],
        "gasUsed": values["gas_used_hex"],
        "effectiveGasPrice": values["effective_gas_price_hex"],
        "contractAddress": values["contract_address"],
        "logs": json.loads(values["logs_json"] or "[]"),
    }
    return transaction, receipt


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return True


def result_splits(results_root: Path) -> Iterable[Path]:
    for directory in sorted(results_root.iterdir()):
        if directory.is_dir() and (directory / "01_transacoes_rpc.parquet").is_file():
            yield directory


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=here / "resultados")
    parser.add_argument("--cache-root", type=Path, default=here / "cache")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    connection = duckdb.connect()
    summary: dict[str, Any] = {
        "created_at_utc": created_at,
        "origin": "reconstructed_from_validated_parquet",
        "network_calls": 0,
        "splits": {},
    }

    for split_dir in result_splits(results_root):
        split = split_dir.name
        parquet = split_dir / "01_transacoes_rpc.parquet"
        manifest_path = split_dir / "03_manifest_rpc.json"
        provider = "reconstructed"
        if manifest_path.is_file():
            provider = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "rpc_provider_host", provider
            )

        query = (
            f"SELECT {', '.join(RESULT_COLUMNS)} FROM read_parquet({sql_path(parquet)}) "
            "ORDER BY CAST(block_number_hex AS UBIGINT), lower(tx_hash)"
        )
        cursor = connection.execute(query)
        block_number: int | None = None
        block_hash: str | None = None
        block_records: list[dict[str, Any]] = []
        block_hashes: list[str] = []
        blocks_written = 0
        transactions = 0

        def flush_block() -> None:
            nonlocal blocks_written
            if block_number is None:
                return
            target = cache_root / "blocks" / split / f"{block_number}.json"
            written = write_json(
                target,
                {
                    "fetched_at_utc": created_at,
                    "provider_host": provider,
                    "block_number": block_number,
                    "block_hash": block_hash,
                    "requested_hashes": block_hashes,
                    "records": block_records,
                    "cache_origin": "reconstructed_from_validated_parquet",
                },
                args.overwrite,
            )
            blocks_written += int(written)

        while rows := cursor.fetchmany(1000):
            for row in rows:
                transaction, receipt = payload_from_row(row)
                current = int(transaction["blockNumber"], 16)
                if block_number is not None and current != block_number:
                    flush_block()
                    block_records = []
                    block_hashes = []
                block_number = current
                block_hash = transaction["blockHash"]
                block_hashes.append(transaction["hash"].lower())
                block_records.append({"transaction": transaction, "receipt": receipt})
                transactions += 1
        flush_block()

        # O coletor transação-a-transação foi usado como baseline em Dencun.
        rpc_written = 0
        if split == "transicao_dencun_2024":
            for row in connection.execute(
                f"SELECT {', '.join(RESULT_COLUMNS)} FROM read_parquet({sql_path(parquet)})"
            ).fetchall():
                transaction, receipt = payload_from_row(row)
                target = cache_root / "rpc" / f"{transaction['hash'].lower()}.json"
                rpc_written += int(
                    write_json(
                        target,
                        {
                            "fetched_at_utc": created_at,
                            "provider_host": provider,
                            "transaction": transaction,
                            "receipt": receipt,
                            "cache_origin": "reconstructed_from_validated_parquet",
                        },
                        args.overwrite,
                    )
                )

        token_parquet = split_dir / "12_metadados_tokens.parquet"
        tokens_written = 0
        if token_parquet.is_file():
            token_rows = connection.execute(
                "SELECT token_address, reference_block, symbol, name, decimals, "
                f"provider_host FROM read_parquet({sql_path(token_parquet)})"
            ).fetchall()
            for token, reference_block, symbol, name, decimals, token_provider in token_rows:
                target = cache_root / "token_metadata" / split / f"{str(token).lower()}.json"
                tokens_written += int(
                    write_json(
                        target,
                        {
                            "token_address": str(token).lower(),
                            "block_number": int(reference_block),
                            "decimals": decimals,
                            "symbol": symbol,
                            "name": name,
                            "raw": {},
                            "rpc_errors": {},
                            "fetched_at_utc": created_at,
                            "provider_host": token_provider or provider,
                            "cache_origin": "reconstructed_from_validated_parquet",
                        },
                        args.overwrite,
                    )
                )

        fingerprint = hashlib.sha256(parquet.read_bytes()).hexdigest()
        summary["splits"][split] = {
            "source_parquet": str(parquet),
            "source_sha256": fingerprint,
            "transactions": transactions,
            "block_files_written": blocks_written,
            "transaction_files_written": rpc_written,
            "token_files_written": tokens_written,
        }
        print(
            f"{split}: {transactions} transações; {blocks_written} blocos; "
            f"{rpc_written} caches individuais; {tokens_written} tokens"
        )

    manifest = cache_root / "00_manifest_cache_reconstruido.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    connection.close()
    print(f"Cache reconstruído sem rede: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
