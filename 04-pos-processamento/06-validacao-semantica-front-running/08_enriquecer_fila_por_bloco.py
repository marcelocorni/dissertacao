# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2", "requests>=2.32,<3"]
# ///

"""Enriquece uma fila por bloco com eth_getBlockByNumber e eth_getBlockReceipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import duckdb
import requests


def q(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def normalized(value: Any) -> str | None:
    return str(value).lower() if value else None


def normalize(tx: dict[str, Any], receipt: dict[str, Any]) -> tuple[Any, ...]:
    return (
        lower(tx["hash"]), tx.get("blockHash"), tx.get("blockNumber"), tx.get("transactionIndex"),
        normalized(tx.get("from")), normalized(tx.get("to")), tx.get("value"), tx.get("gas"),
        tx.get("gasPrice"), tx.get("maxFeePerGas"), tx.get("maxPriorityFeePerGas"), tx.get("input"),
        receipt.get("status"), receipt.get("gasUsed"), receipt.get("effectiveGasPrice"),
        normalized(receipt.get("contractAddress")),
        json.dumps(receipt.get("logs") or [], ensure_ascii=False, separators=(",", ":")),
    )


def lower(value: Any) -> str:
    return str(value).lower()


class BlockRpcClient:
    def __init__(self, url: str, requests_per_second: float) -> None:
        self.url = url
        self.provider_host = urlsplit(url).hostname or "unknown"
        self.minimum_interval = 1.0 / requests_per_second
        self.last_call = 0.0
        self.http_requests = 0
        self.rpc_methods = 0
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "User-Agent": "ethereum-front-running-academic-research/1.0"})

    def blocks(self, block_numbers: list[int]) -> dict[int, tuple[dict[str, Any], list[dict[str, Any]]]]:
        payload = []
        for block in block_numbers:
            tag = hex(block)
            payload.extend([
                {"jsonrpc": "2.0", "id": f"{block}:block", "method": "eth_getBlockByNumber", "params": [tag, True]},
                {"jsonrpc": "2.0", "id": f"{block}:receipts", "method": "eth_getBlockReceipts", "params": [tag]},
            ])
        for attempt in range(6):
            elapsed = time.monotonic() - self.last_call
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            try:
                response = self.session.post(self.url, json=payload, timeout=180)
                self.last_call = time.monotonic(); self.http_requests += 1; self.rpc_methods += len(payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt == 5:
                        raise RuntimeError(
                            f"HTTP transitório persistente {response.status_code} após 6 tentativas"
                        )
                    time.sleep(min(2 ** (attempt + 1), 30)); continue
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, list):
                    raise RuntimeError("O endpoint não aceitou a requisição JSON-RPC em lote.")
                by_id = {str(item.get("id")): item for item in body}
                result = {}
                for block in block_numbers:
                    block_item = by_id.get(f"{block}:block", {})
                    receipt_item = by_id.get(f"{block}:receipts", {})
                    if block_item.get("error") or receipt_item.get("error"):
                        raise RuntimeError(f"Erro RPC no bloco {block}: {block_item.get('error') or receipt_item.get('error')}")
                    block_result, receipt_result = block_item.get("result"), receipt_item.get("result")
                    if not block_result or not isinstance(receipt_result, list):
                        raise RuntimeError(f"Bloco ou receipts históricos indisponíveis: {block}")
                    result[block] = (block_result, receipt_result)
                return result
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"Falha RPC após 6 tentativas: {exc}") from exc
                time.sleep(min(2 ** (attempt + 1), 30))
        raise RuntimeError("Falha RPC sem resposta válida após todas as tentativas")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=here / "cache" / "blocks")
    parser.add_argument("--rpc-url-env", default="ETH_RPC_URL")
    parser.add_argument("--blocks-per-batch", type=int, default=5)
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--limit-blocks", type=int)
    args = parser.parse_args()
    rpc_url = os.environ.get(args.rpc_url_env, "").strip()
    if not rpc_url:
        parser.error(f"Defina {args.rpc_url_env} com o endpoint RPC.")
    queue = args.queue.resolve()
    if not queue.is_file():
        parser.error(f"Fila não encontrada: {queue}")
    if args.blocks_per_batch < 1 or args.requests_per_second <= 0:
        parser.error("Tamanho do lote e taxa devem ser positivos.")
    dataset = args.dataset_name or queue.parent.name
    output = (args.output_dir or here / "resultados-blocos" / dataset).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = (args.cache_root.resolve() / dataset); cache.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    source_rows = connection.execute(
        f"SELECT block_number, list(lower(hash) ORDER BY lower(hash)) hashes FROM read_parquet({q(queue)}) "
        "WHERE hash IS NOT NULL GROUP BY 1 ORDER BY 1"
    ).fetchall()
    if args.limit_blocks:
        source_rows = source_rows[: args.limit_blocks]
    requested = {int(block): list(hashes) for block, hashes in source_rows}
    client = BlockRpcClient(rpc_url, args.requests_per_second)
    pending = []
    cache_hits = 0
    for block, hashes in requested.items():
        path = cache / f"{block}.json"
        if path.is_file():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if sorted(saved.get("requested_hashes", [])) == sorted(hashes):
                    cache_hits += 1; continue
            except (ValueError, OSError):
                pass
        pending.append(block)

    errors = []

    def persist_block(
        block: int, block_data: dict[str, Any], receipts: list[dict[str, Any]]
    ) -> None:
        tx_by_hash = {lower(item["hash"]): item for item in block_data.get("transactions", [])}
        receipt_by_hash = {lower(item["transactionHash"]): item for item in receipts}
        missing = [
            hash_value for hash_value in requested[block]
            if hash_value not in tx_by_hash or hash_value not in receipt_by_hash
        ]
        if missing:
            raise RuntimeError(f"Bloco {block}: {len(missing)} hashes da fila ausentes")
        selected_records = [
            {"transaction": tx_by_hash[hash_value], "receipt": receipt_by_hash[hash_value]}
            for hash_value in requested[block]
        ]
        (cache / f"{block}.json").write_text(
            json.dumps(
                {
                    "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                    "provider_host": client.provider_host,
                    "block_number": block,
                    "block_hash": block_data.get("hash"),
                    "requested_hashes": requested[block],
                    "records": selected_records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    for start in range(0, len(pending), args.blocks_per_batch):
        batch = pending[start : start + args.blocks_per_batch]
        try:
            results = client.blocks(batch)
            for block in batch:
                block_data, receipts = results[block]
                persist_block(block, block_data, receipts)
            print(f"[{min(start + len(batch), len(pending))}/{len(pending)} blocos novos] OK {batch[0]}..{batch[-1]}")
        except Exception as exc:
            print(
                f"AVISO lote {batch[0]}..{batch[-1]}: {exc}; "
                "tentando os blocos individualmente"
            )
            for block in batch:
                try:
                    block_data, receipts = client.blocks([block])[block]
                    persist_block(block, block_data, receipts)
                    print(f"RECUPERADO bloco {block}")
                except Exception as block_exc:
                    errors.append({"block_number": block, "error": str(block_exc)})
                    print(f"ERRO bloco {block}: {block_exc}")

    records = []
    for block, hashes in requested.items():
        path = cache / f"{block}.json"
        if not path.is_file():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        if sorted(saved.get("requested_hashes", [])) != sorted(hashes):
            continue
        records.extend(normalize(item["transaction"], item["receipt"]) for item in saved["records"])
    connection.execute("""CREATE TABLE enriched (
        tx_hash VARCHAR, block_hash VARCHAR, block_number_hex VARCHAR, transaction_index_hex VARCHAR,
        from_address VARCHAR, to_address VARCHAR, value_hex VARCHAR, gas_hex VARCHAR, gas_price_hex VARCHAR,
        max_fee_per_gas_hex VARCHAR, max_priority_fee_per_gas_hex VARCHAR, input_data VARCHAR,
        receipt_status_hex VARCHAR, gas_used_hex VARCHAR, effective_gas_price_hex VARCHAR,
        contract_address VARCHAR, logs_json VARCHAR)""")
    if records:
        connection.executemany("INSERT INTO enriched VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)
    parquet = output / "01_transacoes_rpc.parquet"
    connection.execute(f"COPY enriched TO {q(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    connection.close()
    (output / "02_erros_rpc.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    all_hashes = sorted(hash_value for hashes in requested.values() for hash_value in hashes)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "block_batch",
        "dataset": dataset, "source_queue": str(queue), "requested_blocks": len(requested),
        "requested_transactions": len(all_hashes), "enriched_transactions": len(records),
        "errors": len(errors), "cache_hits_blocks": cache_hits, "new_blocks": len(pending),
        "http_requests": client.http_requests, "rpc_methods": client.rpc_methods,
        "blocks_per_batch": args.blocks_per_batch, "rpc_provider_host": client.provider_host,
        "rpc_url_stored": False,
        "queue_fingerprint": hashlib.sha256("\n".join(all_hashes).encode()).hexdigest(),
    }
    (output / "03_manifest_rpc.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Enriquecimento por bloco: {len(records)}/{len(all_hashes)} transações; erros de bloco: {len(errors)}")
    return 0 if not errors and len(records) == len(all_hashes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
