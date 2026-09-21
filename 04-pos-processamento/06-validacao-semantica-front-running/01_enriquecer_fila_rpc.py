# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2", "requests>=2.32,<3"]
# ///

"""Enriquece a fila do C# com transaction e receipt usando JSON-RPC Ethereum."""

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


class PermanentRpcError(RuntimeError):
    """Erro que não será resolvido repetindo a mesma consulta no provedor."""


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def discover_queues(path: Path) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        files = sorted(path.rglob("05_enrichment_queue.parquet"))
        if files:
            return [item.resolve() for item in files]
    raise FileNotFoundError(f"Nenhuma fila de enriquecimento encontrada em {path}")


class RpcClient:
    def __init__(self, url: str, cache_dir: Path, calls_per_second: float) -> None:
        self.url = url
        self.provider_host = urlsplit(url).hostname or "unknown"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.minimum_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "ethereum-front-running-academic-research/1.0",
            }
        )

    def transaction_and_receipt(self, tx_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
        cache_path = self.cache_dir / f"{tx_hash.lower()}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return payload["transaction"], payload["receipt"]

        request_payload = [
            {"jsonrpc": "2.0", "id": 1, "method": "eth_getTransactionByHash", "params": [tx_hash]},
            {"jsonrpc": "2.0", "id": 2, "method": "eth_getTransactionReceipt", "params": [tx_hash]},
        ]
        for attempt in range(6):
            elapsed = time.monotonic() - self.last_call
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            try:
                response = self.session.post(self.url, json=request_payload, timeout=60)
                self.last_call = time.monotonic()
                self.calls += 1
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(min(2 ** (attempt + 1), 30))
                    continue
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, list):
                    raise RuntimeError("O provedor não aceitou a requisição JSON-RPC em lote.")
                by_id = {item.get("id"): item for item in body}
                for identifier in (1, 2):
                    if identifier not in by_id:
                        raise RuntimeError(f"Resposta RPC sem o id {identifier}.")
                    if by_id[identifier].get("error"):
                        raise RuntimeError(str(by_id[identifier]["error"]))
                transaction = by_id[1].get("result")
                receipt = by_id[2].get("result")
                if not transaction:
                    raise PermanentRpcError("Transação histórica indisponível neste RPC.")
                if not receipt:
                    raise PermanentRpcError(
                        "Receipt histórico indisponível neste RPC; use um endpoint com histórico completo."
                    )
                cache_path.write_text(
                    json.dumps(
                        {
                            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                            "provider_host": self.provider_host,
                            "transaction": transaction,
                            "receipt": receipt,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return transaction, receipt
            except PermanentRpcError:
                raise
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"Falha RPC após 6 tentativas: {exc}") from exc
                time.sleep(min(2 ** (attempt + 1), 30))
        raise AssertionError("fluxo de repetição inválido")


def normalize(tx_hash: str, tx: dict[str, Any], receipt: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tx_hash.lower(),
        tx.get("blockHash"),
        tx.get("blockNumber"),
        tx.get("transactionIndex"),
        normalized(tx.get("from")),
        normalized(tx.get("to")),
        tx.get("value"),
        tx.get("gas"),
        tx.get("gasPrice"),
        tx.get("maxFeePerGas"),
        tx.get("maxPriorityFeePerGas"),
        tx.get("input"),
        receipt.get("status"),
        receipt.get("gasUsed"),
        receipt.get("effectiveGasPrice"),
        normalized(receipt.get("contractAddress")),
        json.dumps(receipt.get("logs") or [], ensure_ascii=False, separators=(",", ":")),
    )


def normalized(value: Any) -> str | None:
    return str(value).lower() if value else None


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=here.parent / "05-rotulador-front-running-csharp" / "resultados",
        help="Arquivo 05_enrichment_queue.parquet ou diretório que contenha as filas.",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "resultados")
    parser.add_argument("--cache-dir", type=Path, default=here / "cache" / "rpc")
    parser.add_argument("--rpc-url-env", default="ETH_RPC_URL")
    parser.add_argument("--calls-per-second", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rpc_url = os.environ.get(args.rpc_url_env, "").strip()
    if not rpc_url:
        parser.error(
            f"Defina {args.rpc_url_env} com o endpoint RPC. A URL e sua chave não serão gravadas."
        )
    if args.calls_per_second <= 0:
        parser.error("--calls-per-second deve ser positivo.")

    queues = discover_queues(args.queue.resolve())
    connection = duckdb.connect()
    queue_list = "[" + ",".join(sql_path(path) for path in queues) + "]"
    limit = f" LIMIT {args.limit}" if args.limit else ""
    hashes = [
        row[0]
        for row in connection.execute(
            f"SELECT DISTINCT lower(hash) AS hash FROM read_parquet({queue_list}, union_by_name=true) "
            f"WHERE hash IS NOT NULL ORDER BY hash{limit}"
        ).fetchall()
    ]
    if not hashes:
        parser.error("A fila não contém hashes.")

    client = RpcClient(rpc_url, args.cache_dir.resolve(), args.calls_per_second)
    records: list[tuple[Any, ...]] = []
    errors: list[dict[str, str]] = []
    for index, tx_hash in enumerate(hashes, start=1):
        try:
            transaction, receipt = client.transaction_and_receipt(tx_hash)
            records.append(normalize(tx_hash, transaction, receipt))
            print(f"[{index}/{len(hashes)}] OK {tx_hash}")
        except Exception as exc:  # mantém a retomada auditável
            errors.append({"tx_hash": tx_hash, "error": str(exc)})
            print(f"[{index}/{len(hashes)}] ERRO {tx_hash}: {exc}")

    connection.execute("""
        CREATE TABLE enriched (
            tx_hash VARCHAR, block_hash VARCHAR, block_number_hex VARCHAR,
            transaction_index_hex VARCHAR, from_address VARCHAR, to_address VARCHAR,
            value_hex VARCHAR, gas_hex VARCHAR, gas_price_hex VARCHAR,
            max_fee_per_gas_hex VARCHAR, max_priority_fee_per_gas_hex VARCHAR,
            input_data VARCHAR, receipt_status_hex VARCHAR, gas_used_hex VARCHAR,
            effective_gas_price_hex VARCHAR, contract_address VARCHAR, logs_json VARCHAR
        )
    """)
    if records:
        connection.executemany(
            "INSERT INTO enriched VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    parquet_path = output / "01_transacoes_rpc.parquet"
    connection.execute(
        f"COPY enriched TO {sql_path(parquet_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()

    (output / "02_erros_rpc.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_queues": [str(path) for path in queues],
        "requested_transactions": len(hashes),
        "enriched_transactions": len(records),
        "errors": len(errors),
        "rpc_calls": client.calls,
        "cache_hits": client.cache_hits,
        "rpc_provider_host": client.provider_host,
        "rpc_url_env": args.rpc_url_env,
        "rpc_url_stored": False,
        "queue_fingerprint": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
    }
    (output / "03_manifest_rpc.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Enriquecimento concluído: {len(records)}/{len(hashes)}; erros: {len(errors)}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
