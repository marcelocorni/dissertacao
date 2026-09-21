# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2", "requests>=2.32,<3"]
# ///

"""Obtém símbolo, nome e decimais dos tokens ERC-20 presentes no dossiê."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import duckdb
import requests


CALLS = {
    1: ("decimals", "0x313ce567"),
    2: ("symbol", "0x95d89b41"),
    3: ("name", "0x06fdde03"),
}


def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def decode_text(result: str | None) -> str | None:
    if not result or result == "0x":
        return None
    try:
        payload = bytes.fromhex(result[2:])
        if len(payload) >= 64:
            offset = int.from_bytes(payload[:32], "big")
            if offset + 32 <= len(payload):
                length = int.from_bytes(payload[offset : offset + 32], "big")
                end = offset + 32 + length
                if end <= len(payload):
                    return payload[offset + 32 : end].decode("utf-8", errors="replace").strip("\x00")
        return payload[:32].rstrip(b"\x00").decode("utf-8", errors="replace") or None
    except (ValueError, UnicodeError):
        return None


def decode_decimals(result: str | None) -> int | None:
    if not result or result == "0x":
        return None
    try:
        value = int(result, 16)
        return value if 0 <= value <= 255 else None
    except ValueError:
        return None


class RpcClient:
    def __init__(self, url: str, calls_per_second: float) -> None:
        self.url = url
        self.provider_host = urlsplit(url).hostname or "unknown"
        self.minimum_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self.calls = 0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "ethereum-front-running-academic-research/1.0",
            }
        )

    def metadata(self, token: str, block_number: int) -> dict[str, Any]:
        block_tag = hex(block_number)
        request_body = [
            {
                "jsonrpc": "2.0",
                "id": identifier,
                "method": "eth_call",
                "params": [{"to": token, "data": calldata}, block_tag],
            }
            for identifier, (_, calldata) in CALLS.items()
        ]
        for attempt in range(6):
            elapsed = time.monotonic() - self.last_call
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            try:
                response = self.session.post(self.url, json=request_body, timeout=60)
                self.last_call = time.monotonic()
                self.calls += 1
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(min(2 ** (attempt + 1), 30))
                    continue
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, list):
                    raise RuntimeError("O endpoint não aceitou o lote de eth_call.")
                by_id = {item.get("id"): item for item in body}
                raw: dict[str, str | None] = {}
                errors: dict[str, Any] = {}
                for identifier, (field, _) in CALLS.items():
                    item = by_id.get(identifier, {})
                    raw[field] = item.get("result")
                    if item.get("error"):
                        errors[field] = item["error"]
                return {
                    "token_address": token,
                    "block_number": block_number,
                    "decimals": decode_decimals(raw["decimals"]),
                    "symbol": decode_text(raw["symbol"]),
                    "name": decode_text(raw["name"]),
                    "raw": raw,
                    "rpc_errors": errors,
                }
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt == 5:
                    raise RuntimeError(f"Falha RPC após 6 tentativas: {exc}") from exc
                time.sleep(min(2 ** (attempt + 1), 30))
        raise AssertionError("fluxo de repetição inválido")


def main() -> int:
    here = Path(__file__).resolve().parent
    default_dir = here / "resultados" / "transicao_dencun_2024"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows", type=Path, default=default_dir / "09_fluxos_tokens_atacante.parquet")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--cache-dir", type=Path, default=here / "cache" / "token_metadata")
    parser.add_argument("--rpc-url-env", default="ETH_RPC_URL")
    parser.add_argument("--calls-per-second", type=float, default=2.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rpc_url = os.environ.get(args.rpc_url_env, "").strip()
    if not rpc_url:
        parser.error(f"Defina {args.rpc_url_env} com o endpoint RPC.")
    flows = args.flows.resolve()
    if not flows.is_file():
        parser.error(f"Fluxos não encontrados: {flows}")
    if args.calls_per_second <= 0:
        parser.error("--calls-per-second deve ser positivo.")

    connection = duckdb.connect()
    limit = f" LIMIT {args.limit}" if args.limit else ""
    tokens = connection.execute(
        f"SELECT lower(token_address), min(block_number)::UBIGINT "
        f"FROM read_parquet({sql_path(flows)}) GROUP BY 1 ORDER BY 1{limit}"
    ).fetchall()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = RpcClient(rpc_url, args.calls_per_second)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    cache_hits = 0
    for index, (token, block_number) in enumerate(tokens, start=1):
        cache_path = cache_dir / f"{token}.json"
        try:
            if cache_path.is_file():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cache_hits += 1
            else:
                payload = client.metadata(token, int(block_number))
                payload.update(
                    {
                        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                        "provider_host": client.provider_host,
                    }
                )
                cache_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            status = "complete" if payload.get("decimals") is not None and payload.get("symbol") else "partial"
            records.append(
                {
                    "token_address": token,
                    "reference_block": int(block_number),
                    "symbol": payload.get("symbol"),
                    "name": payload.get("name"),
                    "decimals": payload.get("decimals"),
                    "metadata_status": status,
                    "provider_host": payload.get("provider_host", client.provider_host),
                }
            )
            if status == "partial":
                errors.append({"token_address": token, "error": "Metadados ERC-20 incompletos"})
            print(f"[{index}/{len(tokens)}] {status.upper()} {token} {payload.get('symbol') or ''}")
        except Exception as exc:
            errors.append({"token_address": token, "error": str(exc)})
            records.append(
                {
                    "token_address": token,
                    "reference_block": int(block_number),
                    "symbol": None,
                    "name": None,
                    "decimals": None,
                    "metadata_status": "error",
                    "provider_host": client.provider_host,
                }
            )
            print(f"[{index}/{len(tokens)}] ERRO {token}: {exc}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "token_address", "reference_block", "symbol", "name", "decimals",
        "metadata_status", "provider_host",
    ]
    connection.execute(
        "CREATE TABLE metadata (token_address VARCHAR, reference_block UBIGINT, "
        "symbol VARCHAR, name VARCHAR, decimals UTINYINT, metadata_status VARCHAR, provider_host VARCHAR)"
    )
    if records:
        connection.executemany(
            "INSERT INTO metadata VALUES (?,?,?,?,?,?,?)",
            [[record[column] for column in columns] for record in records],
        )
    parquet_path = output / "12_metadados_tokens.parquet"
    connection.execute(
        f"COPY metadata TO {sql_path(parquet_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    with (output / "12_metadados_tokens.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    (output / "13_erros_metadados_tokens.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_flows": str(flows),
        "tokens_requested": len(tokens),
        "complete": sum(record["metadata_status"] == "complete" for record in records),
        "partial": sum(record["metadata_status"] == "partial" for record in records),
        "errors": sum(record["metadata_status"] == "error" for record in records),
        "rpc_calls": client.calls,
        "cache_hits": cache_hits,
        "rpc_provider_host": client.provider_host,
        "rpc_url_stored": False,
    }
    (output / "14_manifest_metadados_tokens.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    connection.close()
    print(f"Metadados concluídos: {len(records)} tokens; pendências: {len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
