# /// script
# requires-python = ">=3.11"
# dependencies = ["requests>=2.32,<3"]
# ///

"""Testa, sem expor credenciais, os provedores RPC configurados."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests


WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
DECIMALS_SELECTOR = "0x313ce567"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def endpoint_for(settings: dict[str, Any]) -> tuple[str | None, str | None]:
    names = [settings.get("env"), settings.get("fallback_env")]
    for name in names:
        if isinstance(name, str) and os.environ.get(name, "").strip():
            return os.environ[name].strip(), name
    return None, None


def test_provider(
    name: str, settings: dict[str, Any], block_number: int, timeout: float
) -> dict[str, Any]:
    url, used_env = endpoint_for(settings)
    base: dict[str, Any] = {
        "provider": name,
        "configured_env": settings.get("env"),
        "used_env": used_env,
        "tested_at_utc": utc_now(),
        "endpoint_stored": False,
        "status": "not_configured" if not url else "testing",
    }
    if not url:
        return base
    base["host"] = urlsplit(url).hostname or "unknown"
    started = time.monotonic()
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "User-Agent": "ethereum-front-running-academic-research/1.0",
        }
    )
    try:
        chain_response = session.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "eth_chainId", "params": []},
            timeout=timeout,
        )
        chain_status = chain_response.status_code
        chain_response.raise_for_status()
        chain_body = chain_response.json()
        if chain_body.get("result") != "0x1":
            raise RuntimeError("chain_id_incorreto")

        tag = hex(block_number)
        batch_response = session.post(
            url,
            json=[
                {
                    "jsonrpc": "2.0",
                    "id": "block",
                    "method": "eth_getBlockByNumber",
                    "params": [tag, True],
                },
                {
                    "jsonrpc": "2.0",
                    "id": "receipts",
                    "method": "eth_getBlockReceipts",
                    "params": [tag],
                },
            ],
            timeout=timeout,
        )
        batch_status = batch_response.status_code
        batch_response.raise_for_status()
        batch_body = batch_response.json()
        if not isinstance(batch_body, list):
            raise RuntimeError("batch_nao_suportado")
        by_id = {str(item.get("id")): item for item in batch_body}
        block = by_id.get("block", {}).get("result")
        receipts = by_id.get("receipts", {}).get("result")
        if not isinstance(block, dict):
            raise RuntimeError("bloco_historico_indisponivel")
        if not isinstance(receipts, list):
            raise RuntimeError("eth_getBlockReceipts_indisponivel")
        transactions = block.get("transactions") or []
        if len(transactions) != len(receipts):
            raise RuntimeError("quantidade_receipts_divergente")

        call_response = session.post(
            url,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "eth_call",
                "params": [{"to": WETH, "data": DECIMALS_SELECTOR}, "latest"],
            },
            timeout=timeout,
        )
        call_status = call_response.status_code
        call_response.raise_for_status()
        call_result = call_response.json().get("result")
        if not isinstance(call_result, str) or int(call_result, 16) != 18:
            raise RuntimeError("eth_call_invalido")

        base.update(
            {
                "status": "passed",
                "chain_id": "0x1",
                "historical_block": block_number,
                "block_hash": block.get("hash"),
                "transactions": len(transactions),
                "receipts": len(receipts),
                "http_status": {
                    "chain_id": chain_status,
                    "batch": batch_status,
                    "eth_call": call_status,
                },
            }
        )
    except requests.RequestException as exc:
        base.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": "falha_http_ou_timeout",
            }
        )
    except (ValueError, RuntimeError) as exc:
        base.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        session.close()
    base["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return base


def main() -> int:
    here = Path(__file__).resolve().parent
    src_root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers-config",
        type=Path,
        default=src_root / "configuracao" / "rpc_providers.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-block", type=int, default=19_000_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.historical_block < 1 or args.timeout <= 0:
        parser.error("Bloco e timeout devem ser positivos.")

    providers_config = args.providers_config.resolve()
    document = read_json(providers_config)
    providers = document.get("providers")
    if not isinstance(providers, dict) or not providers:
        parser.error("Nenhum provedor definido em providers.")
    minimum_approved = document.get("minimum_approved_providers", 1)
    if (
        not isinstance(minimum_approved, int)
        or isinstance(minimum_approved, bool)
        or minimum_approved < 1
        or minimum_approved > len(providers)
    ):
        parser.error(
            "minimum_approved_providers deve ser inteiro entre 1 e o total de provedores."
        )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {
            executor.submit(
                test_provider, name, settings, args.historical_block, args.timeout
            ): name
            for name, settings in providers.items()
            if isinstance(settings, dict)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{result['provider']}] {result['status']}")
    results.sort(key=lambda item: item["provider"])
    passed = [item["provider"] for item in results if item["status"] == "passed"]
    manifest = {
        "created_at_utc": utc_now(),
        "providers_config": str(providers_config),
        "providers_config_sha256": hashlib.sha256(
            providers_config.read_bytes()
        ).hexdigest(),
        "credentials_stored": False,
        "historical_block": args.historical_block,
        "minimum_approved_providers": minimum_approved,
        "passed_providers": passed,
        "configured_providers": sum(
            item["status"] != "not_configured" for item in results
        ),
        "results": results,
    }
    write_json_atomic(args.output.resolve(), manifest)
    print(f"Provedores aprovados: {len(passed)}/{len(results)}")
    if len(passed) < minimum_approved:
        print(
            f"Homologacao insuficiente: minimo exigido = {minimum_approved}.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
