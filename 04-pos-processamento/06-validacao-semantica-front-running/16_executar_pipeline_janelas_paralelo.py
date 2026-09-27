# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4,<2"]
# ///

"""Escalona dinamicamente as janelas entre provedores RPC homologados."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

SPLITS = (
    "estresse_pre_dencun_2024", "transicao_dencun_2024",
    "treino_2024_pos_dencun", "validacao_2025_pre_pectra",
    "transicao_pectra_2025", "teste_final_2025", "estresse_fusaka_2025",
)
RECOVERABLE_STAGES = {"rpc", "tokens"}
PROGRESS_RE = re.compile(r"\[(\d+)/(\d+) blocos novos\]")
TOKEN_PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\] (?:COMPLETE|PARTIAL|ERRO)\b")
STAGE_LABELS = {
    "rpc": "coleta RPC",
    "semantica": "validação semântica",
    "dossie": "geração do dossiê",
    "tokens": "metadados de tokens",
    "adjudicacao": "adjudicação assistida",
    "validacao": "validação das adjudicações",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def endpoint_for(settings: dict[str, Any]) -> tuple[str | None, str | None]:
    for field in ("env", "fallback_env"):
        name = settings.get(field)
        if isinstance(name, str) and os.environ.get(name, "").strip():
            return os.environ[name].strip(), name
    return None, None


def tail_text(path: Path, maximum_bytes: int = 65536) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - maximum_bytes))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def progress_from_log(path: Path) -> tuple[int, int] | None:
    matches = PROGRESS_RE.findall(tail_text(path))
    return tuple(map(int, matches[-1])) if matches else None


def stage_detail(manifest_path: Path, log_path: Path, worker_started: float) -> str:
    """Mostra a etapa ativa, sem confundir comandos com progresso."""
    if not manifest_path.is_file():
        return "iniciando trabalhador"
    try:
        document = read_json(manifest_path)
        window = next(iter((document.get("windows") or {}).values()))
        stages = window.get("stages") or {}
    except (OSError, ValueError, StopIteration, AttributeError):
        return "preparando etapa"
    active = next(
        ((name, state) for name, state in reversed(list(stages.items()))
         if state.get("status") == "running"),
        None,
    )
    if active is None:
        return "preparando próxima etapa"
    stage, state = active
    label = STAGE_LABELS.get(stage, stage)
    if stage == "rpc":
        progress = progress_from_log(log_path)
        if progress:
            done, total = progress
            if done >= total:
                return f"{label}: {done}/{total} blocos; finalizando arquivos"
            try:
                elapsed = max(
                    1.0,
                    (datetime.now(timezone.utc) - datetime.fromisoformat(
                        state["started_at_utc"]
                    )).total_seconds(),
                )
            except (KeyError, ValueError, TypeError):
                elapsed = max(1.0, time.monotonic() - worker_started)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            return (
                f"{label}: {done}/{total} ({100 * done / total:.1f}%), "
                f"{rate:.2f} blocos/s, ETA {eta / 60:.1f} min"
            )
    elif stage == "tokens":
        matches = TOKEN_PROGRESS_RE.findall(tail_text(log_path))
        if matches:
            done, total = map(int, matches[-1])
            return f"{label}: {done}/{total} ({100 * done / total:.1f}%)"
    return f"{label}: em andamento"


def requested_blocks(queue: Path) -> set[int]:
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            "SELECT DISTINCT block_number FROM read_parquet(?)", [str(queue)]
        ).fetchall()
    finally:
        connection.close()
    return {int(row[0]) for row in rows}


def remaining_work(queue: Path, cache: Path) -> tuple[int, int]:
    requested = requested_blocks(queue)
    cached = {
        int(path.stem) for path in cache.glob("*.json")
        if path.stem.isdigit() and int(path.stem) in requested
    }
    return len(requested), len(requested - cached)


def failed_stage(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return None
    try:
        document = read_json(manifest_path)
    except (OSError, ValueError):
        return None
    for window in (document.get("windows") or {}).values():
        for stage, state in (window.get("stages") or {}).items():
            if state.get("status") == "failed":
                return stage
    return None


def rpc_speed(manifest_path: Path, output: Path) -> float | None:
    if not manifest_path.is_file() or not (output / "03_manifest_rpc.json").is_file():
        return None
    try:
        worker = read_json(manifest_path)
        rpc = read_json(output / "03_manifest_rpc.json")
        window = next(iter((worker.get("windows") or {}).values()))
        stage = (window.get("stages") or {}).get("rpc") or {}
        start = datetime.fromisoformat(stage["started_at_utc"])
        end = datetime.fromisoformat(stage["completed_at_utc"])
        elapsed = (end - start).total_seconds()
        new_blocks = int(rpc.get("new_blocks", 0))
    except (KeyError, TypeError, ValueError, StopIteration, OSError):
        return None
    return new_blocks / elapsed if elapsed > 0 and new_blocks > 0 else None


def main() -> int:
    here = Path(__file__).resolve().parent
    src_root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--csharp-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--regression-manifest", type=Path, required=True)
    parser.add_argument("--provider-audit", type=Path, required=True)
    parser.add_argument("--providers-config", type=Path, default=src_root / "configuracao" / "rpc_providers.json")
    parser.add_argument("--reviewer", default="Marcelo Corni Alves")
    parser.add_argument("--status-interval", type=float, default=30.0)
    args = parser.parse_args()
    if args.status_interval < 5:
        parser.error("--status-interval deve ser pelo menos 5 segundos.")
    if not shutil.which("uv"):
        parser.error("Executável uv não encontrado.")

    csharp_root = args.csharp_root.resolve()
    results_root = args.results_root.resolve()
    cache_root = args.cache_root.resolve()
    workers_root = results_root / "_workers"
    workers_root.mkdir(parents=True, exist_ok=True)
    provider_audit = read_json(args.provider_audit.resolve())
    approved = set(provider_audit.get("passed_providers") or [])
    providers_config = args.providers_config.resolve()
    config_sha256 = hashlib.sha256(providers_config.read_bytes()).hexdigest()
    if provider_audit.get("providers_config_sha256") != config_sha256:
        parser.error("A configuração RPC mudou; execute novamente auditoria_rpc.")
    config = read_json(providers_config)
    providers = config.get("providers")
    if not isinstance(providers, dict):
        parser.error("Configuração de provedores inválida.")
    minimum = config.get("minimum_approved_providers", 1)
    if len(approved) < minimum:
        parser.error(f"Provedores aprovados insuficientes: {len(approved)}/{minimum}.")

    available: dict[str, dict[str, Any]] = {}
    endpoints: dict[str, tuple[str, str]] = {}
    speeds: dict[str, float] = {}
    for name, settings in providers.items():
        if name not in approved or not isinstance(settings, dict):
            continue
        url, used_env = endpoint_for(settings)
        if url and used_env:
            available[name] = settings
            endpoints[name] = (url, used_env)
            speeds[name] = float(settings.get(
                "initial_effective_blocks_per_second",
                float(settings["blocks_per_batch"]) * float(settings["requests_per_second"]),
            ))
    if not available:
        parser.error("Nenhum provedor aprovado e configurado está disponível.")

    tasks: dict[str, dict[str, Any]] = {}
    skipped_completed: list[str] = []
    print("\nPlanejamento das janelas:", flush=True)
    for split in args.splits:
        if (results_root / split / "19_manifest_validacao_adjudicacoes.json").is_file():
            skipped_completed.append(split)
            print(f"  [concluída] {split}", flush=True)
            continue
        queue = csharp_root / split / "05_enrichment_queue.parquet"
        total, remaining = remaining_work(queue, cache_root / "blocks" / split)
        tasks[split] = {
            "split": split, "total_blocks": total, "remaining_blocks": remaining,
            "attempted_providers": [], "attempts": [], "status": "queued",
        }
        print(f"  [fila] {split}: {remaining}/{total} blocos sem cache", flush=True)
    if not tasks:
        print("Todas as janelas selecionadas já estão concluídas.")
        return 0

    manifest_path = results_root / "00_manifest_pipeline_paralelo.json"
    manifest: dict[str, Any] = {
        "started_at_utc": utc_now(), "updated_at_utc": utc_now(),
        "completed_at_utc": None, "status": "running",
        "scheduler": "dynamic_lpt_with_cross_provider_recovery",
        "credentials_stored": False, "providers_config": str(providers_config),
        "providers_config_sha256": config_sha256,
        "provider_audit": str(args.provider_audit.resolve()),
        "initial_provider_speeds_blocks_per_second": dict(speeds),
        "current_provider_speeds_blocks_per_second": dict(speeds),
        "skipped_completed_splits": skipped_completed, "windows": tasks,
    }
    write_json_atomic(manifest_path, manifest)
    running: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    permanent_failures: set[str] = set()
    last_status = 0.0

    def start_task(provider: str, task: dict[str, Any]) -> None:
        split = task["split"]
        number = len(task["attempts"]) + 1
        provider_dir = workers_root / provider
        provider_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{split}__tentativa-{number:02d}"
        log_path = provider_dir / f"{stem}.log"
        worker_manifest = provider_dir / f"{stem}.json"
        settings = available[provider]
        command = [
            sys.executable, str(here / "11_executar_pipeline_janelas.py"),
            "--splits", split, "--csharp-root", str(csharp_root),
            "--results-root", str(results_root), "--cache-root", str(cache_root),
            "--audit-manifest", str(args.audit_manifest.resolve()),
            "--regression-manifest", str(args.regression_manifest.resolve()),
            "--manifest-path", str(worker_manifest), "--provider-name", provider,
            "--rpc-url-env", "ETH_RPC_WORKER_URL",
            "--blocks-per-batch", str(settings["blocks_per_batch"]),
            "--requests-per-second", str(settings["requests_per_second"]),
            "--token-calls-per-second", str(settings["token_calls_per_second"]),
            "--reviewer", args.reviewer,
        ]
        environment = os.environ.copy()
        environment["ETH_RPC_WORKER_URL"] = endpoints[provider][0]
        environment["PYTHONUNBUFFERED"] = "1"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=src_root, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        attempt = {
            "attempt": number, "provider": provider, "used_env": endpoints[provider][1],
            "started_at_utc": utc_now(), "completed_at_utc": None,
            "status": "running", "return_code": None, "failed_stage": None,
            "log": str(log_path), "worker_manifest": str(worker_manifest),
            "endpoint_stored": False,
        }
        task["attempted_providers"].append(provider)
        task["attempts"].append(attempt)
        task["status"] = "running"
        running[provider] = {
            "process": process, "handle": handle, "task": task, "attempt": attempt,
            "log": log_path, "worker_manifest": worker_manifest,
            "started_monotonic": time.monotonic(),
        }
        print(f"[{provider}] iniciou {split} (tentativa {number}; carga {task['remaining_blocks']}; velocidade {speeds[provider]:.2f} blocos/s)", flush=True)

    def schedule_idle() -> None:
        idle = sorted(
            (name for name in available if name not in running),
            key=lambda name: (-speeds[name], name),
        )
        for provider in idle:
            candidates = [
                task for task in tasks.values()
                if task["status"] == "queued" and provider not in task["attempted_providers"]
            ]
            if candidates:
                candidates.sort(key=lambda task: (-task["remaining_blocks"], task["split"]))
                start_task(provider, candidates[0])

    try:
        while len(completed) + len(permanent_failures) < len(tasks):
            schedule_idle()
            if not running:
                for task in tasks.values():
                    if task["status"] == "queued":
                        task["status"] = "failed_all_providers"
                        permanent_failures.add(task["split"])
                break
            for provider in list(running):
                worker = running[provider]
                code = worker["process"].poll()
                if code is None:
                    continue
                worker["handle"].close()
                task, attempt = worker["task"], worker["attempt"]
                split = task["split"]
                stage = failed_stage(worker["worker_manifest"])
                attempt.update({
                    "completed_at_utc": utc_now(), "status": "completed" if code == 0 else "failed",
                    "return_code": code, "failed_stage": stage,
                })
                del running[provider]
                if code == 0:
                    task["status"] = "completed"
                    completed.add(split)
                    observed = rpc_speed(worker["worker_manifest"], results_root / split)
                    if observed:
                        speeds[provider] = round((speeds[provider] + observed) / 2, 4)
                    print(f"[{provider}] concluiu {split} com sucesso.", flush=True)
                elif stage in RECOVERABLE_STAGES and len(task["attempted_providers"]) < len(available):
                    total, remaining = remaining_work(
                        csharp_root / split / "05_enrichment_queue.parquet", cache_root / "blocks" / split
                    )
                    task.update({"total_blocks": total, "remaining_blocks": remaining, "status": "queued"})
                    speeds[provider] = max(0.05, round(speeds[provider] * 0.75, 4))
                    print(f"[{provider}] falhou em {split} na etapa {stage}; {remaining} blocos pendentes irão para outro provedor.", flush=True)
                else:
                    task["status"] = "failed"
                    permanent_failures.add(split)
                    print(f"[{provider}] falha definitiva em {split} (etapa={stage or 'desconhecida'}, código={code}).", flush=True)
                manifest["current_provider_speeds_blocks_per_second"] = dict(speeds)
                manifest["updated_at_utc"] = utc_now()
                write_json_atomic(manifest_path, manifest)

            now = time.monotonic()
            if now - last_status >= args.status_interval:
                print("\n[status]", flush=True)
                for provider, worker in sorted(running.items()):
                    detail = stage_detail(
                        worker["worker_manifest"], worker["log"],
                        worker["started_monotonic"],
                    )
                    print(f"  [{provider}] {worker['task']['split']}: {detail}", flush=True)
                queued = [task["split"] for task in tasks.values() if task["status"] == "queued"]
                print(f"  fila: {', '.join(queued) if queued else 'vazia'}", flush=True)
                last_status = now
            time.sleep(2)
    except KeyboardInterrupt:
        print("Interrupção recebida; encerrando trabalhadores...", file=sys.stderr)
        for worker in running.values():
            worker["process"].terminate()
        for worker in running.values():
            try:
                worker["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker["process"].kill()
            worker["handle"].close()
            worker["attempt"].update({"status": "interrupted", "completed_at_utc": utc_now(), "return_code": 130})
            worker["task"]["status"] = "queued"
        manifest.update({"status": "interrupted", "updated_at_utc": utc_now(), "completed_at_utc": utc_now()})
        write_json_atomic(manifest_path, manifest)
        return 130

    success = not permanent_failures and len(completed) == len(tasks)
    manifest.update({
        "status": "completed" if success else "failed", "updated_at_utc": utc_now(),
        "completed_at_utc": utc_now(), "completed_windows": sorted(completed),
        "failed_windows": sorted(permanent_failures),
        "current_provider_speeds_blocks_per_second": dict(speeds),
    })
    write_json_atomic(manifest_path, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
