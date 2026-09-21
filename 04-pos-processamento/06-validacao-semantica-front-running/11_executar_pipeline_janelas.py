"""Executa, de forma retomável, o pipeline semântico nas sete janelas.

O executor exige que a auditoria das saidas C# e a regressao do coletor RPC por
bloco tenham sido aprovadas. A chave RPC e lida somente de ETH_RPC_URL e nunca
e gravada no manifesto global.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = (
    "estresse_pre_dencun_2024",
    "transicao_dencun_2024",
    "treino_2024_pos_dencun",
    "validacao_2025_pre_pectra",
    "transicao_pectra_2025",
    "teste_final_2025",
    "estresse_fusaka_2025",
)

STAGES = ("rpc", "semantica", "dossie", "tokens", "adjudicacao", "validacao")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def command_for_stage(
    uv: str,
    here: Path,
    csharp_root: Path,
    split: str,
    blocks_per_batch: int,
    requests_per_second: float,
    token_calls_per_second: float,
    results_root: Path,
    cache_root: Path,
    reviewer: str,
) -> list[str]:
    source = csharp_root / split
    output = results_root / split
    queue = source / "05_enrichment_queue.parquet"
    commands = {
        "rpc": [
            uv, "run", str(here / "08_enriquecer_fila_por_bloco.py"),
            "--queue", str(queue), "--dataset-name", split,
            "--output-dir", str(output),
            "--cache-root", str(cache_root / "blocks"),
            "--blocks-per-batch", str(blocks_per_batch),
            "--requests-per-second", str(requests_per_second),
        ],
        "semantica": [
            uv, "run", str(here / "02_validar_semantica.py"),
            "--events", str(source / "01_detection_events.parquet"),
            "--transactions", str(output / "01_transacoes_rpc.parquet"),
            "--output-dir", str(output),
        ],
        "dossie": [
            uv, "run", str(here / "03_gerar_dossie_auditoria.py"),
            "--validated-events", str(output / "04_eventos_validados.parquet"),
            "--transactions", str(output / "01_transacoes_rpc.parquet"),
            "--output-dir", str(output),
        ],
        "tokens": [
            uv, "run", str(here / "04_enriquecer_tokens_rpc.py"),
            "--flows", str(output / "09_fluxos_tokens_atacante.parquet"),
            "--output-dir", str(output),
            "--cache-dir", str(cache_root / "token_metadata" / split),
            "--calls-per-second", str(token_calls_per_second),
        ],
        "adjudicacao": [
            uv, "run", str(here / "06_adjudicar_dossie_assistido.py"),
            "--dossier", str(output / "08_dossie_auditoria.parquet"),
            "--transactions", str(output / "01_transacoes_rpc.parquet"),
            "--decisions", str(output / "15_decisoes_dossie.csv"),
            "--reviewer", reviewer,
        ],
        "validacao": [
            uv, "run", str(here / "09_validar_adjudicacoes_assistidas.py"),
            "--dossier", str(output / "08_dossie_auditoria.parquet"),
            "--transactions", str(output / "01_transacoes_rpc.parquet"),
            "--decisions", str(output / "15_decisoes_dossie.csv"),
            "--expected-reviewer", reviewer,
            "--output-dir", str(output),
        ],
    }
    return commands


def rpc_is_complete(output: Path, queue: Path, split: str) -> bool:
    manifest_path = output / "03_manifest_rpc.json"
    transactions = output / "01_transacoes_rpc.parquet"
    if not manifest_path.is_file() or not transactions.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return False
    return (
        manifest.get("dataset") == split
        and Path(manifest.get("source_queue", "")).resolve() == queue.resolve()
        and int(manifest.get("requested_transactions", -1))
        == int(manifest.get("enriched_transactions", -2))
        and int(manifest.get("errors", -1)) == 0
    )


def validate_preconditions(
    csharp_root: Path,
    audit_path: Path,
    regression_path: Path,
    dry_run: bool,
) -> None:
    if not audit_path.is_file():
        raise RuntimeError(f"Auditoria C# ausente: {audit_path}")
    audit = read_json(audit_path)
    if audit.get("rerun_csharp_required") or audit.get("found_splits") != 7:
        raise RuntimeError("A auditoria C# nao aprovou integralmente as sete janelas.")
    if not regression_path.is_file() or not read_json(regression_path).get("passed"):
        raise RuntimeError("A regressao do coletor RPC por bloco nao esta aprovada.")
    for split in SPLITS:
        source = csharp_root / split
        for filename in ("01_detection_events.parquet", "05_enrichment_queue.parquet"):
            if not (source / filename).is_file():
                raise RuntimeError(f"Entrada ausente: {source / filename}")
    if not dry_run and not os.environ.get("ETH_RPC_URL", "").strip():
        raise RuntimeError("Defina ETH_RPC_URL na sessao antes da execucao.")


def main() -> int:
    here = Path(__file__).resolve().parent
    default_csharp = (
        here.parent / "05-rotulador-front-running-csharp" / "resultados-v3"
        / "sampled" / "independent_sampled"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--from-stage", choices=STAGES, default="rpc")
    parser.add_argument("--csharp-root", type=Path, default=default_csharp)
    parser.add_argument("--results-root", type=Path, default=here / "resultados")
    parser.add_argument("--cache-root", type=Path, default=here / "cache")
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=here / "resultados" / "auditoria-csharp" / "02_manifest_auditoria_csharp.json",
    )
    parser.add_argument(
        "--regression-manifest",
        type=Path,
        default=here / "resultados-blocos" / "transicao_dencun_2024" / "04_regressao_rpc.json",
    )
    parser.add_argument("--reviewer", default="Marcelo Corni Alves")
    parser.add_argument("--blocks-per-batch", type=int, default=5)
    parser.add_argument("--requests-per-second", type=float, default=1.0)
    parser.add_argument("--token-calls-per-second", type=float, default=2.0)
    parser.add_argument("--force-rpc", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--allow-existing-results",
        action="store_true",
        help="Autoriza substituir artefatos derivados de janelas já concluídas.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if min(args.blocks_per_batch, args.requests_per_second, args.token_calls_per_second) <= 0:
        parser.error("Limites de lote e frequencia devem ser positivos.")
    uv = shutil.which("uv")
    if not uv:
        parser.error("Executavel uv nao encontrado no PATH.")
    csharp_root = args.csharp_root.resolve()
    results_root = args.results_root.resolve()
    cache_root = args.cache_root.resolve()
    try:
        validate_preconditions(
            csharp_root,
            args.audit_manifest.resolve(),
            args.regression_manifest.resolve(),
            args.dry_run,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    selected_stages = STAGES[STAGES.index(args.from_stage):]
    completed_outputs = [
        results_root / split / "19_manifest_validacao_adjudicacoes.json"
        for split in args.splits
        if (results_root / split / "19_manifest_validacao_adjudicacoes.json").is_file()
    ]
    if completed_outputs and not args.allow_existing_results and not args.dry_run:
        parser.error(
            "Há janelas já concluídas na raiz de resultados. Use outra --results-root "
            "ou informe --allow-existing-results conscientemente."
        )
    manifest_path = results_root / "00_manifest_pipeline_global.json"
    manifest: dict[str, Any] = {
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "completed_at_utc": None,
        "reviewer": "Marcelo Corni Alves",
        "rpc_url_stored": False,
        "regression_approved": True,
        "csharp_audit_approved": True,
        "selected_splits": args.splits,
        "selected_stages": list(selected_stages),
        "status": "dry_run" if args.dry_run else "running",
        "windows": {},
    }
    if not args.dry_run:
        write_json_atomic(manifest_path, manifest)

    failures = 0
    for split in args.splits:
        source = csharp_root / split
        output = results_root / split
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
        window = manifest["windows"].setdefault(split, {"status": "running", "stages": {}})
        commands = command_for_stage(
            uv, here, csharp_root, split, args.blocks_per_batch,
            args.requests_per_second, args.token_calls_per_second,
            results_root, cache_root, args.reviewer,
        )
        for stage in selected_stages:
            if stage == "rpc" and not args.force_rpc and rpc_is_complete(
                output, source / "05_enrichment_queue.parquet", split
            ):
                print(f"[{split}] RPC ja completo; etapa ignorada.", flush=True)
                window["stages"][stage] = {"status": "skipped_complete", "at_utc": utc_now()}
                continue
            command = commands[stage]
            printable = subprocess.list2cmdline(command)
            print(f"\n[{split}] ETAPA {stage}\n{printable}", flush=True)
            if args.dry_run:
                window["stages"][stage] = {"status": "planned"}
                continue
            state = {"status": "running", "started_at_utc": utc_now()}
            window["stages"][stage] = state
            manifest["updated_at_utc"] = utc_now()
            write_json_atomic(manifest_path, manifest)
            try:
                subprocess.run(command, check=True, cwd=here)
            except subprocess.CalledProcessError as error:
                failures += 1
                state.update({"status": "failed", "return_code": error.returncode, "completed_at_utc": utc_now()})
                window["status"] = "failed"
                manifest["status"] = "failed"
                manifest["updated_at_utc"] = utc_now()
                write_json_atomic(manifest_path, manifest)
                if not args.continue_on_error:
                    return error.returncode or 1
                break
            state.update({"status": "completed", "completed_at_utc": utc_now()})
            manifest["updated_at_utc"] = utc_now()
            write_json_atomic(manifest_path, manifest)
        else:
            window["status"] = "planned" if args.dry_run else "completed"
            window["completed_at_utc"] = utc_now()
            if not args.dry_run:
                manifest["updated_at_utc"] = utc_now()
                write_json_atomic(manifest_path, manifest)

    if args.dry_run:
        print("\nSimulacao concluida; nenhum arquivo foi alterado.")
        return 0
    manifest["status"] = "completed" if failures == 0 else "completed_with_failures"
    manifest["completed_at_utc"] = utc_now()
    manifest["updated_at_utc"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    print(f"\nPipeline finalizado: {len(args.splits) - failures}/{len(args.splits)} janelas sem falha.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
