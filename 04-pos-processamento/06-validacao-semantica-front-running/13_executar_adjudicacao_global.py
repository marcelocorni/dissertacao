# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Regera, valida e consolida as adjudicações assistidas das sete janelas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SPLITS = (
    "estresse_pre_dencun_2024",
    "transicao_dencun_2024",
    "treino_2024_pos_dencun",
    "validacao_2025_pre_pectra",
    "transicao_pectra_2025",
    "teste_final_2025",
    "estresse_fusaka_2025",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(f'"{item}"' if " " in item else item for item in command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--reviewer", default="Marcelo Corni Alves")
    parser.add_argument("--results-root", type=Path, default=here / "resultados")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autoriza substituir decisões e validações já existentes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = args.results_root.resolve()
    existing = [
        results / split / "15_decisoes_dossie.csv"
        for split in args.splits
        if (results / split / "15_decisoes_dossie.csv").is_file()
    ]
    if existing and not args.overwrite and not args.dry_run:
        parser.error(
            "Já existem decisões na raiz selecionada. Use outra --results-root "
            "ou informe --overwrite conscientemente."
        )
    execution = {
        "started_at_utc": now(),
        "completed_at_utc": None,
        "status": "running",
        "decision_policy": "final_assisted_without_manual_override",
        "reviewer": args.reviewer,
        "splits": {},
    }

    for split in args.splits:
        output = results / split
        dossier = output / "08_dossie_auditoria.parquet"
        transactions = output / "01_transacoes_rpc.parquet"
        decisions = output / "15_decisoes_dossie.csv"
        if not dossier.is_file() or not transactions.is_file():
            parser.error(f"{split}: dossiê ou transações RPC ausentes")

        execution["splits"][split] = {"status": "running", "started_at_utc": now()}
        adjudicate = [
            sys.executable, str(here / "06_adjudicar_dossie_assistido.py"),
            "--dossier", str(dossier),
            "--transactions", str(transactions),
            "--decisions", str(decisions),
            "--reviewer", args.reviewer,
            "--overwrite",
        ]
        validate = [
            sys.executable, str(here / "09_validar_adjudicacoes_assistidas.py"),
            "--dossier", str(dossier),
            "--transactions", str(transactions),
            "--decisions", str(decisions),
            "--expected-reviewer", args.reviewer,
            "--output-dir", str(output),
        ]
        run(adjudicate, args.dry_run)
        run(validate, args.dry_run)

        if not args.dry_run:
            validation_manifest = json.loads(
                (output / "19_manifest_validacao_adjudicacoes.json").read_text(encoding="utf-8")
            )
            counts = validation_manifest.get("counts", {})
            if counts.get("invalid", 0) or counts.get("warning", 0):
                raise RuntimeError(f"{split}: validação não aprovada: {counts}")
            execution["splits"][split].update(
                status="completed",
                completed_at_utc=now(),
                adjudications=validation_manifest.get("events"),
                validation_counts=counts,
            )
        else:
            execution["splits"][split]["status"] = "dry_run"

    run(
        [
            sys.executable,
            str(here / "12_consolidar_rotulos_semanticos.py"),
            "--results-root",
            str(results),
            "--output-dir",
            str(results / "consolidado"),
        ],
        args.dry_run,
    )
    execution["status"] = "dry_run" if args.dry_run else "completed"
    execution["completed_at_utc"] = now()

    if not args.dry_run:
        manifest = results / "consolidado" / "08_manifest_execucao_adjudicacao_global.json"
        manifest.write_text(json.dumps(execution, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Adjudicação global concluída: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
