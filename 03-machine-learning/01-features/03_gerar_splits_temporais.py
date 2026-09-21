# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
# ]
# ///

"""Orquestra a geração dos recortes temporais do experimento.

Cada recorte é produzido pelo artefato 02 com a mesma engenharia de features e
amostragem determinística por blocos completos. O programa valida previamente a
cobertura diária dos Parquets e reutiliza apenas saídas comprovadamente completas.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configuracao import (
    ConfiguracaoPipelineError,
    carregar_amostragem_blocos,
    carregar_root_template,
)


EXPECTED_MATRIX_C_FEATURES = 23


@dataclass(frozen=True)
class TemporalSplit:
    order: int
    key: str
    role: str
    start_date: date
    end_date: date
    protocol_context: str

    @property
    def directory_name(self) -> str:
        return f"{self.order:02d}-{self.key}"

    def serializable(self) -> dict[str, object]:
        result = asdict(self)
        result["start_date"] = self.start_date.isoformat()
        result["end_date"] = self.end_date.isoformat()
        result["directory_name"] = self.directory_name
        return result


SPLITS = [
    TemporalSplit(
        1,
        "estresse_pre_dencun_2024",
        "historical_stress",
        date(2024, 1, 1),
        date(2024, 3, 12),
        "Período anterior à ativação Dencun.",
    ),
    TemporalSplit(
        2,
        "transicao_dencun_2024",
        "protocol_transition",
        date(2024, 3, 13),
        date(2024, 3, 31),
        "Janela de transição iniciada na ativação Dencun em 13/03/2024.",
    ),
    TemporalSplit(
        3,
        "treino_2024_pos_dencun",
        "train",
        date(2024, 4, 1),
        date(2024, 12, 31),
        "Período estável pós-Dencun usado exclusivamente para ajuste.",
    ),
    TemporalSplit(
        4,
        "validacao_2025_pre_pectra",
        "validation",
        date(2025, 1, 1),
        date(2025, 4, 30),
        "Validação temporal posterior ao treino e anterior à Pectra.",
    ),
    TemporalSplit(
        5,
        "transicao_pectra_2025",
        "protocol_transition",
        date(2025, 5, 1),
        date(2025, 5, 31),
        "Janela que contém a ativação Pectra em 07/05/2025.",
    ),
    TemporalSplit(
        6,
        "teste_final_2025",
        "test",
        date(2025, 6, 1),
        date(2025, 11, 30),
        "Teste final intocado posterior à transição Pectra.",
    ),
    TemporalSplit(
        7,
        "estresse_fusaka_2025",
        "future_stress",
        date(2025, 12, 1),
        date(2025, 12, 31),
        "Janela de estresse que contém a ativação Fusaka em 03/12/2025.",
    ),
]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = SRC_ROOT
    parser = argparse.ArgumentParser(
        description="Gera e documenta os recortes temporais de 2024 e 2025."
    )
    try:
        config_path, configured_root = carregar_root_template(
            src_root / "configuracao" / "pipeline.json"
        )
        configured_percent, configured_modulus = carregar_amostragem_blocos(
            config_path
        )
    except ConfiguracaoPipelineError as exc:
        parser.error(str(exc))
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help="Configuração central. Padrão: .\\configuracao\\pipeline.json",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=here / "02_gerar_matrizes_features.py",
        help="Caminho do artefato 02.",
    )
    parser.add_argument(
        "--root-template",
        default=configured_root,
        help="Substitui dados.root_template somente nesta execução.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=here / "resultados-splits",
        help="Diretório raiz dos recortes.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=here / "resultados-matrizes",
        help="Diretório já usado para o conjunto de treinamento.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=src_root / ".tmp" / "duckdb",
        help="Diretório temporário repassado ao artefato 02. Padrão: src/.tmp/duckdb.",
    )
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--threads", type=int, default=max(1, min(8, os.cpu_count() or 4))
    )
    parser.add_argument(
        "--block-modulus",
        type=int,
        default=configured_modulus,
        help=(
            "Substitui a amostragem configurada em todos os recortes; padrão "
            f"atual: {configured_percent:g}%% (módulo {configured_modulus})."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=[split.key for split in SPLITS],
        help="Opcional: gera somente os recortes informados.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera inclusive recortes completos e compatíveis.",
    )
    args = parser.parse_args()
    if args.block_modulus < 1:
        parser.error("--block-modulus deve ser maior ou igual a 1")
    args.block_sampling_percent = 100.0 / args.block_modulus
    if args.threads < 1:
        parser.error("--threads deve ser maior ou igual a 1")
    if not args.generator.is_file():
        parser.error(f"Artefato 02 não encontrado: {args.generator}")
    return args


def iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def validate_daily_coverage(split: TemporalSplit, root_template: str) -> dict[str, int]:
    missing: list[str] = []
    total_files = 0
    for day in iter_days(split.start_date, split.end_date):
        day_dir = Path(root_template.format(year=day.year)) / f"date={day.isoformat()}"
        files = list(day_dir.glob("*.parquet")) if day_dir.is_dir() else []
        if not files:
            missing.append(str(day_dir))
        else:
            total_files += len(files)
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n  ... e mais {len(missing) - 10}"
        raise FileNotFoundError(
            f"Cobertura incompleta em {split.key}: {len(missing)} dias ausentes:\n"
            f"{preview}{suffix}"
        )
    return {
        "expected_days": (split.end_date - split.start_date).days + 1,
        "source_parquet_files": total_files,
    }


def output_dir_for(split: TemporalSplit, args: argparse.Namespace) -> Path:
    if split.role == "train":
        return args.training_dir.resolve()
    return (args.output_root / split.directory_name).resolve()


def completion_status(
    split: TemporalSplit,
    output_dir: Path,
    block_modulus: int,
) -> tuple[bool, str]:
    manifest_path = output_dir / "10_manifest_execucao.json"
    required = [
        output_dir / "05_metadados.parquet",
        output_dir / "06_matriz_A_baseline.parquet",
        output_dir / "07_matriz_B_contexto.parquet",
        output_dir / "08_matriz_C_estendida.parquet",
    ]
    if not manifest_path.is_file() or not all(path.is_file() for path in required):
        return False, "artefatos ausentes"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "manifesto ilegível"
    checks = [
        manifest.get("dataset_name") == split.key,
        manifest.get("start_date") == split.start_date.isoformat(),
        manifest.get("end_date") == split.end_date.isoformat(),
        manifest.get("block_modulus") == block_modulus,
        len(manifest.get("matrix_C", [])) == EXPECTED_MATRIX_C_FEATURES,
        "is_contract_creation" not in manifest.get("matrix_C", []),
        manifest.get("sample_duplicate_hashes") == 0,
    ]
    if not all(checks):
        return False, "manifesto incompatível com o desenho atual"
    if any(path.stat().st_size == 0 for path in required):
        return False, "arquivo de saída vazio"
    return True, "completo e compatível"


def run_generator(
    split: TemporalSplit,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        str(args.generator.resolve()),
        "--config",
        str(args.config.resolve()),
        "--root-template",
        args.root_template,
        "--start-date",
        split.start_date.isoformat(),
        "--end-date",
        split.end_date.isoformat(),
        "--dataset-name",
        split.key,
        "--output-dir",
        str(output_dir),
        "--temp-dir",
        str(args.temp_dir.resolve()),
        "--memory-limit",
        args.memory_limit,
        "--threads",
        str(args.threads),
        "--block-modulus",
        str(args.block_modulus),
    ]
    subprocess.run(command, check=True)


def read_result(output_dir: Path) -> dict[str, object]:
    summary = json.loads(
        (output_dir / "01_resumo_geracao.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "10_manifest_execucao.json").read_text(encoding="utf-8")
    )
    return {
        "output_directory": str(output_dir),
        "sample_rows": summary["sample_rows"],
        "sample_blocks": summary["sample_blocks"],
        "type_4_rows": summary["type_4_rows"],
        "sample_duplicate_hashes": manifest["sample_duplicate_hashes"],
        "matrix_A_features": len(manifest["matrix_A"]),
        "matrix_B_features": len(manifest["matrix_B"]),
        "matrix_C_features": len(manifest["matrix_C"]),
    }


def validate_non_overlap(splits: list[TemporalSplit]) -> None:
    ordered = sorted(splits, key=lambda item: item.start_date)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.end_date >= current.start_date:
            raise RuntimeError(
                f"Sobreposição temporal: {previous.key} e {current.key}"
            )


def write_global_manifest(
    args: argparse.Namespace,
    selected: list[TemporalSplit],
    coverage: dict[str, dict[str, int]],
    results: dict[str, dict[str, object]],
) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "00_manifest_splits_temporais.json"
    document = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "configuration_file": str(args.config.resolve()),
        "root_template": args.root_template,
        "design": "chronological_non_overlapping",
        "block_modulus": args.block_modulus,
        "block_sampling_percent": args.block_sampling_percent,
        "sampling_unit": "complete_block",
        "scaler_policy": "fit_train_only",
        "threshold_policy": "calibrate_validation_only",
        "test_policy": "do_not_use_for_model_or_threshold_selection",
        "row_id_scope": "local_to_each_split",
        "splits": [
            {
                **split.serializable(),
                **coverage[split.key],
                **results[split.key],
            }
            for split in selected
        ],
    }
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def main() -> int:
    args = parse_args()
    selected = [split for split in SPLITS if not args.only or split.key in args.only]
    validate_non_overlap(selected)
    args.output_root.mkdir(parents=True, exist_ok=True)

    coverage: dict[str, dict[str, int]] = {}
    results: dict[str, dict[str, object]] = {}

    print("Validando cobertura diária dos Parquets...")
    for split in selected:
        coverage[split.key] = validate_daily_coverage(split, args.root_template)
        print(
            f"  OK {split.key}: {coverage[split.key]['expected_days']} dias, "
            f"{coverage[split.key]['source_parquet_files']} arquivos"
        )

    for split in selected:
        output_dir = output_dir_for(split, args)
        complete, reason = completion_status(split, output_dir, args.block_modulus)
        if complete and not args.force:
            print(f"REUTILIZADO {split.key}: {reason}")
        else:
            print(f"GERANDO {split.key}: {reason}")
            output_dir.mkdir(parents=True, exist_ok=True)
            run_generator(split, output_dir, args)
            complete, reason = completion_status(
                split, output_dir, args.block_modulus
            )
            if not complete:
                raise RuntimeError(
                    f"O recorte {split.key} terminou, mas falhou na validação: {reason}"
                )
        results[split.key] = read_result(output_dir)

    manifest = write_global_manifest(args, selected, coverage, results)
    print("\nRecortes temporais concluídos e validados.")
    print(f"Manifesto global: {manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
