# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
# ]
# ///

"""Gera as matrizes A, B e C para os modelos AE e Isolation Forest.

Fonte exclusiva: arquivos Parquet de transações Ethereum. Identificadores são
gravados separadamente das matrizes numéricas. O programa trata zeros e strings
vazias usados pelo dataset como sentinelas de campos não aplicáveis/ausentes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import duckdb

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configuracao import (
    ConfiguracaoPipelineError,
    carregar_amostragem_blocos,
    carregar_root_template,
)


FEATURE_SET_A = [
    "gas_used_log",
    "gas_used_ratio",
    "effective_gas_price_log",
    "max_fee_per_gas_log",
    "max_priority_fee_log",
    "fee_cap_utilization",
    "transaction_value_log",
    "nonce_log",
    "receipt_status",
    "tx_type_0",
    "tx_type_1",
    "tx_type_3",
    "transaction_index_pct",
]

FEATURE_SET_B = FEATURE_SET_A + [
    "block_tx_count_log",
    "sender_tx_count_block_log",
    "receiver_tx_count_block_log",
    "gas_price_rank_block",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

FEATURE_SET_C = FEATURE_SET_B + [
    "gas_limit_log",
    "transaction_fee_paid_log",
]

FEATURE_DEFINITIONS = {
    "gas_used_log": (
        "execucao",
        "log1p do gás efetivamente consumido pela transação.",
    ),
    "gas_used_ratio": (
        "execucao",
        "Razão entre o gás consumido e o limite de gás informado.",
    ),
    "effective_gas_price_log": (
        "taxa",
        "log1p do preço efetivo do gás; usa gas_price como fallback.",
    ),
    "max_fee_per_gas_log": (
        "taxa",
        "log1p do teto de taxa; zero quando não aplicável.",
    ),
    "max_priority_fee_log": (
        "taxa",
        "log1p da prioridade máxima; zero quando não aplicável.",
    ),
    "fee_cap_utilization": (
        "taxa",
        "Preço efetivo dividido pelo teto de taxa; zero quando não aplicável.",
    ),
    "transaction_value_log": (
        "valor",
        "log1p do valor nativo transferido.",
    ),
    "nonce_log": (
        "conta",
        "log1p do nonce da conta remetente.",
    ),
    "receipt_status": (
        "execucao",
        "Indicador de sucesso (1) ou falha (0) da execução.",
    ),
    "tx_type_0": (
        "tipo",
        "Indicador de transação Ethereum tipo 0.",
    ),
    "tx_type_1": (
        "tipo",
        "Indicador de transação Ethereum tipo 1.",
    ),
    "tx_type_3": (
        "tipo",
        "Indicador de transação blob tipo 3; tipo 2 é a referência.",
    ),
    "transaction_index_pct": (
        "bloco",
        "Posição relativa da transação dentro do bloco.",
    ),
    "block_tx_count_log": (
        "bloco",
        "log1p da quantidade de transações no bloco.",
    ),
    "sender_tx_count_block_log": (
        "bloco",
        "log1p da frequência do remetente no mesmo bloco.",
    ),
    "receiver_tx_count_block_log": (
        "bloco",
        "log1p da frequência do destinatário no mesmo bloco.",
    ),
    "gas_price_rank_block": (
        "bloco",
        "Percentil do preço efetivo de gás dentro do bloco.",
    ),
    "hour_sin": ("tempo", "Componente seno da hora UTC."),
    "hour_cos": ("tempo", "Componente cosseno da hora UTC."),
    "weekday_sin": ("tempo", "Componente seno do dia da semana UTC."),
    "weekday_cos": ("tempo", "Componente cosseno do dia da semana UTC."),
    "gas_limit_log": (
        "execucao",
        "log1p do limite de gás; feature de ablação do conjunto C.",
    ),
    "transaction_fee_paid_log": (
        "taxa",
        "log1p de receipt_gas_used vezes preço efetivo do gás.",
    ),
}

DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = SRC_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Cria três matrizes numéricas para ablação e um arquivo separado "
            "de metadados/rastreabilidade."
        )
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
        "--root-template",
        default=configured_root,
        help="Substitui dados.root_template somente nesta execução.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date(2024, 4, 1),
        help="Início do período, inclusive.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2024, 12, 31),
        help="Fim do período, inclusive.",
    )
    parser.add_argument(
        "--dataset-name",
        default="treino_2024_pos_dencun",
        help="Nome lógico registrado nos manifestos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "resultados-matrizes",
        help="Diretório de saída deste artefato.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=src_root / ".tmp" / "duckdb",
        help="Diretório temporário do DuckDB. Padrão: src/.tmp/duckdb.",
    )
    parser.add_argument(
        "--block-modulus",
        type=int,
        default=configured_modulus,
        help=(
            "Substitui a amostragem configurada nesta execução. Mantém "
            "block_number %% valor = 0; padrão atual: "
            f"{configured_percent:g}%% (módulo {configured_modulus})."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        default="16GB",
        help="Limite de memória do DuckDB, por exemplo 8GB ou 16GB.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 4)),
        help="Número de threads do DuckDB.",
    )
    parser.add_argument(
        "--compression",
        choices=("ZSTD", "SNAPPY"),
        default="ZSTD",
        help="Compressão dos Parquets de saída.",
    )
    args = parser.parse_args()
    if args.start_date > args.end_date:
        parser.error("--start-date não pode ser posterior a --end-date")
    if args.block_modulus < 1:
        parser.error("--block-modulus deve ser maior ou igual a 1")
    args.block_sampling_percent = 100.0 / args.block_modulus
    if args.threads < 1:
        parser.error("--threads deve ser maior ou igual a 1")
    return args


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_file_list(files: Iterable[Path]) -> str:
    return "[" + ", ".join(sql_quote(p.as_posix()) for p in files) + "]"


def discover_files(root_template: str, start: date, end: date) -> list[Path]:
    files: list[Path] = []
    for year in range(start.year, end.year + 1):
        root = Path(root_template.format(year=year))
        if not root.exists():
            print(f"AVISO: diretório ausente: {root}", file=sys.stderr)
            continue
        for day_dir in sorted(root.glob("date=*")):
            match = DATE_DIR_RE.match(day_dir.name)
            if not match:
                continue
            current = date.fromisoformat(match.group(1))
            if start <= current <= end:
                files.extend(sorted(day_dir.glob("*.parquet")))
    if not files:
        raise FileNotFoundError(
            "Nenhum Parquet encontrado. Confira template e intervalo de datas."
        )
    return files


def configure(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir.as_posix())}")
    con.execute("SET preserve_insertion_order = false")
    return con


def create_raw_view(con: duckdb.DuckDBPyConnection, files: list[Path]) -> None:
    con.execute(
        f"""
        CREATE VIEW raw_transactions AS
        SELECT *
        FROM read_parquet(
            {sql_file_list(files)},
            hive_partitioning = false,
            union_by_name = true
        )
        """
    )


def validate_schema(con: duckdb.DuckDBPyConnection) -> None:
    schema = con.execute("DESCRIBE SELECT * FROM raw_transactions").fetchall()
    available = {row[0] for row in schema}
    required = {
        "hash",
        "block_hash",
        "block_number",
        "block_timestamp",
        "date",
        "from_address",
        "to_address",
        "transaction_index",
        "gas",
        "value",
        "gas_price",
        "receipt_gas_used",
        "receipt_status",
        "transaction_type",
        "max_priority_fee_per_gas",
        "nonce",
        "receipt_effective_gas_price",
        "max_fee_per_gas",
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError("Colunas obrigatórias ausentes: " + ", ".join(missing))


def create_feature_master(
    con: duckdb.DuckDBPyConnection, block_modulus: int
) -> None:
    con.execute(
        f"""
        CREATE TEMP TABLE feature_master AS
        WITH sampled_blocks AS (
            SELECT
                hash,
                block_hash,
                block_number,
                block_timestamp,
                date,
                from_address,
                to_address,
                transaction_index,
                transaction_type,
                gas,
                value,
                receipt_gas_used,
                receipt_status,
                max_priority_fee_per_gas,
                nonce,
                COALESCE(receipt_effective_gas_price, gas_price, 0)
                    AS effective_price,
                max_fee_per_gas,
                COUNT(*) OVER (PARTITION BY block_number) AS block_tx_count,
                COUNT(*) OVER (
                    PARTITION BY block_number, from_address
                ) AS sender_tx_count_block,
                COUNT(*) OVER (
                    PARTITION BY block_number, to_address
                ) AS receiver_tx_count_block,
                PERCENT_RANK() OVER (
                    PARTITION BY block_number
                    ORDER BY COALESCE(receipt_effective_gas_price, gas_price, 0)
                ) AS gas_price_rank_block
            FROM raw_transactions
            WHERE block_number % {block_modulus} = 0
        ),
        engineered AS (
            SELECT
                hash,
                block_hash,
                block_number,
                block_timestamp,
                date,
                from_address,
                to_address,
                transaction_index,
                transaction_type,
                LN(1.0 + GREATEST(COALESCE(receipt_gas_used, 0), 0))
                    AS gas_used_log,
                CASE
                    WHEN gas > 0
                    THEN CAST(receipt_gas_used AS DOUBLE) / CAST(gas AS DOUBLE)
                    ELSE 0.0
                END AS gas_used_ratio,
                LN(1.0 + GREATEST(CAST(effective_price AS DOUBLE), 0.0))
                    AS effective_gas_price_log,
                LN(
                    1.0 + GREATEST(CAST(COALESCE(max_fee_per_gas, 0) AS DOUBLE), 0.0)
                ) AS max_fee_per_gas_log,
                LN(
                    1.0
                    + GREATEST(
                        CAST(COALESCE(max_priority_fee_per_gas, 0) AS DOUBLE),
                        0.0
                    )
                ) AS max_priority_fee_log,
                CASE
                    WHEN max_fee_per_gas > 0
                    THEN CAST(effective_price AS DOUBLE)
                         / CAST(max_fee_per_gas AS DOUBLE)
                    ELSE 0.0
                END AS fee_cap_utilization,
                LN(1.0 + GREATEST(CAST(COALESCE(value, 0) AS DOUBLE), 0.0))
                    AS transaction_value_log,
                LN(1.0 + GREATEST(CAST(COALESCE(nonce, 0) AS DOUBLE), 0.0))
                    AS nonce_log,
                CAST(COALESCE(receipt_status, 0) AS UTINYINT) AS receipt_status,
                CAST(transaction_type = 0 AS UTINYINT) AS tx_type_0,
                CAST(transaction_type = 1 AS UTINYINT) AS tx_type_1,
                CAST(transaction_type = 3 AS UTINYINT) AS tx_type_3,
                CASE
                    WHEN block_tx_count > 1
                    THEN CAST(transaction_index AS DOUBLE) / (block_tx_count - 1)
                    ELSE 0.0
                END AS transaction_index_pct,
                LN(1.0 + block_tx_count) AS block_tx_count_log,
                LN(1.0 + sender_tx_count_block) AS sender_tx_count_block_log,
                LN(1.0 + receiver_tx_count_block) AS receiver_tx_count_block_log,
                gas_price_rank_block,
                SIN(2.0 * PI() * EXTRACT(hour FROM block_timestamp) / 24.0)
                    AS hour_sin,
                COS(2.0 * PI() * EXTRACT(hour FROM block_timestamp) / 24.0)
                    AS hour_cos,
                SIN(2.0 * PI() * EXTRACT(dow FROM block_timestamp) / 7.0)
                    AS weekday_sin,
                COS(2.0 * PI() * EXTRACT(dow FROM block_timestamp) / 7.0)
                    AS weekday_cos,
                LN(1.0 + GREATEST(CAST(COALESCE(gas, 0) AS DOUBLE), 0.0))
                    AS gas_limit_log,
                LN(
                    1.0
                    + GREATEST(CAST(COALESCE(receipt_gas_used, 0) AS DOUBLE), 0.0)
                      * GREATEST(CAST(effective_price AS DOUBLE), 0.0)
                ) AS transaction_fee_paid_log,
                CAST(
                    to_address IS NULL
                    OR TRIM(COALESCE(to_address, '')) IN ('', '0x')
                    AS UTINYINT
                ) AS is_contract_creation
            FROM sampled_blocks
        )
        SELECT
            ROW_NUMBER() OVER ()::UBIGINT AS row_id,
            *
        FROM engineered
        """
    )


def feature_validation(con: duckdb.DuckDBPyConnection) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_SET_C:
        result = con.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN {feature} IS NULL THEN 1 ELSE 0 END) AS nulls,
                SUM(
                    CASE
                        WHEN {feature} IS NOT NULL
                             AND NOT ISFINITE(CAST({feature} AS DOUBLE))
                        THEN 1 ELSE 0
                    END
                ) AS non_finite,
                COUNT(DISTINCT {feature}) AS distinct_count,
                MIN({feature}) AS minimum,
                MAX({feature}) AS maximum,
                AVG({feature}) AS mean,
                STDDEV_SAMP({feature}) AS stddev
            FROM feature_master
            """
        ).fetchone()
        rows.append(
            {
                "feature": feature,
                "total_rows": int(result[0]),
                "null_count": int(result[1]),
                "non_finite_count": int(result[2]),
                "distinct_count": int(result[3]),
                "minimum": result[4],
                "maximum": result[5],
                "mean": result[6],
                "stddev": result[7],
                "is_constant": int(result[3]) <= 1,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_feature_dictionary(path: Path) -> None:
    rows = []
    for feature in FEATURE_SET_C:
        group, definition = FEATURE_DEFINITIONS[feature]
        rows.append(
            {
                "feature": feature,
                "group": group,
                "definition": definition,
                "matrix_A": feature in FEATURE_SET_A,
                "matrix_B": feature in FEATURE_SET_B,
                "matrix_C": feature in FEATURE_SET_C,
            }
        )
    write_csv(path, rows)


def prepare_output_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def copy_query(
    con: duckdb.DuckDBPyConnection,
    query: str,
    path: Path,
    compression: str,
) -> None:
    prepare_output_file(path)
    con.execute(
        f"""
        COPY ({query}) TO {sql_quote(path.as_posix())}
        (FORMAT PARQUET, COMPRESSION {compression}, ROW_GROUP_SIZE 250000)
        """
    )


def export_artifacts(
    con: duckdb.DuckDBPyConnection, args: argparse.Namespace
) -> dict[str, Path]:
    paths = {
        "metadata": args.output_dir / "05_metadados.parquet",
        "matrix_A": args.output_dir / "06_matriz_A_baseline.parquet",
        "matrix_B": args.output_dir / "07_matriz_B_contexto.parquet",
        "matrix_C": args.output_dir / "08_matriz_C_estendida.parquet",
    }
    copy_query(
        con,
        """
        SELECT
            row_id,
            hash,
            block_hash,
            block_number,
            block_timestamp,
            date,
            from_address,
            to_address,
            transaction_index,
            transaction_type,
            CAST(transaction_type = 4 AS UTINYINT) AS is_type_4
        FROM feature_master
        """,
        paths["metadata"],
        args.compression,
    )
    for key, features in (
        ("matrix_A", FEATURE_SET_A),
        ("matrix_B", FEATURE_SET_B),
        ("matrix_C", FEATURE_SET_C),
    ):
        columns = ", ".join(["row_id", *features])
        copy_query(
            con,
            f"SELECT {columns} FROM feature_master",
            paths[key],
            args.compression,
        )
    return paths


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    files: list[Path],
    total_rows: int,
    unique_hashes: int,
    validation: list[dict[str, object]],
    artifact_paths: dict[str, Path],
) -> None:
    constants = [row["feature"] for row in validation if row["is_constant"]]
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "configuration_file": str(args.config.resolve()),
        "root_template": args.root_template,
        "dataset_name": args.dataset_name,
        "source": "AWS Public Blockchain Dataset - Ethereum transaction Parquets",
        "excluded_external_sources": ["Binance", "Fear & Greed", "TagCloud"],
        "input_column_used": False,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "source_parquet_files": len(files),
        "block_modulus": args.block_modulus,
        "block_sampling_percent": args.block_sampling_percent,
        "sample_rows": total_rows,
        "sample_unique_hashes": unique_hashes,
        "sample_duplicate_hashes": total_rows - unique_hashes,
        "matrix_A": FEATURE_SET_A,
        "matrix_B": FEATURE_SET_B,
        "matrix_C": FEATURE_SET_C,
        "constant_features": constants,
        "artifacts": {
            name: {
                "path": str(file.resolve()),
                "size_bytes": file.stat().st_size,
            }
            for name, file in artifact_paths.items()
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_methodology(path: Path, args: argparse.Namespace) -> None:
    text = f"""# Metodologia de geração das matrizes

- Dataset lógico: `{args.dataset_name}`.
- Período: {args.start_date.isoformat()} a {args.end_date.isoformat()}.
- Amostragem: {args.block_sampling_percent:g}% dos blocos completos, pela regra `block_number % {args.block_modulus} = 0`.
- Fonte: somente Parquets on-chain de transações Ethereum.
- Binance, Fear & Greed e TagCloud não participam das features.
- `input` não participa das features.
- O tipo 2 é a categoria de referência da codificação one-hot.
- O tipo 4 é mantido nos metadados, mas não entra nas matrizes treinadas em 2024.
- Valores de taxa não aplicáveis, representados como zero na origem, permanecem
  zero após a transformação.
- `fee_cap_utilization` recebe zero quando o teto de taxa não é aplicável.
- Criação de contrato reconhece `to_address` nulo, vazio ou igual a `0x`.
- Identificadores e endereços ficam em arquivo separado das entradas numéricas.
- `row_id` é a chave de associação entre metadados e as três matrizes.
- Os Parquets usam compressão {args.compression}.

## Objetivo dos conjuntos

- Matriz A: baseline exclusivamente transacional.
- Matriz B: baseline acrescida do contexto do bloco; candidata principal.
- Matriz C: conjunto estendido para testar limite de gás e custo total por meio
  de ablação.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    files = discover_files(args.root_template, args.start_date, args.end_date)
    print(f"Arquivos Parquet encontrados: {len(files)}")
    con = configure(args)
    try:
        create_raw_view(con, files)
        validate_schema(con)

        print("Criando features e contexto dos blocos...")
        create_feature_master(con, args.block_modulus)
        total_rows, unique_hashes, blocks, contract_creations, type_4 = con.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT hash),
                COUNT(DISTINCT block_number),
                SUM(is_contract_creation),
                SUM(CAST(transaction_type = 4 AS INTEGER))
            FROM feature_master
            """
        ).fetchone()

        if total_rows == 0:
            raise RuntimeError("A amostragem não selecionou nenhuma transação.")
        if total_rows != unique_hashes:
            raise RuntimeError(
                f"Foram encontrados {total_rows - unique_hashes} hashes duplicados."
            )

        print("Validando nulos, infinitos e features constantes...")
        validation = feature_validation(con)
        invalid = [
            row
            for row in validation
            if row["null_count"] or row["non_finite_count"]
        ]
        if invalid:
            names = ", ".join(str(row["feature"]) for row in invalid)
            raise RuntimeError(f"Features com valores inválidos: {names}")

        summary = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "configuration_file": str(args.config.resolve()),
            "root_template": args.root_template,
            "dataset_name": args.dataset_name,
            "start_date": args.start_date.isoformat(),
            "end_date": args.end_date.isoformat(),
            "source_parquet_files": len(files),
            "block_modulus": args.block_modulus,
            "block_sampling_percent": args.block_sampling_percent,
            "sample_rows": int(total_rows),
            "sample_blocks": int(blocks),
            "sample_unique_hashes": int(unique_hashes),
            "contract_creation_rows": int(contract_creations),
            "contract_creation_pct": 100.0 * int(contract_creations) / int(total_rows),
            "type_4_rows": int(type_4),
            "matrix_A_features": len(FEATURE_SET_A),
            "matrix_B_features": len(FEATURE_SET_B),
            "matrix_C_features": len(FEATURE_SET_C),
        }
        (args.output_dir / "01_resumo_geracao.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_feature_dictionary(args.output_dir / "02_dicionario_features.csv")
        write_csv(args.output_dir / "03_validacao_features.csv", validation)

        manifest_preview = {
            "matrix_A": FEATURE_SET_A,
            "matrix_B": FEATURE_SET_B,
            "matrix_C": FEATURE_SET_C,
        }
        (args.output_dir / "04_manifest_matrizes.json").write_text(
            json.dumps(manifest_preview, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Gravando metadados e matrizes A, B e C...")
        artifact_paths = export_artifacts(con, args)
        write_methodology(args.output_dir / "09_metodologia_matrizes.md", args)
        write_manifest(
            args.output_dir / "10_manifest_execucao.json",
            args,
            files,
            int(total_rows),
            int(unique_hashes),
            validation,
            artifact_paths,
        )
        (args.output_dir / "11_arquivos_origem.txt").write_text(
            "\n".join(str(file.resolve()) for file in files) + "\n",
            encoding="utf-8",
        )

        constants = [row["feature"] for row in validation if row["is_constant"]]
        print("\nGeração concluída.")
        print(f"Transações: {int(total_rows):,}")
        print(f"Blocos completos: {int(blocks):,}")
        print(f"Criações de contrato: {int(contract_creations):,}")
        print(f"Transações tipo 4: {int(type_4):,}")
        print("Features constantes: " + (", ".join(constants) or "nenhuma"))
        print(f"Resultados: {args.output_dir.resolve()}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
