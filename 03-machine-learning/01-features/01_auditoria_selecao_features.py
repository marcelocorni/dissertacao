# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0,<3",
#   "pandas>=2.2,<3",
# ]
# ///

"""Auditoria e seleção preliminar de features de transações Ethereum.

O programa lê diretamente os arquivos Parquet, seleciona blocos completos de
forma determinística, cria features on-chain e produz relatórios de qualidade,
distribuição e correlação.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configuracao import (
    ConfiguracaoPipelineError,
    carregar_amostragem_blocos,
    carregar_root_template,
)


MODEL_FEATURES = [
    "gas_limit_log",
    "gas_used_log",
    "gas_used_ratio",
    "effective_gas_price_log",
    "max_fee_per_gas_log",
    "max_priority_fee_log",
    "fee_cap_utilization",
    "transaction_fee_paid_log",
    "transaction_value_log",
    "is_zero_value",
    "nonce_log",
    "receipt_status",
    "tx_type_0",
    "tx_type_1",
    "tx_type_2",
    "tx_type_3",
    "tx_type_4",
    "tx_type_other",
    "max_fee_missing",
    "priority_fee_missing",
    "transaction_index_pct",
    "is_contract_creation",
    "creates_contract",
    "block_tx_count_log",
    "sender_tx_count_block_log",
    "receiver_tx_count_block_log",
    "gas_price_rank_block",
    "value_rank_block",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

IDENTIFIER_COLUMNS = [
    "hash",
    "block_hash",
    "block_number",
    "block_timestamp",
    "date",
    "from_address",
    "to_address",
    "transaction_index",
]

RAW_AUDIT_COLUMNS = [
    "gas",
    "value",
    "gas_price",
    "receipt_gas_used",
    "receipt_status",
    "transaction_type",
    "max_priority_fee_per_gas",
    "nonce",
    "block_number",
    "transaction_index",
    "receipt_cumulative_gas_used",
    "receipt_effective_gas_price",
    "max_fee_per_gas",
]

DATE_DIR_RE = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = SRC_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Cria uma amostra por blocos completos e gera auditoria, perfil e "
            "matrizes de correlação das features on-chain."
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
        help="Primeiro dia do período de treinamento (inclusive).",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date(2024, 12, 31),
        help="Último dia do período de treinamento (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "resultados",
        help="Diretório no qual os relatórios serão gravados.",
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
            "Substitui a amostragem configurada nesta execução. Seleciona "
            "block_number %% valor = 0; padrão atual: "
            f"{configured_percent:g}%% (módulo {configured_modulus})."
        ),
    )
    parser.add_argument(
        "--correlation-rows",
        type=int,
        default=500_000,
        help="Máximo de registros usados nas correlações Pearson e Spearman.",
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
        help="Quantidade de threads do DuckDB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240916,
        help="Semente da subamostra usada nas correlações.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="Limiar absoluto para listar pares potencialmente redundantes.",
    )
    parser.add_argument(
        "--keep-feature-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mantém a amostra de features em Parquet para análises posteriores.",
    )
    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("--start-date não pode ser posterior a --end-date")
    if args.block_modulus < 1:
        parser.error("--block-modulus deve ser maior ou igual a 1")
    args.block_sampling_percent = 100.0 / args.block_modulus
    if args.correlation_rows < 1:
        parser.error("--correlation-rows deve ser maior ou igual a 1")
    if args.threads < 1:
        parser.error("--threads deve ser maior ou igual a 1")
    if not 0 < args.correlation_threshold <= 1:
        parser.error("--correlation-threshold deve estar no intervalo (0, 1]")
    return args


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_file_list(files: Iterable[Path]) -> str:
    return "[" + ", ".join(sql_quote(path.as_posix()) for path in files) + "]"


def discover_files(root_template: str, start: date, end: date) -> list[Path]:
    files: list[Path] = []
    for year in range(start.year, end.year + 1):
        root = Path(root_template.format(year=year))
        if not root.exists():
            print(f"AVISO: diretório não encontrado: {root}", file=sys.stderr)
            continue

        for day_dir in sorted(root.glob("date=*")):
            match = DATE_DIR_RE.match(day_dir.name)
            if not match:
                continue
            day = date.fromisoformat(match.group(1))
            if start <= day <= end:
                files.extend(sorted(day_dir.glob("*.parquet")))

    if not files:
        raise FileNotFoundError(
            "Nenhum Parquet foi encontrado no período solicitado. Confira "
            "--root-template, --start-date e --end-date."
        )
    return files


def configure_duckdb(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir.as_posix())}")
    con.execute("SET preserve_insertion_order = false")
    return con


def create_raw_view(con: duckdb.DuckDBPyConnection, files: list[Path]) -> None:
    source = sql_file_list(files)
    con.execute(
        f"""
        CREATE OR REPLACE VIEW raw_transactions AS
        SELECT *
        FROM read_parquet(
            {source},
            hive_partitioning = false,
            union_by_name = true
        )
        """
    )


def validate_schema(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    schema = con.execute("DESCRIBE SELECT * FROM raw_transactions").fetchdf()
    available = set(schema["column_name"].tolist())
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
        "receipt_contract_address",
        "receipt_effective_gas_price",
        "max_fee_per_gas",
    }
    missing = sorted(required - available)
    if missing:
        raise RuntimeError("Colunas obrigatórias ausentes: " + ", ".join(missing))
    return schema


def create_feature_table(
    con: duckdb.DuckDBPyConnection, block_modulus: int
) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE feature_sample AS
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
                gas,
                value,
                gas_price,
                receipt_gas_used,
                receipt_status,
                transaction_type,
                max_priority_fee_per_gas,
                nonce,
                receipt_contract_address,
                receipt_effective_gas_price,
                max_fee_per_gas,
                COALESCE(receipt_effective_gas_price, gas_price) AS effective_price,
                COUNT(*) OVER (PARTITION BY block_number) AS block_tx_count,
                COUNT(*) OVER (
                    PARTITION BY block_number, from_address
                ) AS sender_tx_count_block,
                COUNT(*) OVER (
                    PARTITION BY block_number, to_address
                ) AS receiver_tx_count_block,
                PERCENT_RANK() OVER (
                    PARTITION BY block_number
                    ORDER BY COALESCE(receipt_effective_gas_price, gas_price)
                ) AS gas_price_rank_block,
                PERCENT_RANK() OVER (
                    PARTITION BY block_number ORDER BY value
                ) AS value_rank_block
            FROM raw_transactions
            WHERE block_number % {block_modulus} = 0
        )
        SELECT
            hash,
            block_hash,
            block_number,
            block_timestamp,
            date,
            from_address,
            to_address,
            transaction_index,

            LN(1.0 + GREATEST(COALESCE(gas, 0), 0)) AS gas_limit_log,
            LN(1.0 + GREATEST(COALESCE(receipt_gas_used, 0), 0)) AS gas_used_log,
            CASE
                WHEN gas > 0 AND receipt_gas_used IS NOT NULL
                THEN CAST(receipt_gas_used AS DOUBLE) / CAST(gas AS DOUBLE)
            END AS gas_used_ratio,
            CASE
                WHEN effective_price IS NOT NULL
                THEN LN(1.0 + GREATEST(CAST(effective_price AS DOUBLE), 0.0))
            END AS effective_gas_price_log,
            CASE
                WHEN max_fee_per_gas IS NOT NULL
                THEN LN(1.0 + GREATEST(CAST(max_fee_per_gas AS DOUBLE), 0.0))
            END AS max_fee_per_gas_log,
            CASE
                WHEN max_priority_fee_per_gas IS NOT NULL
                THEN LN(1.0 + GREATEST(
                    CAST(max_priority_fee_per_gas AS DOUBLE), 0.0
                ))
            END AS max_priority_fee_log,
            CASE
                WHEN max_fee_per_gas > 0 AND effective_price IS NOT NULL
                THEN CAST(effective_price AS DOUBLE)
                     / CAST(max_fee_per_gas AS DOUBLE)
            END AS fee_cap_utilization,
            CASE
                WHEN receipt_gas_used IS NOT NULL AND effective_price IS NOT NULL
                THEN LN(
                    1.0
                    + GREATEST(CAST(receipt_gas_used AS DOUBLE), 0.0)
                      * GREATEST(CAST(effective_price AS DOUBLE), 0.0)
                )
            END AS transaction_fee_paid_log,
            CASE
                WHEN value IS NOT NULL
                THEN LN(1.0 + GREATEST(CAST(value AS DOUBLE), 0.0))
            END AS transaction_value_log,
            CAST(COALESCE(value, 0) = 0 AS UTINYINT) AS is_zero_value,
            CASE
                WHEN nonce IS NOT NULL
                THEN LN(1.0 + GREATEST(CAST(nonce AS DOUBLE), 0.0))
            END AS nonce_log,
            CAST(receipt_status AS UTINYINT) AS receipt_status,
            CAST(transaction_type = 0 AS UTINYINT) AS tx_type_0,
            CAST(transaction_type = 1 AS UTINYINT) AS tx_type_1,
            CAST(transaction_type = 2 AS UTINYINT) AS tx_type_2,
            CAST(transaction_type = 3 AS UTINYINT) AS tx_type_3,
            CAST(transaction_type = 4 AS UTINYINT) AS tx_type_4,
            CAST(
                transaction_type IS NULL
                OR transaction_type NOT IN (0, 1, 2, 3, 4)
                AS UTINYINT
            ) AS tx_type_other,
            CAST(max_fee_per_gas IS NULL AS UTINYINT) AS max_fee_missing,
            CAST(
                max_priority_fee_per_gas IS NULL AS UTINYINT
            ) AS priority_fee_missing,
            CASE
                WHEN block_tx_count > 1 AND transaction_index IS NOT NULL
                THEN CAST(transaction_index AS DOUBLE) / (block_tx_count - 1)
            END AS transaction_index_pct,
            CAST(to_address IS NULL AS UTINYINT) AS is_contract_creation,
            CAST(receipt_contract_address IS NOT NULL AS UTINYINT) AS creates_contract,
            LN(1.0 + block_tx_count) AS block_tx_count_log,
            LN(1.0 + sender_tx_count_block) AS sender_tx_count_block_log,
            LN(1.0 + receiver_tx_count_block) AS receiver_tx_count_block_log,
            gas_price_rank_block,
            value_rank_block,
            SIN(2.0 * PI() * EXTRACT(hour FROM block_timestamp) / 24.0) AS hour_sin,
            COS(2.0 * PI() * EXTRACT(hour FROM block_timestamp) / 24.0) AS hour_cos,
            SIN(2.0 * PI() * EXTRACT(dow FROM block_timestamp) / 7.0) AS weekday_sin,
            COS(2.0 * PI() * EXTRACT(dow FROM block_timestamp) / 7.0) AS weekday_cos
        FROM sampled_blocks
        """
    )


def dataset_summary(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    raw = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            APPROX_COUNT_DISTINCT(hash) AS approximate_unique_hashes,
            MIN(block_number) AS min_block_number,
            MAX(block_number) AS max_block_number,
            MIN(block_timestamp) AS min_block_timestamp,
            MAX(block_timestamp) AS max_block_timestamp
        FROM raw_transactions
        """
    ).fetchone()
    sample = con.execute(
        """
        SELECT
            COUNT(*) AS sampled_rows,
            COUNT(DISTINCT block_number) AS sampled_blocks,
            COUNT(DISTINCT hash) AS sample_unique_hashes,
            MIN(block_timestamp) AS sample_min_timestamp,
            MAX(block_timestamp) AS sample_max_timestamp
        FROM feature_sample
        """
    ).fetchone()

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "configuration_file": str(args.config.resolve()),
        "root_template": args.root_template,
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "number_of_parquet_files": len(files),
        "total_rows": int(raw[0]),
        "approximate_unique_hashes": int(raw[1]),
        "note_unique_hashes": (
            "A cardinalidade global é aproximada para evitar uma agregação "
            "muito onerosa sobre centenas de milhões de hashes."
        ),
        "min_block_number": int(raw[2]),
        "max_block_number": int(raw[3]),
        "min_block_timestamp": raw[4].isoformat(),
        "max_block_timestamp": raw[5].isoformat(),
        "block_modulus": args.block_modulus,
        "block_sampling_percent": args.block_sampling_percent,
        "sampled_rows": int(sample[0]),
        "sampled_blocks": int(sample[1]),
        "sample_unique_hashes": int(sample[2]),
        "sample_duplicate_hashes": int(sample[0] - sample[2]),
        "sample_min_timestamp": sample[3].isoformat(),
        "sample_max_timestamp": sample[4].isoformat(),
        "correlation_rows_requested": args.correlation_rows,
        "seed": args.seed,
    }


def raw_null_report(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    expressions: list[str] = []
    for column in RAW_AUDIT_COLUMNS + [
        "hash",
        "from_address",
        "to_address",
        "receipt_contract_address",
        "block_timestamp",
        "last_modified",
    ]:
        expressions.append(
            f"SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END) AS {column}"
        )

    counts = con.execute(
        "SELECT COUNT(*) AS total, " + ", ".join(expressions) + " FROM raw_transactions"
    ).fetchone()
    total = int(counts[0])
    rows = []
    columns = RAW_AUDIT_COLUMNS + [
        "hash",
        "from_address",
        "to_address",
        "receipt_contract_address",
        "block_timestamp",
        "last_modified",
    ]
    for position, column in enumerate(columns, start=1):
        nulls = int(counts[position])
        rows.append(
            {
                "column": column,
                "total_rows": total,
                "null_count": nulls,
                "null_pct": 100.0 * nulls / total if total else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False)


def feature_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = con.execute("SELECT COUNT(*) FROM feature_sample").fetchone()[0]
    for feature in MODEL_FEATURES:
        result = con.execute(
            f"""
            SELECT
                COUNT({feature}) AS valid_count,
                COUNT(*) - COUNT({feature}) AS null_count,
                COUNT(DISTINCT {feature}) AS distinct_count,
                MIN({feature}) AS minimum,
                QUANTILE_CONT({feature}, 0.01) AS p01,
                QUANTILE_CONT({feature}, 0.25) AS p25,
                QUANTILE_CONT({feature}, 0.50) AS median,
                AVG({feature}) AS mean,
                QUANTILE_CONT({feature}, 0.75) AS p75,
                QUANTILE_CONT({feature}, 0.99) AS p99,
                MAX({feature}) AS maximum,
                STDDEV_SAMP({feature}) AS stddev
            FROM feature_sample
            """
        ).fetchone()
        rows.append(
            {
                "feature": feature,
                "total_rows": int(total),
                "valid_count": int(result[0]),
                "null_count": int(result[1]),
                "null_pct": 100.0 * int(result[1]) / int(total) if total else math.nan,
                "distinct_count": int(result[2]),
                "minimum": result[3],
                "p01": result[4],
                "p25": result[5],
                "median": result[6],
                "mean": result[7],
                "p75": result[8],
                "p99": result[9],
                "maximum": result[10],
                "stddev": result[11],
            }
        )
    return pd.DataFrame(rows)


def category_report(
    con: duckdb.DuckDBPyConnection, column: str
) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT
            {column} AS value,
            COUNT(*) AS row_count,
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS percentage
        FROM raw_transactions
        GROUP BY {column}
        ORDER BY row_count DESC
        """
    ).fetchdf()


def correlation_frame(
    con: duckdb.DuckDBPyConnection, rows: int, seed: int
) -> pd.DataFrame:
    selected = ", ".join(MODEL_FEATURES)
    available = con.execute("SELECT COUNT(*) FROM feature_sample").fetchone()[0]
    if available <= rows:
        query = f"SELECT {selected} FROM feature_sample"
    else:
        query = f"""
            SELECT {selected}
            FROM feature_sample
            USING SAMPLE reservoir({rows} ROWS) REPEATABLE ({seed})
        """
    frame = con.execute(query).fetchdf()
    return frame.replace([np.inf, -np.inf], np.nan)


def remove_constant_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    variable = []
    constants = []
    for column in frame.columns:
        if frame[column].nunique(dropna=True) <= 1:
            constants.append(column)
        else:
            variable.append(column)
    return frame[variable], constants


def correlated_pairs(
    matrix: pd.DataFrame, method: str, threshold: float
) -> pd.DataFrame:
    rows = []
    for i, first in enumerate(matrix.columns):
        for second in matrix.columns[i + 1 :]:
            value = matrix.loc[first, second]
            if pd.notna(value) and abs(float(value)) >= threshold:
                rows.append(
                    {
                        "method": method,
                        "feature_1": first,
                        "feature_2": second,
                        "correlation": float(value),
                        "absolute_correlation": abs(float(value)),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "feature_1",
            "feature_2",
            "correlation",
            "absolute_correlation",
        ],
    ).sort_values("absolute_correlation", ascending=False)


def save_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    size = max(12, 0.46 * len(matrix.columns))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(matrix.columns, fontsize=7)
    ax.set_title(title, fontsize=14, pad=16)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlação")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_methodology(args: argparse.Namespace, files: list[Path], path: Path) -> None:
    content = f"""# Metodologia da auditoria de features

- Período: {args.start_date.isoformat()} a {args.end_date.isoformat()}.
- Arquivos Parquet: {len(files)}.
- Fonte: somente transações on-chain do AWS Public Blockchain Dataset.
- Fear & Greed: não utilizado.
- Binance: não utilizado.
- TagCloud: não utilizado como feature; reservado para comparação final.
- `input`: não utilizado.
- Seleção amostral: {args.block_sampling_percent:g}% dos blocos, pela regra determinística `block_number % {args.block_modulus} = 0`.
- A unidade amostral é o bloco completo, preservando posição, frequência e percentis intrabloco.
- Correlações: no máximo {args.correlation_rows:,} registros, semente {args.seed}.
- Limiar de correlação potencialmente redundante: |r| >= {args.correlation_threshold:.2f}.
- Pearson mede associação linear; Spearman mede associação monotônica.
- As correlações são diagnósticas e não determinam remoção automática de features.
- O período analisado deve pertencer exclusivamente ao conjunto de treinamento.
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    files = discover_files(args.root_template, args.start_date, args.end_date)
    print(f"Encontrados {len(files)} arquivos Parquet.")

    con = configure_duckdb(args)
    try:
        create_raw_view(con, files)
        schema = validate_schema(con)
        schema.to_csv(args.output_dir / "01_schema_parquet.csv", index=False)

        print("Criando amostra determinística por blocos completos...")
        create_feature_table(con, args.block_modulus)

        summary = dataset_summary(con, files, args)
        (args.output_dir / "02_resumo_dataset.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("Calculando nulidade das colunas originais...")
        raw_null_report(con).to_csv(
            args.output_dir / "03_nulidade_colunas_originais.csv", index=False
        )
        category_report(con, "transaction_type").to_csv(
            args.output_dir / "04_distribuicao_transaction_type.csv", index=False
        )
        category_report(con, "receipt_status").to_csv(
            args.output_dir / "05_distribuicao_receipt_status.csv", index=False
        )

        print("Calculando perfil estatístico das features criadas...")
        profile = feature_profile(con)
        profile.to_csv(args.output_dir / "06_perfil_features.csv", index=False)

        if args.keep_feature_parquet:
            feature_path = (args.output_dir / "07_amostra_features.parquet").as_posix()
            con.execute(
                f"""
                COPY feature_sample TO {sql_quote(feature_path)}
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
                """
            )

        print("Gerando matrizes de correlação...")
        frame = correlation_frame(con, args.correlation_rows, args.seed)
        frame, constants = remove_constant_columns(frame)
        pd.DataFrame({"constant_or_absent_feature": constants}).to_csv(
            args.output_dir / "08_features_constantes_na_amostra.csv", index=False
        )

        pearson = frame.corr(method="pearson", min_periods=100)
        spearman = frame.corr(method="spearman", min_periods=100)
        pearson.to_csv(args.output_dir / "09_correlacao_pearson.csv")
        spearman.to_csv(args.output_dir / "10_correlacao_spearman.csv")
        save_heatmap(
            pearson,
            "Correlação de Pearson — features on-chain",
            args.output_dir / "11_correlacao_pearson.png",
        )
        save_heatmap(
            spearman,
            "Correlação de Spearman — features on-chain",
            args.output_dir / "12_correlacao_spearman.png",
        )

        pairs = pd.concat(
            [
                correlated_pairs(
                    pearson, "pearson", args.correlation_threshold
                ),
                correlated_pairs(
                    spearman, "spearman", args.correlation_threshold
                ),
            ],
            ignore_index=True,
        )
        pairs.to_csv(
            args.output_dir / "13_pares_alta_correlacao.csv", index=False
        )

        write_methodology(
            args, files, args.output_dir / "14_metodologia_execucao.md"
        )

        result = {
            **summary,
            "correlation_rows_used": int(len(frame)),
            "features_evaluated": list(frame.columns),
            "constant_features": constants,
            "high_correlation_pairs": int(len(pairs)),
            "output_directory": str(args.output_dir.resolve()),
        }
        (args.output_dir / "15_resultado_execucao.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\nAuditoria concluída.")
        print(f"Registros totais no período: {summary['total_rows']:,}")
        print(f"Registros na amostra por blocos: {summary['sampled_rows']:,}")
        print(f"Registros usados na correlação: {len(frame):,}")
        print(f"Resultados: {args.output_dir.resolve()}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
