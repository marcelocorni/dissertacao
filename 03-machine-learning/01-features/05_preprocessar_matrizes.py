# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
# ]
# ///

"""Ajusta o pré-processamento somente no treino e transforma todos os recortes.

As features contínuas/logarítmicas são centralizadas pela mediana e escaladas
pelo intervalo interquartil (IQR) calculado no treinamento pós-Dencun. Features
binárias, razões, ranks e componentes cíclicos permanecem em sua escala natural.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import duckdb


ROBUST_FEATURES = [
    "gas_used_log",
    "effective_gas_price_log",
    "max_fee_per_gas_log",
    "max_priority_fee_log",
    "transaction_value_log",
    "nonce_log",
    "block_tx_count_log",
    "sender_tx_count_block_log",
    "receiver_tx_count_block_log",
    "gas_limit_log",
    "transaction_fee_paid_log",
]

# Tipos 0 e 1 não utilizam os campos de taxa EIP-1559. Os zeros dessas linhas
# são sentinelas de "não aplicável" e não devem participar do ajuste estatístico.
CONDITIONAL_SCALING = {
    "max_fee_per_gas_log": "tx_type_0 = 0 AND tx_type_1 = 0",
    "max_priority_fee_log": "tx_type_0 = 0 AND tx_type_1 = 0",
}

PASSTHROUGH_FEATURES = [
    "gas_used_ratio",
    "fee_cap_utilization",
    "receipt_status",
    "tx_type_0",
    "tx_type_1",
    "tx_type_3",
    "transaction_index_pct",
    "gas_price_rank_block",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

MATRIX_FILES = {
    "A": ("06_matriz_A_baseline.parquet", "12_matriz_A_preprocessada.parquet"),
    "B": ("07_matriz_B_contexto.parquet", "13_matriz_B_preprocessada.parquet"),
    "C": ("08_matriz_C_estendida.parquet", "14_matriz_C_preprocessada.parquet"),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = here.parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Ajusta RobustScaler no treino e transforma as matrizes A, B e C "
            "de todos os recortes sem vazamento temporal."
        )
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=here / "resultados-splits" / "00_manifest_splits_temporais.json",
        help="Manifesto global produzido pelo artefato 03.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "resultados-preprocessamento",
        help="Diretório central dos parâmetros e relatórios.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=src_root / ".tmp" / "duckdb",
        help="Diretório temporário do DuckDB. Padrão: src/.tmp/duckdb.",
    )
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--threads", type=int, default=max(1, min(8, os.cpu_count() or 4))
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescreve matrizes processadas existentes.",
    )
    args = parser.parse_args()
    if not args.split_manifest.is_file():
        parser.error(f"Manifesto de recortes não encontrado: {args.split_manifest}")
    if args.threads < 1:
        parser.error("--threads deve ser maior ou igual a 1")
    return args


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def configure(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir.as_posix())}")
    con.execute("SET preserve_insertion_order = false")
    return con


def load_splits(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    splits = document.get("splits")
    if not isinstance(splits, list) or not splits:
        raise RuntimeError("O manifesto não contém recortes temporais.")
    train = [split for split in splits if split.get("role") == "train"]
    if len(train) != 1:
        raise RuntimeError("O manifesto deve conter exatamente um recorte de treino.")
    return train[0], splits


def load_matrix_manifest(directory: Path) -> dict[str, object]:
    path = directory / "10_manifest_execucao.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifesto da matriz não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_feature_contract(
    train_directory: Path, splits: list[dict[str, object]]
) -> dict[str, list[str]]:
    train_manifest = load_matrix_manifest(train_directory)
    contract = {
        name: list(train_manifest[f"matrix_{name}"])
        for name in ("A", "B", "C")
    }
    for split in splits:
        directory = Path(str(split["output_directory"]))
        manifest = load_matrix_manifest(directory)
        for name in ("A", "B", "C"):
            current = list(manifest[f"matrix_{name}"])
            if current != contract[name]:
                raise RuntimeError(
                    f"Contrato da matriz {name} diverge em {split['key']}."
                )
    all_features = set(contract["C"])
    classified = set(ROBUST_FEATURES) | set(PASSTHROUGH_FEATURES)
    missing = sorted(all_features - classified)
    unexpected = sorted(classified - all_features)
    if missing or unexpected:
        raise RuntimeError(
            f"Classificação incompatível. Não classificadas={missing}; "
            f"não existentes={unexpected}"
        )
    return contract


def fit_robust_parameters(
    con: duckdb.DuckDBPyConnection, training_matrix_c: Path
) -> dict[str, dict[str, float]]:
    expressions = []
    for feature in ROBUST_FEATURES:
        condition = CONDITIONAL_SCALING.get(feature)
        aggregate_filter = f" FILTER (WHERE {condition})" if condition else ""
        expressions.extend(
            [
                f"QUANTILE_CONT({feature}, 0.25){aggregate_filter} AS {feature}__q1",
                f"QUANTILE_CONT({feature}, 0.50){aggregate_filter} AS {feature}__median",
                f"QUANTILE_CONT({feature}, 0.75){aggregate_filter} AS {feature}__q3",
            ]
        )
    query = (
        "SELECT "
        + ", ".join(expressions)
        + f" FROM read_parquet({sql_quote(training_matrix_c.as_posix())})"
    )
    result = con.execute(query)
    row = result.fetchone()
    names = [item[0] for item in result.description]
    values = dict(zip(names, row))

    parameters: dict[str, dict[str, float]] = {}
    for feature in ROBUST_FEATURES:
        q1 = float(values[f"{feature}__q1"])
        median = float(values[f"{feature}__median"])
        q3 = float(values[f"{feature}__q3"])
        iqr = q3 - q1
        if not all(math.isfinite(value) for value in (q1, median, q3, iqr)):
            raise RuntimeError(f"Parâmetros não finitos para {feature}.")
        scale = iqr if iqr > 0 else 1.0
        parameters[feature] = {
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "scale": scale,
            "scale_source": "iqr" if iqr > 0 else "unit_fallback_zero_iqr",
            "fit_filter": CONDITIONAL_SCALING.get(feature),
            "formula": (
                "0 quando não aplicável; (x - median) / scale quando aplicável"
                if feature in CONDITIONAL_SCALING
                else "(x - median) / scale"
            ),
        }
    return parameters


def transformation_expression(
    feature: str, parameters: dict[str, dict[str, float]]
) -> str:
    if feature in parameters:
        median = parameters[feature]["median"]
        scale = parameters[feature]["scale"]
        scaled = (
            f"((CAST({feature} AS DOUBLE) - ({median:.17g})) "
            f"/ ({scale:.17g}))"
        )
        condition = parameters[feature].get("fit_filter")
        if condition:
            return f"CASE WHEN {condition} THEN {scaled} ELSE 0.0 END AS {feature}"
        return f"{scaled} AS {feature}"
    return f"CAST({feature} AS DOUBLE) AS {feature}"


def output_is_reusable(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    output_path: Path,
    features: list[str],
) -> bool:
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False
    expected_columns = ["row_id", *features]
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_quote(output_path.as_posix())})"
    ).fetchall()
    if [row[0] for row in schema] != expected_columns:
        return False
    input_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_quote(input_path.as_posix())})"
    ).fetchone()[0]
    output_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_quote(output_path.as_posix())})"
    ).fetchone()[0]
    return input_rows == output_rows


def transform_matrix(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    output_path: Path,
    features: list[str],
    parameters: dict[str, dict[str, float]],
) -> None:
    columns = ["row_id", *[transformation_expression(f, parameters) for f in features]]
    con.execute(
        f"""
        COPY (
            SELECT {", ".join(columns)}
            FROM read_parquet({sql_quote(input_path.as_posix())})
        ) TO {sql_quote(output_path.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
        """
    )


def validate_output(
    con: duckdb.DuckDBPyConnection,
    split_key: str,
    matrix_name: str,
    input_path: Path,
    output_path: Path,
    features: list[str],
) -> dict[str, object]:
    invalid_expressions = [
        f"SUM(CASE WHEN {feature} IS NULL OR "
        f"NOT ISFINITE(CAST({feature} AS DOUBLE)) THEN 1 ELSE 0 END)"
        for feature in features
    ]
    input_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_quote(input_path.as_posix())})"
    ).fetchone()[0]
    validation = con.execute(
        f"""
        SELECT COUNT(*), {" + ".join(invalid_expressions)}
        FROM read_parquet({sql_quote(output_path.as_posix())})
        """
    ).fetchone()
    output_rows = int(validation[0])
    invalid_count = int(validation[1])
    if input_rows != output_rows:
        raise RuntimeError(
            f"Contagem divergente em {split_key}, matriz {matrix_name}."
        )
    if invalid_count:
        raise RuntimeError(
            f"{invalid_count} valores inválidos em {split_key}, matriz {matrix_name}."
        )
    return {
        "split": split_key,
        "matrix": matrix_name,
        "rows": output_rows,
        "features": len(features),
        "invalid_values": invalid_count,
        "output_path": str(output_path.resolve()),
        "size_bytes": output_path.stat().st_size,
    }


def training_profile(
    con: duckdb.DuckDBPyConnection,
    matrix_path: Path,
    features: list[str],
) -> list[dict[str, object]]:
    rows = []
    for feature in features:
        values = con.execute(
            f"""
            SELECT
                MIN({feature}),
                QUANTILE_CONT({feature}, 0.25),
                QUANTILE_CONT({feature}, 0.50),
                AVG({feature}),
                QUANTILE_CONT({feature}, 0.75),
                MAX({feature}),
                STDDEV_SAMP({feature})
            FROM read_parquet({sql_quote(matrix_path.as_posix())})
            """
        ).fetchone()
        rows.append(
            {
                "feature": feature,
                "transformation": (
                    "robust_scaler_train" if feature in ROBUST_FEATURES else "passthrough"
                ),
                "minimum": values[0],
                "q1": values[1],
                "median": values[2],
                "mean": values[3],
                "q3": values[4],
                "maximum": values[5],
                "stddev": values[6],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    train, splits = load_splits(args.split_manifest)
    train_directory = Path(str(train["output_directory"]))
    contract = validate_feature_contract(train_directory, splits)
    con = configure(args)
    try:
        training_c = train_directory / MATRIX_FILES["C"][0]
        print("Ajustando RobustScaler exclusivamente no treinamento...")
        parameters = fit_robust_parameters(con, training_c)
        preprocessing_signature = hashlib.sha256(
            json.dumps(
                {"parameters": parameters, "matrices": contract},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        parameter_document = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "fit_split": train["key"],
            "fit_start_date": train["start_date"],
            "fit_end_date": train["end_date"],
            "method": "RobustScaler univariado com taxas EIP-1559 condicionais",
            "formula": "(x - mediana_treino) / escala_treino",
            "zero_iqr_policy": "usar escala 1 após centralização pela mediana",
            "not_applicable_fee_policy": (
                "max_fee e max_priority permanecem zero nos tipos 0 e 1; "
                "ajuste calculado somente nos tipos 2, 3 e 4"
            ),
            "preprocessing_signature": preprocessing_signature,
            "parameters": parameters,
        }
        (args.output_dir / "01_parametros_robust_scaler.json").write_text(
            json.dumps(parameter_document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        dictionary = []
        for feature in contract["C"]:
            dictionary.append(
                {
                    "feature": feature,
                    "transformation": (
                        "conditional_robust_scaler_train"
                        if feature in CONDITIONAL_SCALING
                        else "robust_scaler_train"
                        if feature in parameters
                        else "passthrough"
                    ),
                    "fit_source": (
                        "treino_2024_pos_dencun"
                        if feature in parameters
                        else "not_applicable"
                    ),
                    "matrix_A": feature in contract["A"],
                    "matrix_B": feature in contract["B"],
                    "matrix_C": feature in contract["C"],
                }
            )
        write_csv(args.output_dir / "02_dicionario_transformacoes.csv", dictionary)

        validations: list[dict[str, object]] = []
        for split in splits:
            split_key = str(split["key"])
            directory = Path(str(split["output_directory"]))
            split_manifest_path = directory / "15_manifest_preprocessamento.json"
            split_signature_matches = False
            if split_manifest_path.is_file():
                try:
                    previous = json.loads(
                        split_manifest_path.read_text(encoding="utf-8")
                    )
                    split_signature_matches = (
                        previous.get("preprocessing_signature")
                        == preprocessing_signature
                    )
                except (OSError, json.JSONDecodeError):
                    split_signature_matches = False
            print(f"Processando {split_key}...")
            split_outputs = []
            for matrix_name, (input_name, output_name) in MATRIX_FILES.items():
                input_path = directory / input_name
                output_path = directory / output_name
                if not input_path.is_file():
                    raise FileNotFoundError(f"Matriz de entrada ausente: {input_path}")
                reusable = split_signature_matches and output_is_reusable(
                    con, input_path, output_path, contract[matrix_name]
                )
                if reusable and not args.force:
                    print(f"  REUTILIZADA matriz {matrix_name}")
                else:
                    if output_path.exists():
                        output_path.unlink()
                    transform_matrix(
                        con,
                        input_path,
                        output_path,
                        contract[matrix_name],
                        parameters,
                    )
                    print(f"  GERADA matriz {matrix_name}")
                validation = validate_output(
                    con,
                    split_key,
                    matrix_name,
                    input_path,
                    output_path,
                    contract[matrix_name],
                )
                validations.append(validation)
                split_outputs.append(validation)
            split_manifest_path.write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now().astimezone().isoformat(),
                        "split": split_key,
                        "preprocessing_signature": preprocessing_signature,
                        "fit_split": train["key"],
                        "outputs": split_outputs,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        training_b_processed = train_directory / MATRIX_FILES["B"][1]
        profile = training_profile(con, training_b_processed, contract["B"])
        write_csv(
            args.output_dir / "03_perfil_treino_matriz_B_preprocessada.csv",
            profile,
        )
        write_csv(args.output_dir / "04_validacao_recortes.csv", validations)

        manifest = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "source_split_manifest": str(args.split_manifest.resolve()),
            "fit_split": train["key"],
            "preprocessing_signature": preprocessing_signature,
            "leakage_policy": (
                "Parâmetros ajustados somente no treino e congelados nos demais recortes."
            ),
            "robust_scaled_features": ROBUST_FEATURES,
            "passthrough_features": PASSTHROUGH_FEATURES,
            "matrices": contract,
            "validation": validations,
        }
        (args.output_dir / "05_manifest_preprocessamento.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\nPré-processamento concluído sem vazamento temporal.")
        print(f"Parâmetros e relatórios: {args.output_dir.resolve()}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
