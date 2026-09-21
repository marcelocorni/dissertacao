# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
#   "joblib>=1.4,<2",
#   "matplotlib>=3.10,<4",
#   "numpy>=2.1,<3",
#   "pyarrow>=18,<24",
#   "scikit-learn>=1.6,<2",
# ]
# ///

"""Treina variantes congeladas do Isolation Forest e pontua todos os recortes.

O estimador aprende somente no recorte de treino. Os limiares são quantis dos
escores da validação pré-Pectra. Transições, teste e estresse futuro são apenas
pontuados e nunca alteram modelo, conjunto de features ou limiares.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import IsolationForest


MATRIX_FILES = {
    "A": "12_matriz_A_preprocessada.parquet",
    "B": "13_matriz_B_preprocessada.parquet",
    "C": "14_matriz_C_preprocessada.parquet",
}

# Família sem taxas absolutas, definida antes do treino dos modelos. Mantém
# medidas relativas (fee_cap_utilization e gas_price_rank_block).
ABSOLUTE_FEE_FEATURES = {
    "effective_gas_price_log",
    "max_fee_per_gas_log",
    "max_priority_fee_log",
    "transaction_fee_paid_log",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = here.parents[1]
    feature_root = here.parent / "01-features"
    parser = argparse.ArgumentParser(
        description="Treina e avalia o baseline Isolation Forest sem vazamento temporal."
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=feature_root / "resultados-splits" / "00_manifest_splits_temporais.json",
    )
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        default=feature_root / "resultados-preprocessamento" / "05_manifest_preprocessamento.json",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "resultados")
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--training-sample-size", type=int, default=500_000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-samples-tree", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for path in (args.split_manifest, args.preprocessing_manifest):
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")
    for name in ("threads", "training_sample_size", "n_estimators", "max_samples_tree", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} deve ser positivo")
    return args


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_duckdb(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir.as_posix())}")
    con.execute("SET preserve_insertion_order = false")
    return con


def input_path(split: dict[str, object], matrix: str) -> Path:
    return Path(str(split["output_directory"])) / MATRIX_FILES[matrix]


def metadata_path(split: dict[str, object]) -> Path:
    return Path(str(split["output_directory"])) / "05_metadados.parquet"


def validate_inputs(
    splits: list[dict[str, object]], preprocessing: dict[str, object]
) -> None:
    expected = str(preprocessing["preprocessing_signature"])
    for split in splits:
        directory = Path(str(split["output_directory"]))
        manifest_path = directory / "15_manifest_preprocessamento.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifesto ausente: {manifest_path}")
        if str(load_json(manifest_path).get("preprocessing_signature")) != expected:
            raise RuntimeError(f"Pré-processamento incompatível em {split['key']}.")
        for matrix in MATRIX_FILES:
            if not input_path(split, matrix).is_file():
                raise FileNotFoundError(f"Matriz ausente: {input_path(split, matrix)}")
        if not metadata_path(split).is_file():
            raise FileNotFoundError(f"Metadados ausentes: {metadata_path(split)}")


def build_configurations(preprocessing: dict[str, object]) -> list[dict[str, object]]:
    matrices = preprocessing["matrices"]
    configs: list[dict[str, object]] = []
    for matrix in ("A", "B", "C"):
        configs.append(
            {
                "key": f"{matrix.lower()}_completa",
                "matrix": matrix,
                "family": "completa",
                "features": list(matrices[matrix]),
                "excluded_features": [],
            }
        )
    for matrix in ("B", "C"):
        original = list(matrices[matrix])
        excluded = [feature for feature in original if feature in ABSOLUTE_FEE_FEATURES]
        configs.append(
            {
                "key": f"{matrix.lower()}_sem_taxas_absolutas",
                "matrix": matrix,
                "family": "sem_taxas_absolutas",
                "features": [feature for feature in original if feature not in ABSOLUTE_FEE_FEATURES],
                "excluded_features": excluded,
            }
        )
    return configs


def deterministic_training_sample(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    features: list[str],
    requested_rows: int,
) -> tuple[np.ndarray, int, int]:
    total = int(con.execute(f"SELECT COUNT(*) FROM read_parquet({sql_quote(path.as_posix())})").fetchone()[0])
    target = min(total, requested_rows)
    columns = ", ".join(ident(feature) for feature in features)
    if target == total:
        where = ""
    else:
        # hash(row_id) distribui a amostra por todo o período sem ordenação cara.
        where = f"WHERE hash(row_id) % {total} < {target}"
    arrays = con.execute(
        f"SELECT {columns} FROM read_parquet({sql_quote(path.as_posix())}) {where}"
    ).fetchnumpy()
    matrix = np.column_stack([arrays[feature] for feature in features]).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("A amostra de treinamento contém valores inválidos.")
    return matrix, total, matrix.shape[0]


def score_batches(
    estimator: IsolationForest,
    path: Path,
    features: list[str],
    batch_size: int,
    output_path: Path | None,
    thresholds: dict[str, float] | None,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    parquet = pq.ParquetFile(path)
    writer: pq.ParquetWriter | None = None
    all_scores: list[np.ndarray] = []
    all_ids: list[np.ndarray] = []
    try:
        with joblib.parallel_backend("threading", n_jobs=threads):
            for batch in parquet.iter_batches(
                batch_size=batch_size, columns=["row_id", *features]
            ):
                row_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
                values = np.column_stack(
                    [batch.column(index).to_numpy(zero_copy_only=False) for index in range(1, len(features) + 1)]
                ).astype(np.float32, copy=False)
                scores = (-estimator.score_samples(values)).astype(np.float32)
                all_ids.append(row_ids)
                all_scores.append(scores)
                if output_path is not None and thresholds is not None:
                    table = pa.table(
                        {
                            "row_id": row_ids,
                            "anomaly_score": scores,
                            "is_anomaly_q990": scores >= thresholds["q990"],
                            "is_anomaly_q995": scores >= thresholds["q995"],
                            "is_anomaly_q999": scores >= thresholds["q999"],
                        }
                    )
                    if writer is None:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return np.concatenate(all_ids), np.concatenate(all_scores)


def score_statistics(scores: np.ndarray) -> dict[str, float]:
    return {
        "score_min": float(np.min(scores)),
        "score_q250": float(np.quantile(scores, 0.25)),
        "score_median": float(np.quantile(scores, 0.50)),
        "score_mean": float(np.mean(scores)),
        "score_q750": float(np.quantile(scores, 0.75)),
        "score_q950": float(np.quantile(scores, 0.95)),
        "score_q990": float(np.quantile(scores, 0.99)),
        "score_q995": float(np.quantile(scores, 0.995)),
        "score_q999": float(np.quantile(scores, 0.999)),
        "score_max": float(np.max(scores)),
        "score_std": float(np.std(scores)),
    }


def create_rate_plot(
    path_pdf: Path,
    path_png: Path,
    rates: list[dict[str, object]],
    splits: list[dict[str, object]],
    configs: list[dict[str, object]],
) -> None:
    import matplotlib.pyplot as plt

    split_keys = [str(split["key"]) for split in splits]
    lookup = {(str(row["configuration"]), str(row["split"])): float(row["anomaly_rate_q995"]) * 100 for row in rates}
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for config in configs:
        key = str(config["key"])
        ax.plot(range(len(split_keys)), [lookup[(key, split)] for split in split_keys], marker="o", linewidth=1.8, label=key)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="calibração 0,5%")
    ax.set_xticks(range(len(split_keys)), [key.replace("_", "\n") for key in split_keys], fontsize=8)
    ax.set_ylabel("Transações sinalizadas (%)")
    ax.set_xlabel("Recorte temporal")
    ax.set_title("Isolation Forest: taxa de anomalias pelo limiar Q99,5 da validação")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "modelos"
    score_root = args.output_dir / "escores"
    model_dir.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)

    split_document = load_json(args.split_manifest)
    preprocessing = load_json(args.preprocessing_manifest)
    splits = list(split_document["splits"])
    validate_inputs(splits, preprocessing)
    train_matches = [split for split in splits if split["role"] == "train"]
    validation_matches = [split for split in splits if split["role"] == "validation"]
    if len(train_matches) != 1 or len(validation_matches) != 1:
        raise RuntimeError("O desenho deve conter exatamente um treino e uma validação.")
    train, validation = train_matches[0], validation_matches[0]
    configs = build_configurations(preprocessing)
    con = configure_duckdb(args)

    experiment = {
        "generated_at": now_iso(),
        "method": "IsolationForest",
        "fit_split": train["key"],
        "threshold_split": validation["key"],
        "selection_policy": "configurações congeladas antes da pontuação do teste",
        "stable_family_policy": "remove taxas absolutas; preserva razões e ranks",
        "random_state": args.random_state,
        "training_sample_requested": args.training_sample_size,
        "n_estimators": args.n_estimators,
        "max_samples_tree": args.max_samples_tree,
        "contamination": "auto; não usado como limiar final",
        "threshold_quantiles_validation": [0.990, 0.995, 0.999],
        "configurations": configs,
        "preprocessing_signature": preprocessing["preprocessing_signature"],
        "software_versions": {
            "python": platform.python_version(),
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    signature_payload = {
        key: value for key, value in experiment.items() if key != "generated_at"
    }
    experiment_signature = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    experiment["experiment_signature"] = experiment_signature
    (args.output_dir / "01_configuracoes_experimento.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    training_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    statistics_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []
    type4_rows: list[dict[str, object]] = []

    for config_index, config in enumerate(configs, start=1):
        key = str(config["key"])
        matrix_name = str(config["matrix"])
        features = list(config["features"])
        print(f"\n[{config_index}/{len(configs)}] Treinando {key} ({len(features)} features)...")
        start = time.perf_counter()
        train_sample, train_total, train_sample_rows = deterministic_training_sample(
            con,
            input_path(train, matrix_name),
            features,
            args.training_sample_size,
        )
        estimator = IsolationForest(
            n_estimators=args.n_estimators,
            max_samples=min(args.max_samples_tree, train_sample_rows),
            contamination="auto",
            max_features=1.0,
            bootstrap=False,
            n_jobs=args.threads,
            random_state=args.random_state,
            verbose=0,
        )
        estimator.fit(train_sample)
        fit_seconds = time.perf_counter() - start
        model_path = model_dir / f"{key}.joblib"
        joblib.dump(
            {
                "estimator": estimator,
                "features": features,
                "configuration": config,
                "experiment_signature": experiment_signature,
                "preprocessing_signature": preprocessing["preprocessing_signature"],
            },
            model_path,
            compress=3,
        )
        training_rows.append(
            {
                "configuration": key,
                "matrix": matrix_name,
                "family": config["family"],
                "features": len(features),
                "training_rows_total": train_total,
                "training_rows_sample": train_sample_rows,
                "sample_share": train_sample_rows / train_total,
                "n_estimators": args.n_estimators,
                "max_samples_tree": min(args.max_samples_tree, train_sample_rows),
                "fit_seconds": fit_seconds,
                "model_path": str(model_path.resolve()),
                "model_size_bytes": model_path.stat().st_size,
            }
        )
        del train_sample

        print("  Calibrando quantis exclusivamente na validação...")
        _, validation_scores = score_batches(
            estimator,
            input_path(validation, matrix_name),
            features,
            args.batch_size,
            None,
            None,
            args.threads,
        )
        thresholds = {
            "q990": float(np.quantile(validation_scores, 0.990)),
            "q995": float(np.quantile(validation_scores, 0.995)),
            "q999": float(np.quantile(validation_scores, 0.999)),
        }
        threshold_rows.append(
            {
                "configuration": key,
                "calibration_split": validation["key"],
                "validation_rows": len(validation_scores),
                **thresholds,
            }
        )
        del validation_scores

        for split_index, split in enumerate(splits, start=1):
            split_key = str(split["key"])
            print(f"  [{split_index}/{len(splits)}] Pontuando {split_key}...")
            output_path = score_root / key / f"{int(split['order']):02d}_{split_key}.parquet"
            if output_path.exists() and not args.force:
                output_path.unlink()
            _, scores = score_batches(
                estimator,
                input_path(split, matrix_name),
                features,
                args.batch_size,
                output_path,
                thresholds,
                args.threads,
            )
            stats = score_statistics(scores)
            flags = {name: scores >= threshold for name, threshold in thresholds.items()}
            statistics_rows.append(
                {
                    "split_order": split["order"],
                    "split": split_key,
                    "role": split["role"],
                    "configuration": key,
                    "matrix": matrix_name,
                    "rows": len(scores),
                    **stats,
                }
            )
            rate_rows.append(
                {
                    "split_order": split["order"],
                    "split": split_key,
                    "role": split["role"],
                    "configuration": key,
                    "matrix": matrix_name,
                    "rows": len(scores),
                    "anomalies_q990": int(np.sum(flags["q990"])),
                    "anomaly_rate_q990": float(np.mean(flags["q990"])),
                    "anomalies_q995": int(np.sum(flags["q995"])),
                    "anomaly_rate_q995": float(np.mean(flags["q995"])),
                    "anomalies_q999": int(np.sum(flags["q999"])),
                    "anomaly_rate_q999": float(np.mean(flags["q999"])),
                    "score_path": str(output_path.resolve()),
                    "score_size_bytes": output_path.stat().st_size,
                }
            )
            type4 = con.execute(
                f"""
                SELECT
                    m.is_type_4,
                    COUNT(*)::BIGINT AS rows,
                    SUM(CASE WHEN s.is_anomaly_q995 THEN 1 ELSE 0 END)::BIGINT AS anomalies,
                    AVG(CASE WHEN s.is_anomaly_q995 THEN 1.0 ELSE 0.0 END)::DOUBLE AS rate,
                    AVG(s.anomaly_score)::DOUBLE AS mean_score,
                    QUANTILE_CONT(s.anomaly_score, 0.50)::DOUBLE AS median_score,
                    QUANTILE_CONT(s.anomaly_score, 0.995)::DOUBLE AS q995_score
                FROM read_parquet({sql_quote(output_path.as_posix())}) AS s
                INNER JOIN read_parquet({sql_quote(metadata_path(split).as_posix())}) AS m USING (row_id)
                GROUP BY m.is_type_4 ORDER BY m.is_type_4
                """
            ).fetchall()
            for is_type_4, rows, anomalies, rate, mean_score, median_score, q995_score in type4:
                type4_rows.append(
                    {
                        "split_order": split["order"],
                        "split": split_key,
                        "role": split["role"],
                        "configuration": key,
                        "is_type_4": int(is_type_4),
                        "rows": int(rows),
                        "anomalies_q995": int(anomalies),
                        "anomaly_rate_q995": float(rate),
                        "score_mean": float(mean_score),
                        "score_median": float(median_score),
                        "score_q995": float(q995_score),
                    }
                )
            del scores, flags

    write_csv(args.output_dir / "02_resumo_treinamento.csv", training_rows)
    write_csv(args.output_dir / "03_limiares_validacao.csv", threshold_rows)
    write_csv(args.output_dir / "04_estatisticas_escores.csv", statistics_rows)
    write_csv(args.output_dir / "05_taxas_anomalias_recortes.csv", rate_rows)
    write_csv(args.output_dir / "06_taxas_anomalias_tipo4.csv", type4_rows)

    validation_sets: dict[str, set[int]] = {}
    for config in configs:
        key = str(config["key"])
        validation_score_path = score_root / key / f"{int(validation['order']):02d}_{validation['key']}.parquet"
        values = con.execute(
            f"SELECT row_id FROM read_parquet({sql_quote(validation_score_path.as_posix())}) WHERE is_anomaly_q995"
        ).fetchall()
        validation_sets[key] = {int(row[0]) for row in values}
    agreement_rows: list[dict[str, object]] = []
    for left, right in combinations(validation_sets, 2):
        intersection = len(validation_sets[left] & validation_sets[right])
        union = len(validation_sets[left] | validation_sets[right])
        agreement_rows.append(
            {
                "calibration_split": validation["key"],
                "configuration_left": left,
                "configuration_right": right,
                "anomalies_left": len(validation_sets[left]),
                "anomalies_right": len(validation_sets[right]),
                "intersection": intersection,
                "union": union,
                "jaccard": intersection / union if union else 1.0,
            }
        )
    write_csv(args.output_dir / "07_concordancia_validacao.csv", agreement_rows)
    create_rate_plot(
        args.output_dir / "08_taxas_anomalias_recortes.pdf",
        args.output_dir / "08_taxas_anomalias_recortes.png",
        rate_rows,
        splits,
        configs,
    )

    methodology = f"""# Metodologia do baseline Isolation Forest

## Ajuste e amostragem

Os cinco modelos foram ajustados exclusivamente em `{train['key']}`. A amostra
determinística usa `hash(row_id)` para cobrir todo o período de treinamento sem
ordenar ou concentrar as observações no início da janela. Cada floresta possui
{args.n_estimators} árvores e cada árvore recebe no máximo
{args.max_samples_tree} transações.

O escore registrado é `-score_samples`: valores maiores indicam maior anomalia.
O parâmetro `contamination=auto` não define a classificação final.

## Configurações

- `a_completa`, `b_completa` e `c_completa` usam os contratos A, B e C;
- `b_sem_taxas_absolutas` e `c_sem_taxas_absolutas` removem a família de taxas
  absolutas, preservando `fee_cap_utilization` e `gas_price_rank_block`;
- as configurações foram congeladas antes da pontuação do teste.

## Limiares

Os limiares Q99, Q99,5 e Q99,9 são calculados exclusivamente em
`{validation['key']}`. O Q99,5 é o ponto principal e os demais formam uma análise
de sensibilidade. Por construção, aproximadamente 0,5% da validação é marcada;
isso não representa uma estimativa da prevalência real de fraude.

## Interpretação

Sem rótulos, taxa de sinalização e concordância entre modelos não medem
precisão ou recall. A escolha de uma configuração deve combinar estabilidade
temporal, inspeção dos casos, comparação externa e, posteriormente, concordância
com o Autoencoder. O teste não pode ser usado para reajustar o modelo.
"""
    (args.output_dir / "09_metodologia_isolation_forest.md").write_text(methodology, encoding="utf-8")

    expected_outputs = [
        "01_configuracoes_experimento.json",
        "02_resumo_treinamento.csv",
        "03_limiares_validacao.csv",
        "04_estatisticas_escores.csv",
        "05_taxas_anomalias_recortes.csv",
        "06_taxas_anomalias_tipo4.csv",
        "07_concordancia_validacao.csv",
        "08_taxas_anomalias_recortes.pdf",
        "08_taxas_anomalias_recortes.png",
        "09_metodologia_isolation_forest.md",
        "10_manifest_isolation_forest.json",
    ]
    manifest = {
        "generated_at": now_iso(),
        "experiment_signature": experiment_signature,
        "preprocessing_signature": preprocessing["preprocessing_signature"],
        "source_split_manifest": str(args.split_manifest.resolve()),
        "source_preprocessing_manifest": str(args.preprocessing_manifest.resolve()),
        "fit_split": train["key"],
        "threshold_split": validation["key"],
        "configurations": [config["key"] for config in configs],
        "outputs": expected_outputs,
        "models_directory": str(model_dir.resolve()),
        "scores_directory": str(score_root.resolve()),
    }
    (args.output_dir / "10_manifest_isolation_forest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    con.close()
    print(f"\nIsolation Forest concluído: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
