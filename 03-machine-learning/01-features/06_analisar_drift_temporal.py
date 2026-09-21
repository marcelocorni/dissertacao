# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
#   "matplotlib>=3.10,<4",
# ]
# ///

"""Mede drift temporal das features sem usar validação ou teste no ajuste.

A distribuição do treino pós-Dencun é a única referência. O programa
calcula PSI, uma aproximação determinística do KS sobre percentis do treino,
mudanças robustas de localização/escala e extrapolação. O tipo 4 permanece
fora das matrizes, mas é analisado separadamente por meio dos metadados.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb


MATRIX_INPUTS = {
    "A": "12_matriz_A_preprocessada.parquet",
    "B": "13_matriz_B_preprocessada.parquet",
    "C": "14_matriz_C_preprocessada.parquet",
}
METADATA_FILE = "05_metadados.parquet"
PSI_EPSILON = 1e-6
PSI_MODERATE = 0.10
PSI_HIGH = 0.25
KS_MODERATE = 0.10
KS_HIGH = 0.20


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = here.parents[1]
    parser = argparse.ArgumentParser(
        description="Compara todos os recortes com o treino e documenta drift temporal."
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=here / "resultados-splits" / "00_manifest_splits_temporais.json",
    )
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        default=here / "resultados-preprocessamento" / "05_manifest_preprocessamento.json",
    )
    parser.add_argument(
        "--transformation-dictionary",
        type=Path,
        default=here / "resultados-preprocessamento" / "02_dicionario_transformacoes.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=here / "resultados-drift"
    )
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument(
        "--threads", type=int, default=max(1, min(8, os.cpu_count() or 4))
    )
    parser.add_argument(
        "--ks-grid-size",
        type=int,
        default=101,
        help="Número de quantis do treino usados na aproximação do KS.",
    )
    args = parser.parse_args()
    for path in (
        args.split_manifest,
        args.preprocessing_manifest,
        args.transformation_dictionary,
    ):
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")
    if args.threads < 1:
        parser.error("--threads deve ser maior ou igual a 1")
    if args.ks_grid_size < 21:
        parser.error("--ks-grid-size deve ser maior ou igual a 21")
    return args


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def configure(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir.as_posix())}")
    con.execute("SET preserve_insertion_order = false")
    return con


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix_path(split: dict[str, object], matrix: str = "C") -> Path:
    return Path(str(split["output_directory"])) / MATRIX_INPUTS[matrix]


def metadata_path(split: dict[str, object]) -> Path:
    return Path(str(split["output_directory"])) / METADATA_FILE


def validate_inputs(
    splits: list[dict[str, object]], preprocessing: dict[str, object]
) -> None:
    expected_signature = str(preprocessing["preprocessing_signature"])
    for split in splits:
        directory = Path(str(split["output_directory"]))
        local_manifest = directory / "15_manifest_preprocessamento.json"
        if not local_manifest.is_file():
            raise FileNotFoundError(f"Manifesto de pré-processamento ausente: {local_manifest}")
        document = load_json(local_manifest)
        if str(document.get("preprocessing_signature")) != expected_signature:
            raise RuntimeError(f"Assinatura incompatível em {split['key']}.")
        for name in ("A", "B", "C"):
            path = matrix_path(split, name)
            if not path.is_file():
                raise FileNotFoundError(f"Matriz ausente: {path}")
        if not metadata_path(split).is_file():
            raise FileNotFoundError(f"Metadados ausentes: {metadata_path(split)}")


def read_transformations(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            row["feature"]: row["transformation"]
            for row in csv.DictReader(stream)
        }


def relation(path: Path, where: str | None = None, metadata: Path | None = None) -> str:
    base = f"read_parquet({sql_quote(path.as_posix())}) AS x"
    if metadata is not None:
        base += (
            f" INNER JOIN read_parquet({sql_quote(metadata.as_posix())}) AS m "
            "USING (row_id)"
        )
    if where:
        base += f" WHERE {where}"
    return base


def describe_feature(
    con: duckdb.DuckDBPyConnection, source: str, feature: str
) -> dict[str, float | int]:
    f = f"x.{ident(feature)}"
    row = con.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT,
            MIN({f})::DOUBLE,
            QUANTILE_CONT({f}, 0.25)::DOUBLE,
            QUANTILE_CONT({f}, 0.50)::DOUBLE,
            AVG({f})::DOUBLE,
            QUANTILE_CONT({f}, 0.75)::DOUBLE,
            MAX({f})::DOUBLE,
            STDDEV_SAMP({f})::DOUBLE,
            AVG(CASE WHEN {f} = 0 THEN 1.0 ELSE 0.0 END)::DOUBLE
        FROM {source}
        """
    ).fetchone()
    if row is None or row[0] == 0:
        raise RuntimeError(f"Relação vazia ao analisar {feature}.")
    names = ("rows", "min", "q1", "median", "mean", "q3", "max", "std", "zero_share")
    return dict(zip(names, row))


def quantile_grid(
    con: duckdb.DuckDBPyConnection,
    source: str,
    feature: str,
    probabilities: Iterable[float],
) -> list[float]:
    probs = list(probabilities)
    values = con.execute(
        f"SELECT QUANTILE_CONT(x.{ident(feature)}, {probs}) FROM {source}"
    ).fetchone()[0]
    return [float(value) for value in values if value is not None and math.isfinite(value)]


def cdf_at(
    con: duckdb.DuckDBPyConnection,
    source: str,
    feature: str,
    thresholds: list[float],
) -> list[float]:
    if not thresholds:
        return []
    f = f"x.{ident(feature)}"
    expressions = [
        f"AVG(CASE WHEN {f} <= {value:.17g} THEN 1.0 ELSE 0.0 END)"
        for value in thresholds
    ]
    return [float(value) for value in con.execute("SELECT " + ",".join(expressions) + f" FROM {source}").fetchone()]


def bucket_shares(
    con: duckdb.DuckDBPyConnection,
    source: str,
    feature: str,
    boundaries: list[float],
) -> list[float]:
    f = f"x.{ident(feature)}"
    clauses = [f"WHEN {f} <= {value:.17g} THEN {index}" for index, value in enumerate(boundaries)]
    bucket = "CASE " + " ".join(clauses) + f" ELSE {len(boundaries)} END"
    rows = con.execute(
        f"SELECT {bucket} AS bucket, COUNT(*)::BIGINT FROM {source} GROUP BY bucket"
    ).fetchall()
    total = sum(int(count) for _, count in rows)
    counts = [0] * (len(boundaries) + 1)
    for index, count in rows:
        counts[int(index)] = int(count)
    return [count / total for count in counts]


def psi(expected: list[float], actual: list[float]) -> float:
    return sum(
        (max(a, PSI_EPSILON) - max(e, PSI_EPSILON))
        * math.log(max(a, PSI_EPSILON) / max(e, PSI_EPSILON))
        for e, a in zip(expected, actual)
    )


def severity(value: float, moderate: float, high: float) -> str:
    if value >= high:
        return "alto"
    if value >= moderate:
        return "moderado"
    return "baixo"


def combined_severity(psi_value: float, ks_value: float) -> str:
    levels = {"baixo": 0, "moderado": 1, "alto": 2}
    labels = {value: key for key, value in levels.items()}
    return labels[
        max(
            levels[severity(psi_value, PSI_MODERATE, PSI_HIGH)],
            levels[severity(ks_value, KS_MODERATE, KS_HIGH)],
        )
    ]


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator != 0 else None


def drift_row(
    con: duckdb.DuckDBPyConnection,
    source: str,
    feature: str,
    reference: dict[str, object],
) -> dict[str, object]:
    current = describe_feature(con, source, feature)
    actual_cdf = cdf_at(con, source, feature, reference["ks_thresholds"])
    ks_value = max(
        abs(actual - expected)
        for actual, expected in zip(actual_cdf, reference["ks_cdf"])
    )
    actual_buckets = bucket_shares(
        con, source, feature, reference["psi_boundaries"]
    )
    psi_value = psi(reference["psi_shares"], actual_buckets)
    f = f"x.{ident(feature)}"
    low = float(reference["profile"]["min"])
    high = float(reference["profile"]["max"])
    q1 = float(reference["profile"]["q1"])
    q3 = float(reference["profile"]["q3"])
    iqr = q3 - q1
    fence_low, fence_high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outside = con.execute(
        f"""
        SELECT
          AVG(CASE WHEN {f} < {low:.17g} OR {f} > {high:.17g} THEN 1.0 ELSE 0.0 END),
          AVG(CASE WHEN {f} < {fence_low:.17g} OR {f} > {fence_high:.17g} THEN 1.0 ELSE 0.0 END)
        FROM {source}
        """
    ).fetchone()
    current_iqr = float(current["q3"]) - float(current["q1"])
    reference_iqr = q3 - q1
    return {
        "rows": int(current["rows"]),
        "psi": psi_value,
        "psi_level": severity(psi_value, PSI_MODERATE, PSI_HIGH),
        "ks_approx": ks_value,
        "ks_level": severity(ks_value, KS_MODERATE, KS_HIGH),
        "drift_level": combined_severity(psi_value, ks_value),
        "reference_median": float(reference["profile"]["median"]),
        "current_median": float(current["median"]),
        "median_shift": float(current["median"]) - float(reference["profile"]["median"]),
        "reference_iqr": reference_iqr,
        "current_iqr": current_iqr,
        "iqr_ratio": safe_ratio(current_iqr, reference_iqr),
        "reference_zero_share": float(reference["profile"]["zero_share"]),
        "current_zero_share": float(current["zero_share"]),
        "zero_share_delta": float(current["zero_share"]) - float(reference["profile"]["zero_share"]),
        "outside_training_range_share": float(outside[0]),
        "outside_training_robust_fence_share": float(outside[1]),
        "current_min": float(current["min"]),
        "current_max": float(current["max"]),
        "current_mean": float(current["mean"]),
        "current_std": float(current["std"] or 0.0),
        "psi_bins": len(actual_buckets),
        "ks_grid_points": len(reference["ks_thresholds"]),
    }


def make_heatmap(
    path_pdf: Path,
    path_png: Path,
    rows: list[dict[str, object]],
    splits: list[dict[str, object]],
    features: list[str],
    metric: str,
    title: str,
    vmax: float,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    split_keys = [str(split["key"]) for split in splits]
    lookup = {(str(row["feature"]), str(row["split"])): float(row[metric]) for row in rows}
    data = np.array([[lookup[(feature, key)] for key in split_keys] for feature in features])
    width = max(12.0, len(split_keys) * 1.65)
    height = max(9.0, len(features) * 0.38)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(split_keys)), [key.replace("_", "\n") for key in split_keys], fontsize=8)
    ax.set_yticks(range(len(features)), features, fontsize=8)
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("Recorte temporal")
    ax.set_ylabel("Feature")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label(metric)
    for y in range(len(features)):
        for x in range(len(split_keys)):
            value = data[y, x]
            if value >= 0.10 or (metric == "psi" and value >= 0.05):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if value > vmax * 0.55 else "black")
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    split_document = load_json(args.split_manifest)
    preprocessing = load_json(args.preprocessing_manifest)
    splits = list(split_document["splits"])
    train_matches = [split for split in splits if split["role"] == "train"]
    if len(train_matches) != 1:
        raise RuntimeError("Deve existir exatamente um recorte com papel train.")
    train = train_matches[0]
    validate_inputs(splits, preprocessing)
    transformations = read_transformations(args.transformation_dictionary)
    matrices = {name: list(values) for name, values in preprocessing["matrices"].items()}
    features = matrices["C"]
    memberships = {
        feature: "/".join(name for name in ("A", "B", "C") if feature in matrices[name])
        for feature in features
    }

    settings = {
        "preprocessing_signature": preprocessing["preprocessing_signature"],
        "reference_split": train["key"],
        "psi_quantiles": 10,
        "ks_grid_size": args.ks_grid_size,
        "psi_thresholds": [PSI_MODERATE, PSI_HIGH],
        "ks_thresholds": [KS_MODERATE, KS_HIGH],
    }
    analysis_signature = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")
    ).hexdigest()

    con = configure(args)
    train_source = relation(matrix_path(train))
    print("Construindo a referência exclusivamente no treino...")
    references: dict[str, dict[str, object]] = {}
    ks_probabilities = [index / (args.ks_grid_size - 1) for index in range(args.ks_grid_size)]
    psi_probabilities = [index / 10 for index in range(1, 10)]
    for index, feature in enumerate(features, start=1):
        profile = describe_feature(con, train_source, feature)
        ks_thresholds = sorted(set(quantile_grid(con, train_source, feature, ks_probabilities)))
        psi_boundaries = sorted(set(quantile_grid(con, train_source, feature, psi_probabilities)))
        references[feature] = {
            "profile": profile,
            "ks_thresholds": ks_thresholds,
            "ks_cdf": cdf_at(con, train_source, feature, ks_thresholds),
            "psi_boundaries": psi_boundaries,
            "psi_shares": bucket_shares(con, train_source, feature, psi_boundaries),
        }
        print(f"  [{index:02d}/{len(features)}] {feature}")

    detailed: list[dict[str, object]] = []
    type4_drift: list[dict[str, object]] = []
    transaction_types: list[dict[str, object]] = []
    for split in splits:
        split_key = str(split["key"])
        print(f"Analisando {split_key}...")
        source = relation(matrix_path(split))
        for feature in features:
            metrics = drift_row(con, source, feature, references[feature])
            detailed.append(
                {
                    "split_order": split["order"],
                    "split": split_key,
                    "role": split["role"],
                    "start_date": split["start_date"],
                    "end_date": split["end_date"],
                    "feature": feature,
                    "matrices": memberships[feature],
                    "transformation": transformations.get(feature, "unknown"),
                    **metrics,
                }
            )

        metadata = metadata_path(split)
        type_rows = con.execute(
            f"""
            SELECT transaction_type, COUNT(*)::BIGINT
            FROM read_parquet({sql_quote(metadata.as_posix())})
            GROUP BY transaction_type ORDER BY transaction_type
            """
        ).fetchall()
        type_total = sum(int(count) for _, count in type_rows)
        for transaction_type, count in type_rows:
            transaction_types.append(
                {
                    "split_order": split["order"],
                    "split": split_key,
                    "role": split["role"],
                    "transaction_type": int(transaction_type),
                    "rows": int(count),
                    "share": int(count) / type_total,
                }
            )
        type4_rows = sum(int(count) for tx_type, count in type_rows if int(tx_type) == 4)
        if type4_rows:
            type4_source = relation(
                matrix_path(split), where="m.is_type_4 = 1", metadata=metadata
            )
            for feature in features:
                metrics = drift_row(con, type4_source, feature, references[feature])
                type4_drift.append(
                    {
                        "split_order": split["order"],
                        "split": split_key,
                        "role": split["role"],
                        "feature": feature,
                        "type4_rows": type4_rows,
                        "type4_share_split": type4_rows / type_total,
                        **metrics,
                    }
                )

    detail_fields = list(detailed[0].keys())
    write_csv(args.output_dir / "01_drift_features_recortes.csv", detailed, detail_fields)

    summary_rows: list[dict[str, object]] = []
    for split in splits:
        selected = [row for row in detailed if row["split"] == split["key"]]
        summary_rows.append(
            {
                "split_order": split["order"],
                "split": split["key"],
                "role": split["role"],
                "rows": selected[0]["rows"],
                "features": len(selected),
                "psi_mean": sum(float(row["psi"]) for row in selected) / len(selected),
                "psi_max": max(float(row["psi"]) for row in selected),
                "ks_mean": sum(float(row["ks_approx"]) for row in selected) / len(selected),
                "ks_max": max(float(row["ks_approx"]) for row in selected),
                "features_drift_moderado": sum(row["drift_level"] == "moderado" for row in selected),
                "features_drift_alto": sum(row["drift_level"] == "alto" for row in selected),
            }
        )
    write_csv(args.output_dir / "02_resumo_drift_recortes.csv", summary_rows, list(summary_rows[0].keys()))

    ranking_rows: list[dict[str, object]] = []
    evaluation = [row for row in detailed if row["role"] != "train"]
    for feature in features:
        selected = [row for row in evaluation if row["feature"] == feature]
        ranking_rows.append(
            {
                "feature": feature,
                "matrices": memberships[feature],
                "transformation": transformations.get(feature, "unknown"),
                "psi_mean_non_train": sum(float(row["psi"]) for row in selected) / len(selected),
                "psi_max_non_train": max(float(row["psi"]) for row in selected),
                "ks_mean_non_train": sum(float(row["ks_approx"]) for row in selected) / len(selected),
                "ks_max_non_train": max(float(row["ks_approx"]) for row in selected),
                "splits_drift_alto": sum(row["drift_level"] == "alto" for row in selected),
                "splits_drift_moderado": sum(row["drift_level"] == "moderado" for row in selected),
            }
        )
    ranking_rows.sort(key=lambda row: (float(row["psi_max_non_train"]), float(row["ks_max_non_train"])), reverse=True)
    write_csv(args.output_dir / "03_ranking_features_drift.csv", ranking_rows, list(ranking_rows[0].keys()))

    matrix_rows: list[dict[str, object]] = []
    for split in splits:
        for matrix_name, matrix_features in matrices.items():
            selected = [
                row for row in detailed
                if row["split"] == split["key"] and row["feature"] in matrix_features
            ]
            matrix_rows.append(
                {
                    "split_order": split["order"],
                    "split": split["key"],
                    "role": split["role"],
                    "matrix": matrix_name,
                    "features": len(selected),
                    "psi_mean": sum(float(row["psi"]) for row in selected) / len(selected),
                    "psi_max": max(float(row["psi"]) for row in selected),
                    "ks_mean": sum(float(row["ks_approx"]) for row in selected) / len(selected),
                    "ks_max": max(float(row["ks_approx"]) for row in selected),
                    "features_drift_alto": sum(row["drift_level"] == "alto" for row in selected),
                }
            )
    write_csv(args.output_dir / "04_drift_por_matriz.csv", matrix_rows, list(matrix_rows[0].keys()))
    if type4_drift:
        write_csv(args.output_dir / "05_drift_transacoes_tipo_4.csv", type4_drift, list(type4_drift[0].keys()))
    write_csv(
        args.output_dir / "06_distribuicao_tipos_transacao.csv",
        transaction_types,
        list(transaction_types[0].keys()),
    )

    make_heatmap(
        args.output_dir / "07_heatmap_psi.pdf",
        args.output_dir / "07_heatmap_psi.png",
        detailed,
        splits,
        features,
        "psi",
        "Population Stability Index por feature e recorte",
        vmax=max(0.50, min(2.0, max(float(row["psi"]) for row in detailed))),
    )
    make_heatmap(
        args.output_dir / "08_heatmap_ks_aproximado.pdf",
        args.output_dir / "08_heatmap_ks_aproximado.png",
        detailed,
        splits,
        features,
        "ks_approx",
        "Distância KS aproximada por feature e recorte",
        vmax=max(0.30, max(float(row["ks_approx"]) for row in detailed)),
    )

    methodology = f"""# Metodologia da análise de drift temporal

## Referência imutável

Todas as comparações usam exclusivamente `{train['key']}` ({train['start_date']} a
{train['end_date']}) como distribuição de referência. Validação, teste e janelas
de mudança de protocolo não ajustam limites, bins ou transformações.

## Métricas

- **PSI:** bins definidos pelos decis do treino. Limiares descritivos: abaixo de
  0,10, drift baixo; de 0,10 a 0,25, moderado; a partir de 0,25, alto.
- **KS aproximado:** maior diferença entre CDFs avaliada em
  {args.ks_grid_size} quantis do treino. Não é calculado p-valor, pois amostras de
  milhões de linhas tornariam diferenças pequenas estatisticamente significativas.
  Limiares descritivos: abaixo de 0,10, baixo; de 0,10 a 0,20, moderado; a partir
  de 0,20, alto.
- **Mediana e IQR:** registram mudanças robustas de localização e dispersão.
- **Zeros:** registra mudanças em features esparsas e zeros estruturais.
- **Extrapolação:** proporção fora do mínimo/máximo e das cercas robustas do treino.

Esses limites são heurísticos de diagnóstico, não critérios automáticos para
eliminar features. Drift pode representar mudança de protocolo ou comportamento
relevante para detecção de anomalias.

## Transações tipo 4

O tipo 4 não entra como indicador nas matrizes treinadas em 2024, pois não existia
no período de ajuste. Ele é recuperado pelos metadados e comparado separadamente
com a distribuição do treino. Isso permite verificar se os escores futuros estão
associados à novidade de protocolo sem fornecer essa informação diretamente ao modelo.

## Uso no pipeline

O teste final continua intocado para seleção de features, hiperparâmetros e limiar.
Este relatório serve para interpretar estabilidade e planejar análises estratificadas;
não autoriza otimização orientada pelos resultados do teste.
"""
    (args.output_dir / "09_metodologia_drift.md").write_text(methodology, encoding="utf-8")

    output_names = sorted(path.name for path in args.output_dir.iterdir() if path.is_file())
    if "10_manifest_drift.json" not in output_names:
        output_names.append("10_manifest_drift.json")
        output_names.sort()
    manifest = {
        "generated_at": now_iso(),
        "analysis_signature": analysis_signature,
        "source_split_manifest": str(args.split_manifest.resolve()),
        "source_preprocessing_manifest": str(args.preprocessing_manifest.resolve()),
        "preprocessing_signature": preprocessing["preprocessing_signature"],
        "reference_split": train["key"],
        "features": features,
        "settings": settings,
        "outputs": output_names,
    }
    (args.output_dir / "10_manifest_drift.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    con.close()
    print(f"\nAnálise concluída: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
