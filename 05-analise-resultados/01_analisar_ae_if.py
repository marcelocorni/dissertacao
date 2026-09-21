# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2", "matplotlib>=3.10,<4", "numpy>=2.1,<3"]
# ///

"""Consolida AE e IF, gera candidatos rastreáveis e figuras de escores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = here.parent
    ml_root = here.parent / "03-machine-learning"
    parser = argparse.ArgumentParser(description="Análise integrada dos escores AE e IF.")
    parser.add_argument("--split-manifest", type=Path, default=ml_root / "01-features" / "resultados-splits" / "00_manifest_splits_temporais.json")
    parser.add_argument("--ae-results-dir", type=Path, default=ml_root / "02-ae" / "resultados")
    parser.add_argument("--if-results-dir", type=Path, default=ml_root / "03-if" / "resultados")
    parser.add_argument("--output-dir", type=Path, default=here / "resultados")
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--ecdf-sample-size", type=int, default=100_000)
    parser.add_argument("--top-per-split", type=int, default=100)
    args = parser.parse_args()
    for path in (args.split_manifest, args.ae_results_dir / "13_manifest_autoencoder.json", args.if_results_dir / "10_manifest_isolation_forest.json"):
        if not path.is_file(): parser.error(f"Arquivo não encontrado: {path}")
    return args


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows: return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def configure(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.output_dir.mkdir(parents=True, exist_ok=True); args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}"); con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir.as_posix())}"); con.execute("SET preserve_insertion_order = false")
    return con


def score_path(root: Path, config: str, split: dict[str, object]) -> Path:
    return root / "escores" / config / f"{int(split['order']):02d}_{split['key']}.parquet"


def metadata_path(split: dict[str, object]) -> Path:
    return Path(str(split["output_directory"])) / "05_metadados.parquet"


def make_candidates(con: duckdb.DuckDBPyConnection, ae_path: Path, if_path: Path, metadata: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): output.unlink()
    con.execute(f"""
        COPY (
          WITH paired AS (
            SELECT a.row_id, a.reconstruction_error AS ae_score,
              i.anomaly_score AS if_score, a.is_anomaly_q995 AS ae_anomaly,
              i.is_anomaly_q995 AS if_anomaly
            FROM read_parquet({sql_quote(ae_path.as_posix())}) a
            INNER JOIN read_parquet({sql_quote(if_path.as_posix())}) i USING (row_id)
          ), ranked AS (
            SELECT *, PERCENT_RANK() OVER (ORDER BY ae_score) AS ae_percentile,
              PERCENT_RANK() OVER (ORDER BY if_score) AS if_percentile FROM paired
          )
          SELECT r.row_id, m.hash, m.block_hash, m.block_number, m.block_timestamp,
            m.date, m.from_address, m.to_address, m.transaction_index,
            m.transaction_type, m.is_type_4, r.ae_score, r.if_score,
            r.ae_percentile, r.if_percentile,
            (r.ae_percentile + r.if_percentile) / 2.0 AS consensus_score,
            r.ae_anomaly, r.if_anomaly,
            CASE WHEN r.ae_anomaly AND r.if_anomaly THEN 'ambos'
              WHEN r.ae_anomaly THEN 'somente_ae' ELSE 'somente_if' END AS detection_group
          FROM ranked r INNER JOIN read_parquet({sql_quote(metadata.as_posix())}) m USING (row_id)
          WHERE r.ae_anomaly OR r.if_anomaly ORDER BY consensus_score DESC
        ) TO {sql_quote(output.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def plot_if_quantiles(path_pdf: Path, path_png: Path, stats: list[dict[str, str]], configs: list[str], splits: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    keys = [str(split["key"]) for split in splits]; lookup = {(row["configuration"], row["split"]): row for row in stats}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, config in zip(axes.flat, configs):
        for field, label in (("score_median", "mediana"), ("score_q950", "Q95"), ("score_q990", "Q99"), ("score_q995", "Q99,5"), ("score_q999", "Q99,9")):
            axis.plot(range(len(keys)), [float(lookup[(config, key)][field]) for key in keys], marker="o", label=label)
        axis.set_title(config); axis.grid(alpha=0.25); axis.set_ylabel("Escore IF"); axis.set_xlabel("Recorte")
        axis.set_xticks(range(len(keys)), [str(index + 1) for index in range(len(keys))]); axis.legend(fontsize=7, ncol=2)
    axes.flat[-1].axis("off"); fig.suptitle("Isolation Forest: evolução dos quantis de escore por recorte", fontsize=14)
    fig.savefig(path_pdf, bbox_inches="tight"); fig.savefig(path_png, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_if_survival(con: duckdb.DuckDBPyConnection, path_pdf: Path, path_png: Path, if_root: Path, configs: list[str], splits: list[dict[str, object]], sample_size: int) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for axis, config in zip(axes.flat, configs):
        for split in splits:
            path = score_path(if_root, config, split); total = int(split["sample_rows"]); target = min(total, sample_size)
            values = con.execute(f"SELECT anomaly_score FROM read_parquet({sql_quote(path.as_posix())}) WHERE hash(row_id) % {total} < {target} ORDER BY anomaly_score").fetchnumpy()["anomaly_score"]
            if len(values) == 0: continue
            survival = (len(values) - np.arange(len(values))) / len(values)
            axis.plot(values, survival, linewidth=1.1, label=str(split["order"]))
        axis.set_yscale("log"); axis.set_ylim(1e-4, 1.05); axis.set_title(config)
        axis.set_xlabel("Escore IF"); axis.set_ylabel("P(escore ≥ x)"); axis.grid(alpha=0.25); axis.legend(title="recorte", fontsize=7, ncol=2)
    axes.flat[-1].axis("off"); fig.suptitle("Isolation Forest: curvas de sobrevivência dos escores", fontsize=14)
    fig.savefig(path_pdf, bbox_inches="tight"); fig.savefig(path_png, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_rate_heatmaps(path_pdf: Path, path_png: Path, rows: list[dict[str, object]], configs: list[str], splits: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    keys = [str(split["key"]) for split in splits]; fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for axis, method, field in zip(axes, ("AE", "IF"), ("ae_rate_q995", "if_rate_q995")):
        lookup = {(str(row["configuration"]), str(row["split"])): float(row[field]) * 100 for row in rows}
        data = np.array([[lookup[(config, key)] for key in keys] for config in configs])
        image = axis.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(1.0, float(np.quantile(data, 0.95))))
        axis.set_title(f"{method}: taxa sinalizada Q99,5 (%)"); axis.set_yticks(range(len(configs)), configs, fontsize=8)
        axis.set_xticks(range(len(keys)), [str(i + 1) for i in range(len(keys))])
        for y in range(len(configs)):
            for x in range(len(keys)): axis.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.supxlabel("Recortes 1 a 7 em ordem cronológica")
    fig.savefig(path_pdf, bbox_inches="tight"); fig.savefig(path_png, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_agreement(path_pdf: Path, path_png: Path, rows: list[dict[str, object]], configs: list[str], splits: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    keys = [str(split["key"]) for split in splits]; lookup = {(str(row["configuration"]), str(row["split"])): float(row["jaccard"]) for row in rows}
    data = np.array([[lookup[(config, key)] for key in keys] for config in configs])
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True); image = axis.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axis.set_title("Concordância AE–IF no conjunto sinalizado por Q99,5"); axis.set_yticks(range(len(configs)), configs)
    axis.set_xticks(range(len(keys)), [str(i + 1) for i in range(len(keys))]); axis.set_xlabel("Recortes 1 a 7 em ordem cronológica")
    for y in range(len(configs)):
        for x in range(len(keys)): axis.text(x, y, f"{data[y, x]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Jaccard"); fig.savefig(path_pdf, bbox_inches="tight"); fig.savefig(path_png, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    args = parse_args(); con = configure(args)
    splits = list(load_json(args.split_manifest)["splits"])
    ae_manifest = load_json(args.ae_results_dir / "13_manifest_autoencoder.json"); if_manifest = load_json(args.if_results_dir / "10_manifest_isolation_forest.json")
    configs = list(ae_manifest["configurations"])
    if configs != list(if_manifest["configurations"]): raise RuntimeError("As configurações de AE e IF não coincidem.")
    ae_rates = read_csv(args.ae_results_dir / "06_taxas_anomalias_recortes.csv"); if_rates = read_csv(args.if_results_dir / "05_taxas_anomalias_recortes.csv"); cross = read_csv(args.ae_results_dir / "09_concordancia_ae_if.csv")
    ae_lookup = {(row["configuration"], row["split"]): row for row in ae_rates}; if_lookup = {(row["configuration"], row["split"]): row for row in if_rates}; cross_lookup = {(row["configuration"], row["split"]): row for row in cross}
    integrated: list[dict[str, object]] = []; candidate_summary: list[dict[str, object]] = []; top_rows: list[dict[str, object]] = []; candidate_root = args.output_dir / "candidatos"
    for config_index, config in enumerate(configs, start=1):
        for split in splits:
            key = str(split["key"]); pair = cross_lookup[(config, key)]; ae = ae_lookup[(config, key)]; iff = if_lookup[(config, key)]
            integrated.append({"split_order": split["order"], "split": key, "role": split["role"], "configuration": config, "rows": ae["rows"], "ae_anomalies_q995": ae["anomalies_q995"], "ae_rate_q995": ae["anomaly_rate_q995"], "if_anomalies_q995": iff["anomalies_q995"], "if_rate_q995": iff["anomaly_rate_q995"], "intersection": pair["intersection"], "union": pair["union"], "jaccard": pair["jaccard"]})
            print(f"[{config_index}/{len(configs)}] Gerando candidatos {config} / {key}...")
            candidate_path = candidate_root / config / f"{int(split['order']):02d}_{key}.parquet"
            make_candidates(con, score_path(args.ae_results_dir, config, split), score_path(args.if_results_dir, config, split), metadata_path(split), candidate_path)
            counts = con.execute(f"SELECT COUNT(*)::BIGINT, SUM(ae_anomaly AND if_anomaly)::BIGINT, SUM(ae_anomaly AND NOT if_anomaly)::BIGINT, SUM(if_anomaly AND NOT ae_anomaly)::BIGINT, SUM(is_type_4)::BIGINT FROM read_parquet({sql_quote(candidate_path.as_posix())})").fetchone()
            candidate_summary.append({"split_order": split["order"], "split": key, "role": split["role"], "configuration": config, "union_candidates": int(counts[0]), "both": int(counts[1]), "only_ae": int(counts[2]), "only_if": int(counts[3]), "type_4_candidates": int(counts[4]), "candidate_path": str(candidate_path.resolve()), "size_bytes": candidate_path.stat().st_size})
            result = con.execute(f"SELECT * FROM read_parquet({sql_quote(candidate_path.as_posix())}) ORDER BY consensus_score DESC LIMIT {args.top_per_split}")
            columns = [item[0] for item in result.description]
            for row in result.fetchall(): top_rows.append({"split_order": split["order"], "split": key, "role": split["role"], "configuration": config, **dict(zip(columns, row))})
    write_csv(args.output_dir / "01_resumo_integrado_ae_if.csv", integrated); write_csv(args.output_dir / "02_resumo_candidatos.csv", candidate_summary); write_csv(args.output_dir / "03_top_candidatos.csv", top_rows)
    plot_if_quantiles(args.output_dir / "04_if_quantis_escores.pdf", args.output_dir / "04_if_quantis_escores.png", read_csv(args.if_results_dir / "04_estatisticas_escores.csv"), configs, splits)
    plot_if_survival(con, args.output_dir / "05_if_curvas_sobrevivencia.pdf", args.output_dir / "05_if_curvas_sobrevivencia.png", args.if_results_dir, configs, splits, args.ecdf_sample_size)
    plot_rate_heatmaps(args.output_dir / "06_heatmaps_taxas_ae_if.pdf", args.output_dir / "06_heatmaps_taxas_ae_if.png", integrated, configs, splits)
    plot_agreement(args.output_dir / "07_heatmap_concordancia_ae_if.pdf", args.output_dir / "07_heatmap_concordancia_ae_if.png", integrated, configs, splits)
    methodology = """# Análise integrada AE e Isolation Forest

O conjunto de candidatos usa a união dos casos acima do Q99,5 de cada método.
`ambos` representa consenso; `somente_ae` e `somente_if` preservam resultados
complementares. Os percentis são calculados dentro de cada recorte e configuração,
permitindo combinar escalas diferentes sem misturar os valores brutos.

Os Parquets em `candidatos/` contêm hashes, bloco, data, endereços, tipo da
transação, escores, percentis e grupo de detecção. Esses campos servem apenas
para rastreabilidade e pós-processamento; não participaram do treinamento.

As curvas de sobrevivência do IF exibem a cauda dos escores em escala logarítmica.
AUC-ROC e PR-AUC exigem rótulos externos e são tratadas pelo segundo script.
"""
    (args.output_dir / "08_metodologia_analise_integrada.md").write_text(methodology, encoding="utf-8")
    signature = hashlib.sha256(json.dumps({"ae": ae_manifest["experiment_signature"], "if": if_manifest["experiment_signature"], "top": args.top_per_split, "ecdf_sample": args.ecdf_sample_size}, sort_keys=True).encode()).hexdigest()
    outputs = sorted({path.name for path in args.output_dir.iterdir() if path.is_file()} | {"09_manifest_analise_integrada.json"})
    manifest = {"generated_at": datetime.now().astimezone().isoformat(), "analysis_signature": signature, "ae_experiment_signature": ae_manifest["experiment_signature"], "if_experiment_signature": if_manifest["experiment_signature"], "configurations": configs, "splits": [split["key"] for split in splits], "candidate_directory": str(candidate_root.resolve()), "outputs": outputs}
    (args.output_dir / "09_manifest_analise_integrada.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close(); print(f"\nAnálise integrada concluída: {args.output_dir.resolve()}"); return 0


if __name__ == "__main__": raise SystemExit(main())
