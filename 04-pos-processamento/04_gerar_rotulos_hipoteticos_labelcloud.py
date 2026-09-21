# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Gera rótulos binários hipotéticos a partir do Label Cloud histórico."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


ASSUMPTION = (
    "Experimento hipotético: endereços presentes na categoria MEV Bot do Etherscan "
    "Label Cloud são classe positiva (1), e todos os demais endereços são tratados "
    "como classe negativa (0)."
)


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    here = Path(__file__).resolve().parent
    src_root = here.parent
    parser = argparse.ArgumentParser(description="Cria ground truth hipotético baseado no Label Cloud.")
    parser.add_argument(
        "--tags",
        type=Path,
        default=here / "resultados-labelcloud" / "01_labelcloud_historico_normalizado.parquet",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=here.parent
        / "03-machine-learning"
        / "01-features"
        / "resultados-splits"
        / "00_manifest_splits_temporais.json",
    )
    parser.add_argument("--output-dir", type=Path, default=here / "resultados-labelcloud")
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="12GB")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    tags = args.tags.resolve()
    split_manifest = args.split_manifest.resolve()
    if not tags.is_file():
        parser.error(f"Tags não encontradas: {tags}")
    if not split_manifest.is_file():
        parser.error(f"Manifesto de recortes não encontrado: {split_manifest}")
    splits = json.loads(split_manifest.read_text(encoding="utf-8"))["splits"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output / "05_rotulos_binarios_hipoteticos_labelcloud.parquet"

    con = duckdb.connect()
    con.execute(f"SET memory_limit={sql_quote(args.memory_limit)}")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory={sql_quote(args.temp_dir.resolve())}")
    con.execute(
        f"""
        CREATE TEMP VIEW mev_bots AS
        SELECT DISTINCT lower(address) AS address
        FROM read_parquet({sql_quote(tags.as_posix())})
        WHERE mev_category = 'mev_bot'
        """
    )

    parts: list[str] = []
    metadata_by_split: dict[str, str] = {}
    for split in splits:
        split_key = str(split["key"])
        metadata = Path(split["output_directory"]) / "05_metadados.parquet"
        if not metadata.is_file():
            raise RuntimeError(f"Metadados não encontrados para {split_key}: {metadata}")
        metadata_by_split[split_key] = str(metadata)
        parts.append(
            f"""
            SELECT
                {sql_quote(split_key)}::VARCHAR AS split,
                m.row_id,
                CASE WHEN b.address IS NOT NULL THEN 1 ELSE 0 END::UTINYINT AS label
            FROM read_parquet({sql_quote(metadata.as_posix())}) m
            LEFT JOIN mev_bots b ON lower(m.from_address) = b.address
            """
        )
    union_sql = " UNION ALL ".join(parts)
    con.execute(
        f"COPY ({union_sql}) TO {sql_quote(labels_path.as_posix())} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
    )

    rows = con.execute(
        f"""
        SELECT split, count(*) AS rows, sum(label)::BIGINT AS positives,
               count(*) - sum(label)::BIGINT AS negatives,
               100.0 * avg(label) AS prevalence_percent
        FROM read_parquet({sql_quote(labels_path.as_posix())})
        GROUP BY split ORDER BY split
        """
    ).fetchall()
    summary_path = output / "06_resumo_rotulos_hipoteticos.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "rows", "positives", "negatives", "prevalence_percent"])
        writer.writerows(rows)

    validation = con.execute(
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT split || ':' || cast(row_id AS VARCHAR)) AS unique_keys,
               min(label), max(label), count(*) FILTER (WHERE label NOT IN (0, 1))
        FROM read_parquet({sql_quote(labels_path.as_posix())})
        """
    ).fetchone()
    con.close()
    if validation[0] != validation[1] or validation[2:] != (0, 1, 0):
        raise RuntimeError(f"Falha na validação dos rótulos: {validation}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "hypothetical_binary_ground_truth",
        "assumption": ASSUMPTION,
        "positive_definition": "from_address belongs to historical Etherscan Label Cloud category mev_bot",
        "negative_definition": "from_address absent from the historical mev_bot category (hypothetical assumption)",
        "join_key": "lower(metadata.from_address) = lower(tags.address)",
        "tags_file": str(tags),
        "tags_sha256": sha256(tags),
        "split_manifest": str(split_manifest),
        "metadata_by_split": metadata_by_split,
        "labels_file": str(labels_path),
        "labels_sha256": sha256(labels_path),
        "total_rows": validation[0],
        "unique_split_row_id": validation[1],
        "warning": "The class-0 assignment is an experimental assumption, not an observed negative label.",
    }
    manifest_path = output / "07_manifest_rotulos_hipoteticos.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"labels": str(labels_path), "rows": validation[0], "summary": str(summary_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
