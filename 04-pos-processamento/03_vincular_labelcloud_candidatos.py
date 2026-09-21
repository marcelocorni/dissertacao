# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Vincula tags históricas aos candidatos AE/IF por endereço de origem."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def main() -> int:
    here = Path(__file__).resolve().parent
    src_root = here.parent
    parser = argparse.ArgumentParser(
        description="Anota candidatos AE/IF com categorias externas do Etherscan Label Cloud."
    )
    parser.add_argument(
        "--tags",
        type=Path,
        default=here / "resultados-labelcloud" / "01_labelcloud_historico_normalizado.parquet",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=here.parent / "05-analise-resultados" / "resultados" / "candidatos",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=here / "resultados-labelcloud" / "vinculos-candidatos"
    )
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    tags = args.tags.resolve()
    candidates_root = args.candidates_dir.resolve()
    if not tags.is_file():
        parser.error(f"Parquet de tags não encontrado: {tags}")
    if not candidates_root.is_dir():
        parser.error(f"Diretório de candidatos não encontrado: {candidates_root}")
    candidate_files = sorted(candidates_root.rglob("*.parquet"))
    if not candidate_files:
        parser.error("Nenhum Parquet de candidatos encontrado.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit={sql_quote(args.memory_limit)}")
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET temp_directory={sql_quote(args.temp_dir.resolve())}")
    con.execute(
        f"""
        CREATE TEMP VIEW tags_by_address AS
        SELECT
            lower(address) AS address,
            string_agg(DISTINCT mev_category, '|' ORDER BY mev_category) AS mev_categories,
            string_agg(DISTINCT nullif(name_tag, ''), '|' ORDER BY nullif(name_tag, '')) AS mev_name_tags,
            max(CASE WHEN mev_category = 'mev_bot' THEN 1 ELSE 0 END) = 1 AS is_known_mev_bot
        FROM read_parquet({sql_quote(tags.as_posix())})
        GROUP BY lower(address)
        """
    )

    summaries: list[dict[str, object]] = []
    for index, source in enumerate(candidate_files, start=1):
        relative = source.relative_to(candidates_root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sql = sql_quote(source.as_posix())
        destination_sql = sql_quote(destination.as_posix())
        con.execute(
            f"""
            COPY (
                SELECT
                    c.*,
                    t.address IS NOT NULL AS from_matches_labelcloud,
                    t.mev_categories AS from_mev_categories,
                    t.mev_name_tags AS from_mev_name_tags,
                    CASE WHEN t.is_known_mev_bot THEN 1 ELSE NULL END::UTINYINT AS weak_label_mev_bot,
                    CASE
                        WHEN t.is_known_mev_bot THEN 'weak_positive_mev_bot'
                        WHEN t.address IS NOT NULL THEN 'mev_context_only'
                        ELSE 'unlabeled'
                    END AS external_label_status
                FROM read_parquet({source_sql}) c
                LEFT JOIN tags_by_address t ON lower(c.from_address) = t.address
            ) TO {destination_sql} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        stats = con.execute(
            f"""
            SELECT
                count(*) AS candidate_rows,
                count(DISTINCT lower(from_address)) AS unique_from_addresses,
                count(*) FILTER (WHERE from_matches_labelcloud) AS matched_rows,
                count(DISTINCT lower(from_address)) FILTER (WHERE from_matches_labelcloud) AS matched_addresses,
                count(*) FILTER (WHERE weak_label_mev_bot = 1) AS weak_positive_rows,
                count(DISTINCT lower(from_address)) FILTER (WHERE weak_label_mev_bot = 1) AS weak_positive_addresses
            FROM read_parquet({destination_sql})
            """
        ).fetchone()
        summaries.append(
            {
                "configuration": relative.parent.as_posix(),
                "candidate_file": relative.name,
                "candidate_rows": stats[0],
                "unique_from_addresses": stats[1],
                "matched_rows": stats[2],
                "matched_addresses": stats[3],
                "weak_positive_mev_bot_rows": stats[4],
                "weak_positive_mev_bot_addresses": stats[5],
                "output_file": str(destination),
            }
        )
        print(f"[{index}/{len(candidate_files)}] {relative}: {stats[2]} vínculos; {stats[4]} positivos fracos")

    summary_path = output / "00_resumo_vinculos.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    configuration_rows: list[dict[str, object]] = []
    for configuration in sorted({str(row["configuration"]) for row in summaries}):
        paths = sorted((output / configuration).glob("*.parquet"))
        path_list = "[" + ",".join(sql_quote(path.as_posix()) for path in paths) + "]"
        stats = con.execute(
            f"""
            SELECT
                count(*) AS candidate_rows,
                count(DISTINCT hash) AS unique_transactions,
                count(DISTINCT lower(from_address)) AS unique_from_addresses,
                count(*) FILTER (WHERE from_matches_labelcloud) AS matched_rows,
                count(DISTINCT hash) FILTER (WHERE weak_label_mev_bot = 1) AS weak_positive_transactions,
                count(DISTINCT lower(from_address)) FILTER (WHERE weak_label_mev_bot = 1) AS weak_positive_addresses
            FROM read_parquet({path_list}, union_by_name=true)
            """
        ).fetchone()
        configuration_rows.append(
            {
                "configuration": configuration,
                "candidate_rows": stats[0],
                "unique_transactions": stats[1],
                "unique_from_addresses": stats[2],
                "matched_rows": stats[3],
                "weak_positive_mev_bot_transactions": stats[4],
                "weak_positive_mev_bot_addresses": stats[5],
                "weak_positive_rate_percent": round(100 * stats[4] / stats[1], 6) if stats[1] else 0,
            }
        )
    configuration_summary_path = output / "01_resumo_por_configuracao.csv"
    with configuration_summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(configuration_rows[0]))
        writer.writeheader()
        writer.writerows(configuration_rows)

    all_outputs = "[" + ",".join(sql_quote(path.as_posix()) for path in sorted(output.glob("*/*.parquet"))) + "]"
    global_stats = con.execute(
        f"""
        SELECT
            count(DISTINCT hash) AS unique_candidate_transactions,
            count(DISTINCT hash) FILTER (WHERE weak_label_mev_bot = 1) AS unique_weak_positive_transactions,
            count(DISTINCT lower(from_address)) FILTER (WHERE weak_label_mev_bot = 1) AS unique_weak_positive_addresses
        FROM read_parquet({all_outputs}, union_by_name=true)
        """
    ).fetchone()

    totals = {
        "candidate_rows": sum(int(row["candidate_rows"]) for row in summaries),
        "matched_rows": sum(int(row["matched_rows"]) for row in summaries),
        "weak_positive_mev_bot_rows": sum(int(row["weak_positive_mev_bot_rows"]) for row in summaries),
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tags_file": str(tags),
        "candidates_dir": str(candidates_root),
        "candidate_files": len(candidate_files),
        "join_key": "lower(candidates.from_address) = lower(tags.address)",
        "totals_across_candidate_files": totals,
        "deduplicated_across_configurations": {
            "unique_candidate_transactions": global_stats[0],
            "unique_weak_positive_mev_bot_transactions": global_stats[1],
            "unique_weak_positive_mev_bot_addresses": global_stats[2],
        },
        "label_semantics": {
            "1": "address listed in the historical MEV Bot category; weak positive",
            "null": "unlabeled; must not be interpreted as negative",
            "context": "Builder, Relay and Protection matches are contextual, not class 0",
        },
        "warning": "Files overlap across model configurations; totals must not be interpreted as unique transactions globally.",
        "summary_csv": str(summary_path),
        "configuration_summary_csv": str(configuration_summary_path),
    }
    (output / "00_manifest_vinculos.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    con.close()
    print(json.dumps({"output_dir": str(output), **totals}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
