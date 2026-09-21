# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Normaliza o CSV histórico do Etherscan Label Cloud fornecido pelo pesquisador."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb


EXPECTED_ROWS = 3_913
EXPECTED_SHA256 = "fc4652e4b9f2c9ee3faebd85e1e9dbf0dff973ddf33937c6a5713f73574042c4"
ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# O CSV original não possui a categoria. Os intervalos reproduzem a ordem de
# concatenação da extração histórica e são registrados como reconstrução
# posicional, nunca como informação observada diretamente em uma coluna.
CATEGORY_RANGES = (
    (1, 163, "mev_builder", "infrastructure_builder"),
    (164, 3_911, "mev_bot", "weak_positive_mev_bot"),
    (3_912, 3_912, "mev_relay", "infrastructure_relay"),
    (3_913, 3_913, "mev_protection", "protection_context"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def category_for(position: int) -> tuple[str, str]:
    for start, end, category, role in CATEGORY_RANGES:
        if start <= position <= end:
            return category, role
    raise ValueError(f"Posição fora dos intervalos conhecidos: {position}")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Normaliza a base histórica do Etherscan Label Cloud sem alterar o arquivo-fonte."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=here / "resultados-labelcloud")
    parser.add_argument(
        "--snapshot-date",
        default="",
        help="Data original da coleta, se conhecida (AAAA-MM-DD). Vazio significa desconhecida.",
    )
    parser.add_argument(
        "--accept-other-file",
        action="store_true",
        help="Aceita outro SHA-256, mantendo as demais validações estruturais.",
    )
    args = parser.parse_args()

    source = args.input_csv.resolve()
    if not source.is_file():
        parser.error(f"CSV não encontrado: {source}")
    if args.snapshot_date:
        datetime.strptime(args.snapshot_date, "%Y-%m-%d")

    source_hash = sha256(source)
    if source_hash != EXPECTED_SHA256 and not args.accept_other_file:
        raise RuntimeError(
            "O SHA-256 difere do arquivo auditado. Revise a fonte ou use "
            "--accept-other-file conscientemente."
        )

    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["transaction_from", "label"]:
            raise RuntimeError(f"Cabeçalho inesperado: {reader.fieldnames}")
        raw_rows = list(reader)

    if len(raw_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Esperadas {EXPECTED_ROWS} linhas; encontradas {len(raw_rows)}.")

    invalid_addresses: list[dict[str, object]] = []
    normalized: list[dict[str, object]] = []
    for position, row in enumerate(raw_rows, start=1):
        original_address = row["transaction_from"].strip()
        if not ADDRESS_RE.fullmatch(original_address):
            invalid_addresses.append({"source_position": position, "address": original_address})
        category, role = category_for(position)
        normalized.append(
            {
                "address": original_address.lower(),
                "address_original": original_address,
                "name_tag": row["label"].strip(),
                "mev_category": category,
                "category_role": role,
                "is_mev_related_address": True,
                "weak_positive_mev_bot": 1 if category == "mev_bot" else "",
                "category_assignment_basis": "positional_reconstruction",
                "source_position": position,
                "source": "Etherscan Label Cloud - historical researcher export",
                "source_snapshot_date": args.snapshot_date,
            }
        )

    if invalid_addresses:
        raise RuntimeError(f"Há {len(invalid_addresses)} endereços Ethereum inválidos.")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "01_labelcloud_historico_normalizado.csv"
    parquet_path = output / "01_labelcloud_historico_normalizado.parquet"
    summary_path = output / "02_resumo_labelcloud_historico.csv"
    issues_path = output / "03_pendencias_name_tag.csv"
    manifest_path = output / "04_manifest_labelcloud_historico.json"

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)

    summary_rows: list[dict[str, object]] = []
    counts = Counter(row["mev_category"] for row in normalized)
    for category in (item[2] for item in CATEGORY_RANGES):
        selected = [row for row in normalized if row["mev_category"] == category]
        summary_rows.append(
            {
                "mev_category": category,
                "rows": counts[category],
                "unique_addresses": len({row["address"] for row in selected}),
                "blank_name_tags": sum(not row["name_tag"] for row in selected),
                "assignment_basis": "positional_reconstruction",
            }
        )
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    blank_rows = [row for row in normalized if not row["name_tag"]]
    with issues_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["source_position", "address", "mev_category", "issue"]
        )
        writer.writeheader()
        for row in blank_rows:
            writer.writerow(
                {
                    "source_position": row["source_position"],
                    "address": row["address"],
                    "mev_category": row["mev_category"],
                    "issue": "name_tag_blank_but_category_membership_preserved",
                }
            )

    con = duckdb.connect()
    con.execute(
        "COPY (SELECT * FROM read_csv_auto("
        f"{sql_quote(csv_path.as_posix())}, header=true, all_varchar=true)) "
        f"TO {sql_quote(parquet_path.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()

    exact_duplicates = len(raw_rows) - len(
        {(row["transaction_from"].lower(), row["label"]) for row in raw_rows}
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source),
        "source_sha256": source_hash,
        "source_snapshot_date": args.snapshot_date or None,
        "source_snapshot_date_note": (
            "not encoded in the supplied CSV; filename timestamp was not treated as collection date"
            if not args.snapshot_date
            else None
        ),
        "raw_rows": len(raw_rows),
        "unique_addresses": len({row["address"] for row in normalized}),
        "exact_duplicate_source_rows": exact_duplicates,
        "blank_name_tags": len(blank_rows),
        "category_reconstruction": [
            {"start_position": a, "end_position": b, "mev_category": c, "category_role": d}
            for a, b, c, d in CATEGORY_RANGES
        ],
        "methodological_warning": (
            "Unknown addresses are unlabeled, not negative. Only mev_bot membership is a weak positive; "
            "builder, relay and protection are contextual infrastructure categories."
        ),
        "outputs": {
            "normalized_csv": str(csv_path),
            "normalized_parquet": str(parquet_path),
            "summary_csv": str(summary_path),
            "issues_csv": str(issues_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **{k: manifest[k] for k in ("raw_rows", "unique_addresses", "exact_duplicate_source_rows", "blank_name_tags")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
