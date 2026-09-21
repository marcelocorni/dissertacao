# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2"]
# ///

"""Audita cobertura e compatibilidade das sete saídas C# candidate-first."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def q(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csharp-root",
        type=Path,
        default=here.parent / "05-rotulador-front-running-csharp" / "resultados-v3" / "sampled" / "independent_sampled",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=here / "resultados" / "auditoria-csharp"
    )
    args = parser.parse_args()
    root = args.csharp_root.resolve()
    output = args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f"Raiz C# não encontrada: {root}")
    output.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    records = []
    issues = []
    required = ["01_detection_events.parquet", "02_transaction_labels.parquet", "04_manifest_run.json", "05_enrichment_queue.parquet"]
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            issues.append({"split": directory.name, "severity": "fatal", "issue": f"Ausentes: {', '.join(missing)}"})
            continue
        manifest = json.loads((directory / "04_manifest_run.json").read_text(encoding="utf-8"))
        events = directory / "01_detection_events.parquet"
        labels = directory / "02_transaction_labels.parquet"
        queue = directory / "05_enrichment_queue.parquet"
        event_stats = connection.execute(
            f"SELECT count(*), count(distinct detection_event_id), sum(detection_event_id IS NULL)::BIGINT FROM read_parquet({q(events)})"
        ).fetchone()
        queue_stats = connection.execute(
            f"SELECT count(*), count(distinct lower(hash)), count(distinct block_number), sum(hash IS NULL)::BIGINT FROM read_parquet({q(queue)})"
        ).fetchone()
        label_stats = connection.execute(
            f"SELECT count(*), sum(label=1)::BIGINT, sum(label=0)::BIGINT, count(distinct lower(hash)) FROM read_parquet({q(labels)})"
        ).fetchone()
        detectors = dict(connection.execute(
            f"SELECT detector, count(*) FROM read_parquet({q(events)}) GROUP BY 1 ORDER BY 1"
        ).fetchall())
        record = {
            "split": directory.name,
            "project_version": manifest.get("ProjectVersion"),
            "input_mode": manifest.get("InputMode"),
            "configuration": manifest.get("Configuration"),
            "source_dates": manifest.get("SourceDates"),
            "events": event_stats[0],
            "unique_events": event_stats[1],
            "insertion_events": detectors.get("insertion", 0),
            "displacement_events": detectors.get("displacement", 0),
            "suppression_events": detectors.get("suppression", 0),
            "queue_transactions": queue_stats[0],
            "queue_unique_hashes": queue_stats[1],
            "queue_blocks": queue_stats[2],
            "label_rows": label_stats[0],
            "positive_labels_before_semantics": label_stats[1],
            "negative_labels": label_stats[2],
            "status": "valid",
        }
        if manifest.get("ProjectVersion") != "3.0.0-candidate-first":
            issues.append({"split": directory.name, "severity": "fatal", "issue": "Versão C# incompatível"})
        if event_stats[0] != event_stats[1] or event_stats[2]:
            issues.append({"split": directory.name, "severity": "fatal", "issue": "IDs de evento inválidos ou duplicados"})
        if queue_stats[0] != queue_stats[1] or queue_stats[3]:
            issues.append({"split": directory.name, "severity": "fatal", "issue": "Fila possui hashes inválidos ou duplicados"})
        if label_stats[1] != 0 or not manifest.get("RequiresSemanticValidation"):
            issues.append({"split": directory.name, "severity": "fatal", "issue": "Ground truth promovido antes da validação semântica"})
        records.append(record)
    connection.close()
    if not records:
        parser.error(f"Nenhuma saída C# válida encontrada em: {root}")
    columns = list(records[0])
    with (output / "01_resumo_saidas_csharp.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(records)
    totals = {key: sum(int(row[key]) for row in records) for key in ["events", "queue_transactions", "queue_blocks", "label_rows"]}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "expected_splits": 7,
        "found_splits": len(records),
        "compatible_version": "3.0.0-candidate-first",
        "totals": totals,
        "issues": issues,
        "rerun_csharp_required": bool(issues) or len(records) != 7,
    }
    (output / "02_manifest_auditoria_csharp.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Auditoria C#: {len(records)}/7 janelas; eventos={totals['events']}; fila={totals['queue_transactions']}; blocos={totals['queue_blocks']}; problemas={len(issues)}")
    return 1 if manifest["rerun_csharp_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
