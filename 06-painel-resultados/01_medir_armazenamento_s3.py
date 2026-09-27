"""Compara tamanhos dos Parquets originais no S3 com os locais, sem abri-los.

O resultado mede remoção de input + eventual mudança de compressão/row groups.
O prefixo S3 contém partições date=AAAA-MM-DD; cada ano é filtrado pelo nome
da partição e comparado ao diretório local em dados.root_template.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "ano", "arquivos_s3", "arquivos_locais", "arquivos_pareados",
    "somente_s3", "somente_local", "cobertura_identica",
    "s3_gib", "local_gib", "diferenca_pareada_gib", "reducao_pareada_pct",
    "s3_total_gib", "local_total_gib",
]


def list_s3(uri: str, year: int, *, region: str | None, no_sign: bool) -> dict[str, int]:
    match = re.fullmatch(r"s3://([^/]+)(?:/(.*))?", uri.rstrip("/"))
    if not match:
        raise ValueError(f"URI S3 inválida: {uri}")
    bucket, prefix = match.group(1), (match.group(2) or "").strip("/")
    base_prefix = f"{prefix}/" if prefix else ""
    list_prefix = f"{base_prefix}date={year}-"
    command = ["aws", "s3api", "list-objects-v2", "--bucket", bucket,
               "--prefix", list_prefix, "--output", "json"]
    if region:
        command.extend(["--region", region])
    if no_sign:
        command.append("--no-sign-request")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Falha ao listar {uri}: {result.stderr.strip()[:500]}")
    document = json.loads(result.stdout)
    if document.get("IsTruncated"):
        raise RuntimeError("Listagem S3 truncada; revise a paginação antes de medir.")
    files: dict[str, int] = {}
    for item in document.get("Contents", []):
        key = item.get("Key", "")
        if not key.lower().endswith(".parquet"):
            continue
        relative = key[len(base_prefix):] if key.startswith(list_prefix) else ""
        if not relative or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise RuntimeError(f"Chave S3 fora do prefixo esperado: {key}")
        files[relative] = int(item["Size"])
    return files


def list_local(root: Path) -> dict[str, int]:
    if not root.is_dir():
        raise FileNotFoundError(f"Diretório local inexistente: {root}")
    return {path.relative_to(root).as_posix(): path.stat().st_size
            for path in root.rglob("*.parquet") if path.is_file()}


def gib(value: int) -> float:
    return round(value / 2**30, 3)


def measure(year: int, remote: dict[str, int], local: dict[str, int]) -> dict:
    common = remote.keys() & local.keys()
    s3_paired = sum(remote[key] for key in common)
    local_paired = sum(local[key] for key in common)
    difference = s3_paired - local_paired
    exact = remote.keys() == local.keys()
    return {
        "ano": year, "arquivos_s3": len(remote), "arquivos_locais": len(local),
        "arquivos_pareados": len(common), "somente_s3": len(remote.keys() - local.keys()),
        "somente_local": len(local.keys() - remote.keys()), "cobertura_identica": exact,
        "s3_gib": gib(s3_paired), "local_gib": gib(local_paired),
        "diferenca_pareada_gib": gib(difference),
        "reducao_pareada_pct": round(difference / s3_paired * 100, 2) if s3_paired else None,
        "s3_total_gib": gib(sum(remote.values())), "local_total_gib": gib(sum(local.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3-prefix", default="s3://aws-public-blockchain/v1.0/eth/transactions/",
                        help="Prefixo S3 que contém as partições date=AAAA-MM-DD")
    parser.add_argument("--config", type=Path, default=ROOT / "configuracao/pipeline.json")
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--region")
    parser.add_argument("--no-sign-request", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "06-painel-resultados/resultados-armazenamento")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root_template = config["dados"]["root_template"]
    if "{year}" not in root_template:
        parser.error("dados.root_template deve conter {year}.")
    rows = []
    for year in args.years:
        local_root = Path(root_template.format(year=year))
        row = measure(year, list_s3(args.s3_prefix, year, region=args.region,
                                    no_sign=args.no_sign_request), list_local(local_root))
        rows.append(row)
        print(f"{year}: {row['arquivos_pareados']} arquivos pareados; "
              f"S3 {row['s3_gib']} GiB; local {row['local_gib']} GiB; "
              f"diferença {row['diferenca_pareada_gib']} GiB; "
              f"cobertura idêntica = {row['cobertura_identica']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "01_comparacao_s3_local.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "02_manifest_medicao.json").write_text(
        json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "s3_prefix": args.s3_prefix, "local_root_template": root_template,
                    "method": "S3 object Size versus local file stat; no Parquet content read",
                    "interpretation": "remocao de input mais recompressao ZSTD; nao isola o efeito da coluna",
                    "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
