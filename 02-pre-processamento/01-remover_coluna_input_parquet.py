# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "duckdb>=1.4.3,<2",
# ]
# ///

"""Reescreve Parquets da AWS sem a coluna ``input``.

Cada arquivo e processado isoladamente. O script cria um arquivo temporario no
mesmo diretorio, valida a quantidade de linhas e o esquema e, somente entao,
substitui o original. Arquivos que ja nao possuem ``input`` sao ignorados, de
modo que uma execucao interrompida pode ser retomada com o mesmo comando.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configuracao import ConfiguracaoPipelineError, carregar_root_template


GIB = 1024**3


@dataclass
class Totals:
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_before: int = 0
    bytes_after: int = 0


def sql_literal(value: str | Path) -> str:
    """Retorna um literal SQL seguro para um caminho controlado localmente."""
    normalized = str(value).replace("\\", "/").replace("'", "''")
    return f"'{normalized}'"


def human_gib(size: int) -> str:
    return f"{size / GIB:,.2f} GiB"


def describe_parquet(connection: duckdb.DuckDBPyConnection, path: Path) -> list[tuple[str, str]]:
    rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet("
        f"{sql_literal(path)}, hive_partitioning = false)"
    ).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def parquet_row_count(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    value = connection.execute(
        "SELECT COALESCE(SUM(num_rows), 0)::BIGINT "
        f"FROM parquet_file_metadata({sql_literal(path)})"
    ).fetchone()[0]
    return int(value)


def rewrite_file(
    connection: duckdb.DuckDBPyConnection,
    source: Path,
    compression: str,
    row_group_size: int,
    reserve_bytes: int,
    dry_run: bool,
) -> tuple[str, int, int]:
    source_schema = describe_parquet(connection, source)
    source_columns = [name for name, _ in source_schema]

    if "input" not in source_columns:
        return "skipped", source.stat().st_size, source.stat().st_size

    size_before = source.stat().st_size
    free_bytes = shutil.disk_usage(source.parent).free
    required_bytes = size_before + reserve_bytes
    if free_bytes < required_bytes:
        raise RuntimeError(
            "espaco livre insuficiente para a reescrita temporaria: "
            f"livre={human_gib(free_bytes)}, "
            f"minimo_estimado={human_gib(required_bytes)}"
        )

    if dry_run:
        return "converted", size_before, 0

    temporary = source.with_name(source.name + ".without-input.partial")
    temporary.unlink(missing_ok=True)
    expected_schema = [(name, kind) for name, kind in source_schema if name != "input"]

    try:
        original_rows = parquet_row_count(connection, source)

        connection.execute(
            "COPY ("
            "SELECT * EXCLUDE (input) "
            f"FROM read_parquet({sql_literal(source)}, hive_partitioning = false)"
            ") "
            f"TO {sql_literal(temporary)} "
            "(FORMAT PARQUET, "
            f"COMPRESSION {compression.upper()}, "
            f"ROW_GROUP_SIZE {row_group_size})"
        )

        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("o arquivo temporario nao foi criado corretamente")

        rewritten_rows = parquet_row_count(connection, temporary)
        if rewritten_rows != original_rows:
            raise RuntimeError(
                "a quantidade de linhas mudou: "
                f"original={original_rows:,}, novo={rewritten_rows:,}"
            )

        rewritten_schema = describe_parquet(connection, temporary)
        if rewritten_schema != expected_schema:
            raise RuntimeError(
                "o esquema do novo arquivo difere do esperado: "
                f"esperado={expected_schema!r}, obtido={rewritten_schema!r}"
            )

        size_after = temporary.stat().st_size
        os.replace(temporary, source)
        return "converted", size_before, size_after
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def discover_files(root_template: str, years: list[int], months: list[int]) -> list[Path]:
    files: list[Path] = []
    for year in years:
        root = Path(root_template.format(year=year))
        if not root.is_dir():
            print(f"AVISO: diretorio inexistente, ignorando: {root}", file=sys.stderr)
            continue

        for month in months:
            pattern = f"date={year}-{month:02d}-*"
            for day_directory in sorted(root.glob(pattern)):
                if day_directory.is_dir():
                    files.extend(sorted(day_directory.glob("*.parquet")))
    return files


def parse_args() -> argparse.Namespace:
    default_temp_dir = SRC_ROOT / ".tmp" / "duckdb"
    parser = argparse.ArgumentParser(
        description=(
            "Remove a coluna input dos Parquets, validando cada arquivo antes "
            "de substituir o original."
        )
    )
    try:
        config_path, configured_root = carregar_root_template(
            SRC_ROOT / "configuracao" / "pipeline.json"
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
        help=(
            "Substitui dados.root_template da configuração somente nesta execução."
        ),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2024, 2025],
        help="Anos que serao processados. Padrao: 2024 2025",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        type=int,
        choices=range(1, 13),
        default=list(range(1, 13)),
        metavar="MES",
        help="Meses de 1 a 12. Se omitido, processa todos.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=default_temp_dir,
        help=(
            "Diretorio de temporarios do DuckDB. Padrao: "
            "src/.tmp/duckdb. Use outro disco se faltar espaco."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        default="8GB",
        help="Limite de memoria do DuckDB. Padrao: 8GB",
    )
    parser.add_argument(
        "--compression",
        choices=("zstd", "snappy"),
        default="zstd",
        help="Compressao dos novos arquivos. Padrao: zstd",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=250_000,
        help="Linhas por row group. Padrao: 250000",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=2.0,
        help="Margem livre adicional exigida no disco de origem. Padrao: 2 GiB",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista o trabalho sem modificar arquivos.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = discover_files(args.root_template, args.years, args.months)
    if not files:
        print("Nenhum arquivo Parquet encontrado para os filtros informados.")
        return 1

    print(f"Arquivos encontrados: {len(files)}")
    print(f"Modo: {'SIMULACAO' if args.dry_run else 'REESCRITA DESTRUTIVA VALIDADA'}")
    print("A coluna input sera removida; os demais campos serao preservados.")

    args.temp_dir.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(":memory:")
    connection.execute(f"SET memory_limit = {sql_literal(args.memory_limit)}")
    connection.execute("SET preserve_insertion_order = true")
    connection.execute(f"SET temp_directory = {sql_literal(args.temp_dir)}")

    totals = Totals()
    started = time.monotonic()
    reserve_bytes = max(0, int(args.reserve_gib * GIB))

    try:
        for index, source in enumerate(files, start=1):
            prefix = f"[{index}/{len(files)}]"
            try:
                status, size_before, size_after = rewrite_file(
                    connection=connection,
                    source=source,
                    compression=args.compression,
                    row_group_size=args.row_group_size,
                    reserve_bytes=reserve_bytes,
                    dry_run=args.dry_run,
                )

                if status == "skipped":
                    totals.skipped += 1
                    print(f"{prefix} JA PROCESSADO: {source}")
                    continue

                totals.converted += 1
                totals.bytes_before += size_before
                totals.bytes_after += size_after
                if args.dry_run:
                    print(f"{prefix} PROCESSARIA: {source} ({human_gib(size_before)})")
                else:
                    reduction = size_before - size_after
                    print(
                        f"{prefix} OK: {source} | "
                        f"{human_gib(size_before)} -> {human_gib(size_after)} | "
                        f"liberado {human_gib(reduction)}"
                    )
            except KeyboardInterrupt:
                print("\nInterrompido pelo usuario. Execute novamente para retomar.")
                return 130
            except Exception as error:
                totals.failed += 1
                print(f"{prefix} ERRO: {source}: {error}", file=sys.stderr)
                print("Execucao interrompida; os arquivos anteriores permanecem validos.")
                return 2
    finally:
        connection.close()

    elapsed = time.monotonic() - started
    print("\nResumo")
    print(f"Convertidos: {totals.converted}")
    print(f"Ja processados: {totals.skipped}")
    print(f"Falhas: {totals.failed}")
    if not args.dry_run:
        print(f"Tamanho anterior: {human_gib(totals.bytes_before)}")
        print(f"Tamanho atual: {human_gib(totals.bytes_after)}")
        print(f"Espaco liberado: {human_gib(totals.bytes_before - totals.bytes_after)}")
    print(f"Tempo: {elapsed / 60:,.1f} minutos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
