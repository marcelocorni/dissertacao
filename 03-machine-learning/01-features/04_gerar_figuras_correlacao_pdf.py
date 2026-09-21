# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib>=3.9,<4",
#   "numpy>=2.0,<3",
#   "pandas>=2.2,<3",
#   "pypdf>=6,<7",
# ]
# ///

"""Gera versões vetoriais em PDF das matrizes de correlação já calculadas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Converte as matrizes CSV de Pearson e Spearman em PDF vetorial."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=here / "resultados",
        help="Diretório produzido pelo artefato 01.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Exibe coeficientes nas células; pode ficar denso com muitas features.",
    )
    return parser.parse_args()


def read_matrix(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Matriz não encontrada: {path}")
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"A matriz não é quadrada: {path}")
    if list(matrix.index) != list(matrix.columns):
        raise ValueError(f"Linhas e colunas não possuem as mesmas features: {path}")
    return matrix.apply(pd.to_numeric, errors="coerce")


def load_context(results_dir: Path) -> str:
    result_path = results_dir / "15_resultado_execucao.json"
    if not result_path.is_file():
        return "Features on-chain do conjunto de treinamento"
    data = json.loads(result_path.read_text(encoding="utf-8"))
    start = data.get("start_date", "")
    end = data.get("end_date", "")
    rows = data.get("correlation_rows_used")
    row_text = f"n = {int(rows):,}".replace(",", ".") if rows else ""
    return f"Treinamento {start} a {end} | {row_text} | blocos completos"


def create_figure(
    matrix: pd.DataFrame,
    method_name: str,
    context: str,
    annotate: bool,
) -> plt.Figure:
    count = len(matrix.columns)
    size = max(12.5, count * 0.47)
    fig, ax = plt.subplots(figsize=(size, size))

    cmap = matplotlib.colormaps["coolwarm"].copy()
    cmap.set_bad("#eeeeee")
    values = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    mesh = ax.pcolormesh(
        np.arange(count + 1),
        np.arange(count + 1),
        values,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        shading="flat",
        edgecolors="none",
    )
    ax.set_xlim(0, count)
    ax.set_ylim(count, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(count) + 0.5)
    ax.set_yticks(np.arange(count) + 0.5)
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7.5)
    ax.set_yticklabels(matrix.index, fontsize=7.5)
    ax.tick_params(length=0)
    ax.set_title(
        f"Matriz de correlação de {method_name}\n{context}",
        fontsize=14,
        pad=18,
    )

    if annotate:
        raw = matrix.to_numpy(dtype=float)
        for row in range(count):
            for column in range(count):
                value = raw[row, column]
                if np.isfinite(value):
                    color = "white" if abs(value) >= 0.60 else "black"
                    ax.text(
                        column + 0.5,
                        row + 0.5,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=4.1,
                        color=color,
                    )

    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Coeficiente de correlação", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)
    fig.text(
        0.5,
        0.008,
        "Fonte: elaboração própria a partir dos dados on-chain do Ethereum.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


def main() -> int:
    args = parse_args()
    results = args.results_dir.resolve()
    pearson = read_matrix(results / "09_correlacao_pearson.csv")
    spearman = read_matrix(results / "10_correlacao_spearman.csv")
    context = load_context(results)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    outputs = [
        (pearson, "Pearson", results / "16_correlacao_pearson_vetorial.pdf"),
        (spearman, "Spearman", results / "17_correlacao_spearman_vetorial.pdf"),
    ]
    combined = results / "18_matrizes_correlacao_vetorial.pdf"

    for matrix, method, output in outputs:
        figure = create_figure(matrix, method, context, args.annotate)
        figure.savefig(output, format="pdf", bbox_inches="tight")
        plt.close(figure)

    # Combina os PDFs já finalizados, preservando exatamente a MediaBox e o
    # CropBox de cada página. Isso evita recortes inconsistentes entre páginas.
    writer = PdfWriter()
    for _, _, output in outputs:
        reader = PdfReader(output)
        for page in reader.pages:
            writer.add_page(page)
    with combined.open("wb") as stream:
        writer.write(stream)

    print("PDFs vetoriais gerados:")
    for _, _, output in outputs:
        print(f"  {output}")
    print(f"  {combined} (duas páginas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
