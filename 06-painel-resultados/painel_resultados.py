"""Painel de leitura dos manifestos, CSVs e figuras das execuções independentes.

Não abre Parquets, não altera resultados e não aciona etapas do pipeline.
"""

from __future__ import annotations

import html
import io
import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "execucoes"
PAGES = [
    "Visão geral",
    "01 · Coleta de dados",
    "02 · Remoção de input",
    "03.01 · Features e janelas",
    "03.02 · Pré-processamento e drift",
    "03.03 · Autoencoder",
    "03.04 · Isolation Forest",
    "03.05 · Convergência AE × IF",
    "04.01 · Candidatos C#",
    "04.02 · Validação semântica",
    "04.03 · Label Cloud",
    "05 · Avaliação final",
]
SPLIT_LABELS = {
    "estresse_pre_dencun_2024": "Pré-Dencun",
    "transicao_dencun_2024": "Transição Dencun",
    "treino_2024_pos_dencun": "Treino pós-Dencun",
    "validacao_2025_pre_pectra": "Validação pré-Pectra",
    "transicao_pectra_2025": "Transição Pectra",
    "teste_final_2025": "Teste final",
    "estresse_fusaka_2025": "Estresse Fusaka",
}
ROLE_LABELS = {
    "training": "Treino", "validation": "Validação", "final_test": "Teste final",
    "historical_stress": "Estresse histórico", "protocol_transition": "Transição de protocolo",
    "future_stress": "Estresse futuro",
}


@dataclass
class Report:
    title: str
    notes: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    figures: list[tuple[str, bytes]] = field(default_factory=list)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@st.cache_data(ttl=30, show_spinner=False)
def read_csv_cached(path: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    return pd.read_csv(path, low_memory=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return read_csv_cached(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def run_dirs() -> list[Path]:
    if not RUNS_ROOT.is_dir():
        return []
    return sorted(
        (path for path in RUNS_ROOT.iterdir() if path.is_dir() and re.fullmatch(r"amostra-\d+pct", path.name)),
        key=lambda path: int(re.search(r"\d+", path.name).group()),
    )


def pct(run: Path) -> int:
    return int(re.search(r"\d+", run.name).group())


def stage_complete(run: Path, stage: str) -> bool:
    if stage == "ml":
        return all((run / p).is_file() for p in (
            "03-machine-learning/02-ae/resultados/02_resumo_treinamento.csv",
            "03-machine-learning/03-if/resultados/02_resumo_treinamento.csv",
        ))
    if stage == "semantic":
        return all((run / p).is_file() for p in (
            "04-pos-processamento/06-validacao-semantica-front-running/resultados/consolidado/05_resumo_consolidacao.csv",
            "05-analise-resultados/resultados-rotulos-semanticos/06_selecao_validacao_teste.csv",
        ))
    return False


def status_text(run: Path) -> str:
    if stage_complete(run, "semantic"):
        return "avaliação final disponível"
    if stage_complete(run, "ml"):
        return "ML disponível; avaliação semântica pendente"
    return "execução parcial"


def note(report: Report, message: str, *, warning: bool = False) -> None:
    report.notes.append(message)
    (st.warning if warning else st.info)(message)


def table(report: Report, title: str, frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty:
        return
    st.subheader(title)
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Baixar tabela CSV", frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{key}.csv", mime="text/csv", key=f"csv-{key}",
    )
    report.tables.append((title, frame.copy()))


def compact_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f} mi"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f} mil"
    if abs(value) >= 100:
        return f"{value:,.0f}".replace(",", ".")
    return f"{value:.2f}"


def bar_chart(report: Report, title: str, frame: pd.DataFrame, *, category: str,
              value: str, group: str = "Execução", percent: bool = False,
              y_label: str | None = None, y_lim: tuple[float, float] | None = None) -> None:
    if frame.empty or not {category, value, group}.issubset(frame.columns):
        return
    data = frame[[category, value, group]].dropna().copy()
    if data.empty:
        return
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna(subset=[value])
    if data.empty:
        return
    pivot = data.pivot_table(index=category, columns=group, values=value, aggfunc="first")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 1.3), 4.6))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y_label or ("Percentual" if percent else "Quantidade"))
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title=group, fontsize=8)
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    plt.close(fig)
    content = buffer.getvalue()
    st.image(content, caption=title, width="stretch")
    st.download_button(
        "Baixar gráfico em PDF", pdf_buffer.getvalue(),
        file_name=f"grafico_{len(report.figures) + 1:02d}.pdf",
        mime="application/pdf", key=f"chart-pdf-{len(report.figures)}",
    )
    report.figures.append((title, content))


def stacked_percent_chart(report: Report, title: str, frame: pd.DataFrame, *,
                          label_col: str, segments: list[str], key: str) -> None:
    if frame.empty or not {label_col, *segments}.issubset(frame.columns):
        return
    values = frame[segments].apply(pd.to_numeric, errors="coerce").fillna(0)
    totals = values.sum(axis=1)
    valid = totals.gt(0)
    if not valid.any():
        return
    values = values.loc[valid].reset_index(drop=True)
    labels = frame.loc[valid, label_col].astype(str).tolist()
    totals = totals.loc[valid].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(3.3, 0.55 * len(labels) + 1.8)))
    left = [0.0] * len(labels)
    palette = plt.get_cmap("tab20")
    for index, segment in enumerate(segments):
        shares = (100 * values[segment] / totals).tolist()
        ax.barh(labels, shares, left=left, label=segment,
                color=palette(index), edgecolor="white", linewidth=0.3)
        left = [start + share for start, share in zip(left, shares)]
    ax.set(title=title, xlabel="Parcela (%)", xlim=(0, 100))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=min(3, len(segments)), fontsize=8)
    fig.tight_layout()
    png_buffer, pdf_buffer = io.BytesIO(), io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=150, bbox_inches="tight")
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    plt.close(fig)
    st.image(png_buffer.getvalue(), caption=title, width="stretch")
    st.download_button("Baixar gráfico em PDF", pdf_buffer.getvalue(),
                       file_name=f"{key}.pdf", mime="application/pdf", key=f"chart-pdf-{key}")
    report.figures.append((title, png_buffer.getvalue()))


def selected_semantic_lift(runs: list[Path]) -> pd.DataFrame:
    pieces = []
    base = "05-analise-resultados/resultados-rotulos-semanticos"
    required_selection = {"policy", "method", "selected_configuration", "test_prevalence"}
    required_top = {"split", "policy", "attack_scope", "method", "configuration",
                    "top_fraction", "top_n", "positives_in_top", "precision_at_top",
                    "recall_at_top", "lift_at_top"}
    for run in runs:
        selection = read_csv(run / base / "06_selecao_validacao_teste.csv")
        top = read_csv(run / base / "04_metricas_top_k.csv")
        if not required_selection.issubset(selection) or not required_top.issubset(top):
            continue
        top = top[top["split"].eq("teste_final_2025") & top["attack_scope"].eq("all")]
        joined = top.merge(
            selection[["policy", "method", "selected_configuration", "test_prevalence"]],
            left_on=["policy", "method", "configuration"],
            right_on=["policy", "method", "selected_configuration"],
            how="inner", validate="many_to_one",
        )
        joined = joined[joined["top_fraction"].isin((0.01, 0.05, 0.10))]
        if joined.empty:
            continue
        joined = joined.copy()
        joined.insert(0, "Execução", f"{pct(run)}%")
        pieces.append(joined)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True)


def selected_semantic_pairwise(runs: list[Path]) -> pd.DataFrame:
    pieces = []
    base = "05-analise-resultados/resultados-rotulos-semanticos"
    required_selection = {"policy", "method", "selected_configuration"}
    required_paired = {"split", "policy", "attack_scope", "method", "configuration",
                       "events", "attacker_victim_pairs", "attacker_score_higher",
                       "pairwise_superiority_rate", "events_all_attackers_above_victim",
                       "event_superiority_rate"}
    for run in runs:
        selection = read_csv(run / base / "06_selecao_validacao_teste.csv")
        paired = read_csv(run / base / "05_metricas_pareadas_eventos.csv")
        if not required_selection.issubset(selection) or not required_paired.issubset(paired):
            continue
        paired = paired[paired["split"].eq("teste_final_2025") & paired["attack_scope"].eq("all")]
        joined = paired.merge(
            selection[["policy", "method", "selected_configuration"]],
            left_on=["policy", "method", "configuration"],
            right_on=["policy", "method", "selected_configuration"],
            how="inner", validate="many_to_one",
        )
        if joined.empty:
            continue
        joined = joined.copy()
        joined.insert(0, "Execução", f"{pct(run)}%")
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def selected_semantic_by_type(runs: list[Path]) -> pd.DataFrame:
    pieces = []
    base = "05-analise-resultados/resultados-rotulos-semanticos"
    required_selection = {"policy", "method", "selected_configuration"}
    required_metrics = {"split", "policy", "attack_scope", "method", "configuration",
                        "rows", "positives", "contextual_negatives", "prevalence",
                        "roc_auc", "pr_auc_average_precision"}
    for run in runs:
        selection = read_csv(run / base / "06_selecao_validacao_teste.csv")
        metrics = read_csv(run / base / "02_metricas_discriminacao.csv")
        if not required_selection.issubset(selection) or not required_metrics.issubset(metrics):
            continue
        metrics = metrics[
            metrics["split"].eq("teste_final_2025")
            & metrics["attack_scope"].isin(("insertion", "displacement"))
        ]
        joined = metrics.merge(
            selection[["policy", "method", "selected_configuration"]],
            left_on=["policy", "method", "configuration"],
            right_on=["policy", "method", "selected_configuration"],
            how="inner", validate="many_to_one",
        )
        if joined.empty:
            continue
        joined = joined.copy()
        joined.insert(0, "Execução", f"{pct(run)}%")
        pieces.append(joined)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def show_semantic_by_type(report: Report, runs: list[Path]) -> None:
    data = selected_semantic_by_type(runs)
    if data.empty:
        return
    st.subheader("Desempenho por tipo de ataque · teste final")
    note(report, "Inserção e deslocamento são avaliados separadamente com as configurações já escolhidas pela PR-AUC agregada na validação pré-Pectra; não há nova escolha de modelo usando o teste ou cada subtipo. O denominador de deslocamento pode ser muito menor. ROC-AUC abaixo de 0,5 indica ordenação inversa à esperada nessa subpopulação, não ausência de casos confirmados.")
    view = data[["Execução", "policy", "method", "configuration", "attack_scope", "rows",
                 "positives", "contextual_negatives", "prevalence", "roc_auc",
                 "pr_auc_average_precision"]].copy()
    view["policy"] = view["policy"].replace({"strict": "Estrita", "sensitivity": "Sensibilidade"})
    view["attack_scope"] = view["attack_scope"].replace({
        "insertion": "Inserção", "displacement": "Deslocamento"})
    view = view.rename(columns={
        "policy": "Política", "method": "Modelo", "configuration": "Configuração",
        "attack_scope": "Tipo", "rows": "Transações avaliadas", "positives": "Atacantes",
        "contextual_negatives": "Vítimas contextuais", "prevalence": "Prevalência contextual",
        "roc_auc": "ROC-AUC", "pr_auc_average_precision": "PR-AUC",
    })
    table(report, "Métricas dos modelos selecionados por tipo", view,
          key="avaliacao_por_tipo_ataque")
    policy = st.radio("Política do gráfico por tipo", ("Estrita", "Sensibilidade"),
                      horizontal=True, key="subtype-policy")
    chart = view[view["Política"].eq(policy)].copy()
    chart["Execução · Modelo"] = chart["Execução"] + " · " + chart["Modelo"]
    bar_chart(report, f"ROC-AUC por tipo de ataque · política {policy.lower()}", chart,
              category="Tipo", value="ROC-AUC", group="Execução · Modelo",
              y_label="ROC-AUC", y_lim=(0, 1))
    note(report, "Leia cada barra junto aos números de atacantes e vítimas da tabela. PR-AUC tem como referência a prevalência contextual, que difere entre inserção e deslocamento; seus valores brutos não devem ser comparados entre tipos sem considerar essa base. Estas métricas descrevem apenas os eventos confirmados e vítimas contextuais da política exibida, não a rede Ethereum inteira.")


def show_semantic_lift(report: Report, runs: list[Path]) -> None:
    data = selected_semantic_lift(runs)
    if data.empty:
        return
    st.subheader("Lift no topo do ranking · teste final")
    note(report, "Lift = precisão entre as transações com maiores escores ÷ prevalência de atacantes na população semântica avaliada. A linha 1× equivale a uma seleção aleatória dessa população; valores acima de 1× indicam enriquecimento. Top 1%, 5% e 10% são frações do ranking, independentes dos cortes Q99 e Q99,5. Modelos e configurações foram escolhidos na validação pré-Pectra; o teste final não orienta essa escolha.")
    view = data[["Execução", "policy", "method", "configuration", "top_fraction",
                 "top_n", "positives_in_top", "test_prevalence", "precision_at_top",
                 "recall_at_top", "lift_at_top"]].copy()
    view["policy"] = view["policy"].replace({"strict": "Estrita", "sensitivity": "Sensibilidade"})
    view["top_fraction"] = (view["top_fraction"] * 100).round().astype(int)
    view = view.rename(columns={"policy": "Política", "method": "Modelo",
                                "configuration": "Configuração", "top_fraction": "Topo (%)",
                                "top_n": "Transações no topo", "positives_in_top": "Atacantes no topo",
                                "test_prevalence": "Prevalência contextual",
                                "precision_at_top": "Precisão no topo", "recall_at_top": "Recall no topo",
                                "lift_at_top": "Lift (×)"})
    table(report, "Lift dos modelos selecionados · top 1%, 5% e 10%", view,
          key="avaliacao_lift_top_k")
    run_labels = sorted(data["Execução"].unique(), key=lambda label: int(label.rstrip("%")))
    colors_by_run = {label: plt.get_cmap("tab10")(index) for index, label in enumerate(run_labels)}
    for policy, label in (("strict", "Estrita"), ("sensitivity", "Sensibilidade")):
        subset = data[data["policy"].eq(policy)]
        if subset.empty:
            continue
        title = f"Lift no topo · política {label.lower()} · teste final"
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for (run_label, method), group in subset.groupby(["Execução", "method"]):
            group = group.sort_values("top_fraction")
            ax.plot(100 * group["top_fraction"], group["lift_at_top"],
                    marker="o", linewidth=2, label=f"{run_label} · {method}",
                    color=colors_by_run[run_label], linestyle="-" if method == "AE" else "--")
        ax.axhline(1, color="black", linewidth=1, linestyle=":", label="Referência aleatória · 1×")
        ax.set(title=title, xlabel="Fração superior do ranking (%)", ylabel="Lift sobre a prevalência (×)",
               xticks=[1, 5, 10])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        png_buffer, pdf_buffer = io.BytesIO(), io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=150, bbox_inches="tight")
        fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
        plt.close(fig)
        st.image(png_buffer.getvalue(), caption=title, width="stretch")
        st.download_button("Baixar gráfico de lift em PDF", pdf_buffer.getvalue(),
                           file_name=f"lift_semantico_{policy}_teste_final_2025.pdf",
                           mime="application/pdf", key=f"lift-pdf-{policy}")
        report.figures.append((title, png_buffer.getvalue()))
    note(report, "O lift se refere somente aos candidatos adjudicados e vítimas contextuais, não à prevalência de ataques em toda a Ethereum. Como a prevalência contextual é alta, o lift máximo possível fica próximo de 1 ÷ prevalência; diferenças entre percentuais também dependem dessa base e do número de transações no top 1%.")


def figure_gallery(report: Report, runs: list[Path], title: str,
                   relative_png: str, explanation: str,
                   relative_pdf: str | None = None) -> None:
    items = [(run, run / relative_png) for run in runs if (run / relative_png).is_file()]
    if not items:
        return
    st.subheader(title)
    st.caption(explanation)
    report.notes.append(f"{title}: {explanation}")
    columns = st.columns(min(2, len(items)))
    for index, (run, path) in enumerate(items):
        with columns[index % len(columns)]:
            with st.expander(f"{pct(run)}% · {path.name}", expanded=True):
                image = path.read_bytes()
                st.image(image, width="stretch")
                pdf = run / relative_pdf if relative_pdf else path.with_suffix(".pdf")
                if pdf.is_file():
                    st.download_button(
                        "Baixar PDF original", pdf.read_bytes(),
                        file_name=f"{run.name}_{pdf.name}", mime="application/pdf",
                        key=f"fig-{run.name}-{pdf.name}",
                    )
                else:
                    st.caption("Esta figura não possui PDF original nesta execução.")
                report.figures.append((f"{pct(run)}% · {title}", image))


def collect(runs: list[Path], relative: str, columns: list[str],
            *, filter_fn=None) -> pd.DataFrame:
    pieces = []
    for run in runs:
        frame = read_csv(run / relative)
        if frame.empty:
            continue
        if filter_fn is not None:
            frame = filter_fn(frame)
        if frame.empty:
            continue
        available = [column for column in columns if column in frame.columns]
        if not available:
            continue
        part = frame[available].copy()
        part.insert(0, "Execução", f"{pct(run)}%")
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def split_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "split" in frame:
        frame["split"] = frame["split"].map(SPLIT_LABELS).fillna(frame["split"])
    if "role" in frame:
        frame["role"] = frame["role"].map(ROLE_LABELS).fillna(frame["role"])
    return frame


def confirmed_sets(run: Path) -> dict | None:
    base = run / "04-pos-processamento/06-validacao-semantica-front-running/resultados/consolidado"
    roles = read_csv(base / "02_papeis_evento.csv")
    events = read_csv(base / "01_eventos_adjudicados.csv")
    role_columns = {"split", "adjudicated_type", "decision", "role", "strict_label", "tx_hash"}
    event_columns = {"split", "adjudicated_type", "decision", "strict_positive_event",
                     "attacker_front_hash", "attacker_back_hash", "victim_hash"}
    if roles.empty or events.empty or not role_columns.issubset(roles) or not event_columns.issubset(events):
        return None
    attackers = roles[
        roles["decision"].eq("confirmado")
        & roles["role"].isin(("attacker_front", "attacker_back"))
        & pd.to_numeric(roles["strict_label"], errors="coerce").eq(1)
    ]
    confirmed = events[
        events["decision"].eq("confirmado")
        & pd.to_numeric(events["strict_positive_event"], errors="coerce").eq(1)
    ]
    transactions: set[tuple[str, str]] = set()
    transactions_by_group: dict[tuple[str, str], set[str]] = {}
    for split, kind, tx_hash in attackers[["split", "adjudicated_type", "tx_hash"]].itertuples(index=False, name=None):
        if pd.isna(tx_hash):
            continue
        split, kind, tx_hash = str(split), str(kind), str(tx_hash).lower()
        transactions.add((split, tx_hash))
        transactions_by_group.setdefault((split, kind), set()).add(tx_hash)
    event_ids: set[tuple[str, str, str, str, str]] = set()
    events_by_group: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    for row in confirmed[["split", "adjudicated_type", "attacker_front_hash",
                          "attacker_back_hash", "victim_hash"]].itertuples(index=False, name=None):
        split, kind = str(row[0]), str(row[1])
        hashes = tuple("" if pd.isna(value) else str(value).lower() for value in row[2:])
        event_ids.add((split, kind, *hashes))
        events_by_group.setdefault((split, kind), set()).add(hashes)
    return {"transactions": transactions, "events": event_ids,
            "transactions_by_group": transactions_by_group, "events_by_group": events_by_group}


def set_comparison(first: set, second: set) -> dict:
    common = len(first & second)
    return {"Origem A": len(first), "Destino B": len(second),
            "A ∩ B": common, "Só em A": len(first - second), "Só em B": len(second - first),
            "A preservado em B (%)": round(100 * common / len(first), 2) if first else None,
            "A ⊆ B": first <= second}


def show_confirmed_overlap(report: Report, runs: list[Path]) -> None:
    available = {run: data for run in runs if (data := confirmed_sets(run)) is not None}
    if len(available) < 2:
        note(report, "A comparação de conjuntos exige ao menos duas execuções selecionadas com a consolidação semântica concluída.")
        return
    st.subheader("Sobreposição dos casos confirmados entre execuções")
    note(report, "A é a execução menor e B a maior. A ∩ B contém identificações presentes em ambas; 'só em A' deixou de ser identificado em B; 'só em B' é adicional. Comparamos hashes de transações atacantes únicas e, separadamente, eventos definidos pela combinação dos hashes de atacante(s) e vítima — nunca pelos IDs internos de auditoria.")
    pairs = list(combinations(available, 2))
    summary_rows = []
    for first_run, second_run in pairs:
        first, second = available[first_run], available[second_run]
        for label, key in (("Transações atacantes únicas", "transactions"),
                           ("Eventos confirmados", "events")):
            summary_rows.append({"A → B": f"{pct(first_run)}% → {pct(second_run)}%",
                                 "Unidade": label,
                                 **set_comparison(first[key], second[key])})
    summary = pd.DataFrame(summary_rows)
    table(report, "Interseção e diferenças de conjuntos", summary, key="sobreposicao_confirmados")
    bar_chart(report, "Parcela das identificações de A preservada em B", summary,
              category="A → B", value="A preservado em B (%)", group="Unidade", percent=True)
    note(report, "Os percentuais de blocos usam amostras aninhadas. Uma identificação ausente na execução maior não decorre de o bloco ter saído da amostra: pode refletir diferenças na detecção, no enriquecimento ou na decisão; o resumo de conjuntos, por si só, não determina a causa. Uma mesma transação pode aparecer em mais de um tipo de evento, portanto subtotais por tipo não devem ser somados como transações únicas.")

    choices = {f"{pct(a)}% → {pct(b)}%": (a, b) for a, b in pairs}
    choice = st.selectbox("Detalhar janelas e tipos", list(choices), key="overlap-pair")
    first_run, second_run = choices[choice]
    first, second = available[first_run], available[second_run]
    detail_rows = []
    for label, key in (("Transações atacantes", "transactions_by_group"),
                       ("Eventos", "events_by_group")):
        groups = sorted(first[key].keys() | second[key].keys())
        for split, kind in groups:
            detail_rows.append({"Unidade": label, "split": split,
                                "Tipo": "Inserção" if kind == "insertion" else "Deslocamento",
                                **set_comparison(first[key].get((split, kind), set()),
                                                 second[key].get((split, kind), set()))})
    table(report, f"Detalhe por janela e tipo · {choice}",
          split_labels(pd.DataFrame(detail_rows)), key="sobreposicao_detalhe")

    hash_rows = []
    for split, kind in sorted(first["transactions_by_group"].keys() | second["transactions_by_group"].keys()):
        a = first["transactions_by_group"].get((split, kind), set())
        b = second["transactions_by_group"].get((split, kind), set())
        for status, hashes in (("em ambas", a & b), ("só em A", a - b), ("só em B", b - a)):
            hash_rows.extend({"janela": split, "tipo": kind, "situacao": status, "tx_hash": tx_hash}
                             for tx_hash in sorted(hashes))
    st.download_button("Baixar hashes comparados em CSV",
                       pd.DataFrame(hash_rows, columns=["janela", "tipo", "situacao", "tx_hash"])
                       .to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"sobreposicao_confirmados_{pct(first_run)}_{pct(second_run)}pct.csv",
                       mime="text/csv", key="overlap-hashes")


def show_candidate_disposition(report: Report, runs: list[Path]) -> None:
    statuses = {
        "insertion:confirmed": "Inserção confirmada",
        "displacement:confirmed": "Deslocamento confirmado",
        "displacement:probable": "Deslocamento provável",
        "insertion:rejected": "Inserção rejeitada",
        "displacement:rejected": "Deslocamento rejeitado",
        "suppression:requires_mempool_evidence": "Suppression sem evidência de mempool",
    }
    base = "04-pos-processamento/06-validacao-semantica-front-running/resultados"
    by_run: dict[str, list[dict]] = {}
    complete_rows = []
    for run in runs:
        rows = []
        for split in SPLIT_LABELS:
            manifest = read_json(run / base / split / "07_manifest_validacao.json")
            counts = manifest.get("status_counts")
            if not isinstance(counts, dict) or "events" not in manifest:
                continue
            values = {label: int(counts.get(raw, 0)) for raw, label in statuses.items()}
            events = int(manifest["events"])
            if sum(values.values()) != events:
                note(report, f"A contagem dos estados não fecha para {pct(run)}% · {SPLIT_LABELS[split]}; esta janela foi omitida.", warning=True)
                continue
            rows.append({"Execução": f"{pct(run)}%", "split": split,
                         "Eventos candidatos": events, **values})
        if not rows:
            continue
        by_run[f"{pct(run)}%"] = rows
        if len(rows) == len(SPLIT_LABELS):
            summary = {"Execução": f"{pct(run)}%",
                       "Eventos candidatos": sum(row["Eventos candidatos"] for row in rows)}
            summary.update({label: sum(row[label] for row in rows) for label in statuses.values()})
            complete_rows.append(summary)
        else:
            note(report, f"{pct(run)}% tem {len(rows)} de {len(SPLIT_LABELS)} janelas com a validação registrada. Seus totais parciais não entram no gráfico comparativo.", warning=True)
    if not by_run:
        return
    st.subheader("Destino dos candidatos estruturais C#")
    note(report, "Cada evento candidato recebe um dos estados abaixo. Inserções e deslocamentos podem ser confirmados ou rejeitados; deslocamentos também podem ser prováveis. Suppression permanece sem confirmação por exigir evidência de mempool. As contagens são de eventos, não de transações únicas, e não representam a prevalência de ataques na Ethereum.")
    if complete_rows:
        complete = pd.DataFrame(complete_rows)
        table(report, "Destino dos candidatos · sete janelas completas", complete,
              key="destino_candidatos_resumo")
        stacked_percent_chart(report, "Proporção dos estados dos candidatos por execução",
                              complete, label_col="Execução", segments=list(statuses.values()),
                              key="destino_candidatos_percentual")
    choice = st.selectbox("Detalhar os candidatos por janela", list(by_run),
                          key="candidate-disposition-run")
    details = split_labels(pd.DataFrame(by_run[choice]))
    table(report, f"Destino dos candidatos por janela · {choice}", details,
          key="destino_candidatos_janelas")


def show_label_coverage(report: Report, runs: list[Path]) -> None:
    coverage = collect(
        runs, "05-analise-resultados/resultados-rotulos-semanticos/01_cobertura_rotulos.csv",
        ["split", "policy", "semantic_transactions", "evaluable_transactions",
         "positive_attackers", "contextual_negative_victims", "unlabeled_in_policy",
         "conflicting_excluded", "evaluable_share"],
        filter_fn=lambda frame: frame[frame["split"].eq("teste_final_2025")],
    )
    if coverage.empty:
        return
    st.subheader("Cobertura dos rótulos · teste final")
    note(report, "As métricas finais não usam todas as transações com papel semântico registrado. Em cada política, entram apenas atacantes positivos e vítimas contextuais avaliáveis; casos não rotulados não são presumidos negativos, e conflitos são excluídos.")
    view = coverage.drop(columns=["split"]).copy()
    view["policy"] = view["policy"].replace({"strict": "Estrita", "sensitivity": "Sensibilidade"})
    view = view.rename(columns={
        "policy": "Política", "semantic_transactions": "Transações com papel semântico",
        "evaluable_transactions": "Avaliáveis", "positive_attackers": "Atacantes positivos",
        "contextual_negative_victims": "Vítimas contextuais",
        "unlabeled_in_policy": "Não rotuladas", "conflicting_excluded": "Excluídas por conflito",
        "evaluable_share": "Parcela avaliável",
    })
    table(report, "Composição da população avaliada por política", view,
          key="avaliacao_cobertura_rotulos")
    chart = view.copy()
    chart["Execução e política"] = chart["Execução"] + " · " + chart["Política"]
    stacked_percent_chart(report, "Cobertura da referência semântica no teste final",
                          chart, label_col="Execução e política",
                          segments=["Avaliáveis", "Não rotuladas", "Excluídas por conflito"],
                          key="cobertura_rotulos_teste_final")
    note(report, "O gráfico mostra parcelas dentro das transações com papel semântico da respectiva execução, não de todas as transações da blockchain. A política de sensibilidade inclui deslocamentos prováveis, mas pode excluir transações com papéis conflitantes; mudanças nas métricas entre políticas também refletem a mudança da população avaliada.")


def font_name() -> str:
    for path in (Path("C:/Windows/Fonts/arial.ttf"),
                 Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if path.is_file():
            pdfmetrics.registerFont(TTFont("ReportFont", str(path)))
            return "ReportFont"
    return "Helvetica"


def export_pdf(report: Report, runs: list[Path]) -> bytes:
    output = io.BytesIO()
    font = font_name()
    styles = getSampleStyleSheet()
    normal = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontName=font,
                            fontSize=9, leading=13, spaceAfter=8)
    heading = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], fontName=font,
                             fontSize=13, leading=16, spaceBefore=12, spaceAfter=8)
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontName=font,
                                 alignment=TA_CENTER, fontSize=17, leading=22)
    cell = ParagraphStyle("CellCustom", parent=normal, fontSize=7, leading=9,
                          spaceAfter=0)
    story = [Paragraph(html.escape(report.title), title_style), Spacer(1, 8),
             Paragraph("Execuções selecionadas: " + ", ".join(f"{pct(run)}%" for run in runs), normal)]
    for message in report.notes:
        story.append(Paragraph(html.escape(message), normal))
    doc = SimpleDocTemplate(output, pagesize=landscape(A4), leftMargin=28,
                            rightMargin=28, topMargin=30, bottomMargin=30)
    usable = landscape(A4)[0] - 56
    for title, frame in report.tables:
        story.append(Paragraph(html.escape(title), heading))
        visible = frame.iloc[:60, :9].copy().fillna("")
        rows = [[Paragraph(html.escape(str(column)), cell) for column in visible.columns]]
        for row in visible.itertuples(index=False, name=None):
            rows.append([Paragraph(html.escape(str(item)[:60]), cell) for item in row])
        if rows:
            widths = [usable / len(rows[0])] * len(rows[0])
            pdf_table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            pdf_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF4")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D7DE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(pdf_table)
        if len(frame) > 60 or len(frame.columns) > 9:
            story.append(Paragraph("Tabela resumida no PDF; baixe o CSV para os dados completos.", normal))
    for title, image in report.figures:
        reader = ImageReader(io.BytesIO(image))
        width, height = reader.getSize()
        factor = min(usable / width, 370 / height)
        story.append(KeepTogether([
            Paragraph(html.escape(title), heading),
            Image(io.BytesIO(image), width=width * factor, height=height * factor),
        ]))
    doc.build(story)
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="Resultados · Ethereum", layout="wide")
    st.title("Pipeline Ethereum · análise consolidada")
    st.caption("Painel de consulta. Lê apenas CSV, JSON, PNG e PDF; não abre Parquets nem executa o pipeline.")
    available = run_dirs()
    if not available:
        st.error(f"Nenhuma execução encontrada em {RUNS_ROOT}")
        return
    st.sidebar.header("Navegação")
    page = st.sidebar.radio("Etapa", PAGES)
    st.sidebar.header("Execuções")
    selected = []
    for run in available:
        checked = st.sidebar.checkbox(
            f"{pct(run)}% — {status_text(run)}", value=stage_complete(run, "semantic"),
            key=f"run-{run.name}",
        )
        if checked:
            selected.append(run)
    if not selected:
        st.warning("Selecione ao menos uma execução na barra lateral.")
        return
    st.sidebar.caption("A seleção afeta as comparações a partir da etapa 03. As seções 01 e 02 descrevem entradas compartilhadas.")
    report = Report(page)

    if page == PAGES[0]:
        note(report, "Os mesmos Parquets externos alimentam as execuções. Cada percentual seleciona blocos completos por módulo; as janelas são cronológicas e não se sobrepõem.")
        table(report, "Disponibilidade das etapas", pd.DataFrame([
            {"Execução": f"{pct(run)}%", "ML": "concluído" if stage_complete(run, "ml") else "pendente",
             "Avaliação semântica": "disponível" if stage_complete(run, "semantic") else "pendente"}
            for run in selected]), key="visao_geral_disponibilidade")
        note(report, "Comparações entre percentuais são descritivas: as amostras são aninhadas, mas AE e IF são treinados novamente em cada execução. Contagens maiores não implicam, por si, melhor desempenho.")

    elif page == PAGES[1]:
        config = read_json(ROOT / "configuracao/pipeline.json")
        note(report, "Fonte: Parquets públicos de transações Ethereum, organizados em partições diárias. O diretório é externo ao repositório e configurado por dados.root_template.")
        st.code(str(config.get("dados", {}).get("root_template", "não informado")), language="text")
        source = next((run for run in selected if (run / "03-machine-learning/01-features/resultados-splits/00_manifest_splits_temporais.json").is_file()), None)
        if source:
            manifest = read_json(source / "03-machine-learning/01-features/resultados-splits/00_manifest_splits_temporais.json")
            frame = pd.DataFrame(manifest.get("splits", []))
            cols = [c for c in ["key", "role", "start_date", "end_date", "expected_days", "source_parquet_files"] if c in frame]
            if cols:
                table(report, "Cobertura diária e propósito das janelas", split_labels(frame[cols].rename(columns={"key": "split"})), key="coleta_janelas")
                if "source_parquet_files" in frame:
                    note(report, f"Os manifestos das janelas registram {int(frame['source_parquet_files'].sum())} arquivos diários ao todo. A cobertura reportada deve ser conferida contra os manifestos da execução selecionada.")
        note(report, "A coleta S3 original e a remoção de input antecedem estas execuções; elas não foram repetidas a cada percentual.")

    elif page == PAGES[2]:
        note(report, "A coluna input foi removida dos Parquets brutos para reduzir armazenamento. Ela não entra nas features do AE nem do IF; quando necessária à verificação semântica dos candidatos, a calldata é recuperada seletivamente por RPC.")
        measurement = read_csv(ROOT / "06-painel-resultados/resultados-armazenamento/01_comparacao_s3_local.csv")
        if not measurement.empty:
            table(report, "Tamanho S3 original × tamanho local atual", measurement, key="armazenamento_s3_local")
            note(report, "A diferença mede o efeito conjunto da remoção de input e da regravação com compressão ZSTD; não isola a contribuição da coluna. Só compare anos cuja listagem S3 e local contenham exatamente o mesmo conjunto de arquivos.")
        else:
            note(report, "O script de remoção imprimiu os GiB liberados no terminal, mas não persistiu o total. A medição opcional 01_medir_armazenamento_s3.py compara os tamanhos originais no S3 com os arquivos locais atuais sem abrir Parquets. Até executá-la, este painel não inventa um ganho agregado.", warning=True)
        st.code("02-pre-processamento/01-remover_coluna_input_parquet.py", language="text")
        note(report, "O script valida esquema e contagem de linhas antes de substituir cada arquivo. A comparação S3/local é válida apenas para arquivos pareados; diferenças de cobertura são reportadas separadamente.")

    elif page == PAGES[3]:
        note(report, "A unidade amostral é o bloco completo: 1%, 2%, 4% e 5% correspondem, respectivamente, a block_number % 100, 50, 25 e 20 = 0. Isso preserva a ordem de transações dentro do bloco.")
        rows = []
        for run in selected:
            d = read_json(run / "03-machine-learning/01-features/resultados/02_resumo_dataset.json")
            if d:
                rows.append({"Execução": f"{pct(run)}%", "Transações no treino": d.get("sampled_rows"),
                             "Blocos no treino": d.get("sampled_blocks"), "Arquivos do treino": d.get("number_of_parquet_files"),
                             "Duplicações de hash": d.get("sample_duplicate_hashes")})
        if rows:
            table(report, "Amostra de treinamento", pd.DataFrame(rows), key="features_treino")
        split_rows = []
        for run in selected:
            data = read_json(run / "03-machine-learning/01-features/resultados-splits/00_manifest_splits_temporais.json")
            for item in data.get("splits", []):
                split_rows.append({"Execução": f"{pct(run)}%", "split": item.get("key"),
                                   "role": item.get("role"), "Início": item.get("start_date"),
                                   "Fim": item.get("end_date"), "Transações": item.get("sample_rows"),
                                   "Blocos": item.get("sample_blocks")})
        splits = split_labels(pd.DataFrame(split_rows))
        if not splits.empty:
            table(report, "Sete janelas temporais", splits, key="features_janelas")
            bar_chart(report, "Transações por janela e execução", splits, category="split", value="Transações")
        note(report, "Treino: abril–dezembro/2024, após Dencun. Validação: período de 2025 anterior a Pectra. Teste final: junho–novembro/2025. As transições Dencun/Pectra e os períodos de estresse são mantidos à parte para não misturar regimes de protocolo na calibração.")
        figure_gallery(report, selected, "Correlação de Spearman",
                       "03-machine-learning/01-features/resultados/12_correlacao_spearman.png",
                       "Valores próximos de ±1 indicam relação monotônica forte; alta correlação sinaliza redundância possível, não causalidade.",
                       relative_pdf="03-machine-learning/01-features/resultados/17_correlacao_spearman_vetorial.pdf")

    elif page == PAGES[4]:
        note(report, "Transformações e RobustScaler são ajustados apenas no treino e aplicados sem reajuste nas demais janelas. Isso evita vazamento de informação da validação e do teste.")
        checks = collect(selected, "03-machine-learning/01-features/resultados-preprocessamento/04_validacao_recortes.csv",
                         ["split", "matrix", "rows", "features", "invalid_values"])
        if not checks.empty:
            table(report, "Validação das matrizes preprocessadas", split_labels(checks), key="preprocessamento_validacao")
        drift = collect(selected, "03-machine-learning/01-features/resultados-drift/02_resumo_drift_recortes.csv",
                        ["split", "role", "rows", "psi_mean", "psi_max", "ks_mean", "ks_max", "features_drift_alto"])
        if not drift.empty:
            drift = split_labels(drift)
            table(report, "Drift em relação ao treino", drift, key="drift_recortes")
            bar_chart(report, "PSI médio por janela", drift, category="split", value="psi_mean")
        note(report, "PSI compara a distribuição das features com a do treino; valores maiores sugerem maior deslocamento. O KS aproximado mede a maior diferença entre distribuições acumuladas. Ambos descrevem mudança de distribuição, não provam ataque.")
        figure_gallery(report, selected, "Mapa de PSI",
                       "03-machine-learning/01-features/resultados-drift/07_heatmap_psi.png",
                       "Leia por coluna (janela) e linha (feature): cores mais intensas indicam maior divergência em relação ao treino.")

    elif page == PAGES[5]:
        note(report, "O Autoencoder aprende a reconstruir transações do treino pós-Dencun; erro de reconstrução maior indica maior anomalia. O limiar é calibrado na validação, nunca no teste final.")
        training = collect(selected, "03-machine-learning/02-ae/resultados/02_resumo_treinamento.csv",
                           ["configuration", "features", "training_rows_total", "training_rows_fit",
                            "internal_validation_rows", "best_epoch", "best_internal_validation_loss",
                            "fit_seconds", "device", "peak_gpu_memory_bytes"])
        if not training.empty:
            training["GPU pico (GiB)"] = pd.to_numeric(training.get("peak_gpu_memory_bytes"), errors="coerce") / 2**30
            training = training.drop(columns=["peak_gpu_memory_bytes"])
            table(report, "Treinamento AE por configuração", training, key="ae_treinamento")
            bar_chart(report, "Tempo de ajuste AE · matriz B completa",
                      training[training["configuration"] == "b_completa"], category="configuration",
                      value="fit_seconds", y_label="Segundos")
        rates = collect(selected, "03-machine-learning/02-ae/resultados/06_taxas_anomalias_recortes.csv",
                        ["split", "configuration", "rows", "anomalies_q995", "anomaly_rate_q995"],
                        filter_fn=lambda d: d[d["configuration"] == "b_completa"])
        if not rates.empty:
            table(report, "AE · anomalias q99,5 da matriz B completa", split_labels(rates), key="ae_taxas")
        note(report, "O número de transações usadas no AE é diferente do IF: o AE usa quase todo o treino, com 10% reservados para validação interna. Taxa de anomalias não é precisão nem confirma front-running.")
        figure_gallery(report, selected, "Curvas de treinamento do Autoencoder",
                       "03-machine-learning/02-ae/resultados/10_curvas_treinamento.png",
                       "Cada painel é uma configuração de features. O eixo horizontal mostra as épocas; o vertical, o erro médio quadrático (MSE) de reconstrução. Azul é treino e laranja é validação interna, reservada dentro do treino — não a janela temporal de validação pré-Pectra. Queda conjunta e linhas próximas sugerem convergência sem afastamento evidente; aumento persistente da linha laranja enquanto a azul cai sugeriria sobreajuste. Não compare o MSE bruto entre configurações com conjuntos de features diferentes como se fosse uma métrica de detecção de ataques.")

    elif page == PAGES[6]:
        note(report, "O Isolation Forest usa uma subamostra determinística do treino para ajustar árvores. Por isso sua quantidade de linhas treinadas é menor que a do AE; as pontuações são calculadas nas janelas completas amostradas.")
        training = collect(selected, "03-machine-learning/03-if/resultados/02_resumo_treinamento.csv",
                           ["configuration", "features", "training_rows_total", "training_rows_sample",
                            "sample_share", "n_estimators", "max_samples_tree", "fit_seconds"])
        if not training.empty:
            table(report, "Treinamento IF por configuração", training, key="if_treinamento")
            bar_chart(report, "Tempo de ajuste IF · matriz B completa",
                      training[training["configuration"] == "b_completa"], category="configuration",
                      value="fit_seconds", y_label="Segundos")
        rates = collect(selected, "03-machine-learning/03-if/resultados/05_taxas_anomalias_recortes.csv",
                        ["split", "configuration", "rows", "anomalies_q995", "anomaly_rate_q995"],
                        filter_fn=lambda d: d[d["configuration"] == "b_completa"])
        if not rates.empty:
            table(report, "IF · anomalias q99,5 da matriz B completa", split_labels(rates), key="if_taxas")
        figure_gallery(report, selected, "Taxas de anomalia IF",
                       "03-machine-learning/03-if/resultados/08_taxas_anomalias_recortes.png",
                       "Compare padrões relativos entre janelas; não interprete a altura das barras como taxa de ataques confirmados.")

    elif page == PAGES[7]:
        note(report, "A união AE ∪ IF forma os candidatos para investigação. A interseção indica concordância, e Jaccard = interseção/união resume a sobreposição; não é uma métrica de acerto.")
        joint = collect(selected, "05-analise-resultados/resultados/01_resumo_integrado_ae_if.csv",
                        ["split", "configuration", "rows", "ae_anomalies_q995", "if_anomalies_q995",
                         "intersection", "union", "jaccard"],
                        filter_fn=lambda d: d[d["configuration"] == "b_completa"])
        if not joint.empty:
            joint = split_labels(joint)
            table(report, "Candidatos e concordância · matriz B completa", joint, key="ae_if_convergencia")
            bar_chart(report, "União de candidatos por janela", joint, category="split", value="union")
        note(report, "Ao ampliar a amostra, contagens absolutas tendem a crescer. Compare também taxas e métricas condicionais, não apenas o número de candidatos.")

    elif page == PAGES[8]:
        note(report, "As regras C# adaptadas procuram padrões candidatos de front-running no conjunto amostrado. Nesta etapa, candidato ainda não significa ataque confirmado.")
        rows = []
        for run in selected:
            base = run / "04-pos-processamento/05-rotulador-front-running-csharp/resultados-v3/sampled/independent_sampled"
            for split in SPLIT_LABELS:
                frame = read_csv(base / split / "03_summary.csv")
                if frame.empty:
                    continue
                for _, item in frame.iterrows():
                    rows.append({"Execução": f"{pct(run)}%", "Janela": SPLIT_LABELS[split],
                                 "Detector": item.get("detector"), "Eventos candidatos": item.get("candidate_events"),
                                 "Transações atacante": item.get("candidate_attacker_transactions")})
        candidates = pd.DataFrame(rows)
        if not candidates.empty:
            table(report, "Regras C# por janela", candidates, key="csharp_candidatos")
            total = candidates[candidates["Detector"] == "__total__"]
            bar_chart(report, "Eventos candidatos C# por janela", total, category="Janela", value="Eventos candidatos")
        note(report, "Os detectores podem gerar eventos sobrepostos. Não some linhas de detectores para obter transações únicas; use o total registrado e, depois, a consolidação semântica.")

    elif page == PAGES[9]:
        note(report, "A auditoria semântica enriquece candidatos com transação, input, receipts, logs e metadados ERC-20. Regras assistidas geram decisões de evento; a consolidação aplica políticas estrita e de sensibilidade.")
        show_candidate_disposition(report, selected)
        counts = collect(selected, "04-pos-processamento/06-validacao-semantica-front-running/resultados/consolidado/05_resumo_consolidacao.csv",
                         ["split", "events", "confirmed_insertion", "confirmed_displacement",
                          "probable_displacement", "strict_positive_transactions", "sensitivity_positive_transactions"])
        if not counts.empty:
            counts = split_labels(counts)
            table(report, "Adjudicação consolidada por janela", counts, key="semantica_consolidacao")
            totals = counts.groupby("Execução", as_index=False)[["confirmed_insertion", "confirmed_displacement", "probable_displacement"]].sum()
            totals = totals.melt(id_vars="Execução", var_name="Classe", value_name="Eventos")
            bar_chart(report, "Eventos consolidados por classe", totals, category="Classe", value="Eventos")
        show_confirmed_overlap(report, selected)
        unavailable = [f"{pct(run)}%" for run in selected if not stage_complete(run, "semantic")]
        if unavailable:
            note(report, "Avaliação semântica ainda não disponível para: " + ", ".join(unavailable) + ". A página será atualizada quando os manifestos finais forem gerados.", warning=True)
        note(report, "Na política estrita, os deslocamentos prováveis ficam fora dos positivos; na análise de sensibilidade, entram como positivos. As métricas são condicionais aos candidatos C# e negativos contextuais, não a todas as transações da Ethereum.")

    elif page == PAGES[10]:
        note(report, "O Label Cloud histórico é uma referência externa exploratória. Sua cobertura depende dos endereços catalogados e não constitui verdade-terreno exaustiva para todas as transações.")
        source = next((run for run in selected if (run / "04-pos-processamento/resultados-labelcloud/02_resumo_labelcloud_historico.csv").is_file()), None)
        if source:
            table(report, "Categorias do Label Cloud", read_csv(source / "04-pos-processamento/resultados-labelcloud/02_resumo_labelcloud_historico.csv"), key="labelcloud_categorias")
        label = collect(selected, "05-analise-resultados/resultados-labelcloud-hipotetico/01_metricas_discriminacao.csv",
                        ["split", "configuration", "method", "rows", "positives", "roc_auc", "pr_auc_average_precision"],
                        filter_fn=lambda d: d[(d["split"] == "teste_final_2025") & (d["configuration"] == "b_completa")])
        if not label.empty:
            table(report, "Comparação hipotética com Label Cloud · teste final", split_labels(label), key="labelcloud_teste")
        note(report, "A avaliação assume, hipoteticamente, que endereços MEV Bot são positivos e que os ausentes são negativos. Essa suposição pode produzir falsos negativos; resultados aqui não substituem a avaliação semântica contextual.", warning=True)
        figure_gallery(report, selected, "Curvas ROC/PR com Label Cloud",
                       "05-analise-resultados/resultados-labelcloud-hipotetico/05_curvas_roc_pr_teste_final_2025.png",
                       "ROC compara sensibilidade e falsos positivos; PR enfatiza precisão e recall quando a classe positiva é rara. Interprete sob a hipótese de rótulos adotada.")

    else:
        note(report, "Os modelos e limiares são escolhidos na validação pré-Pectra. O teste final de 2025 é usado somente depois dessa escolha. Compare separadamente as políticas estrita e de sensibilidade.")
        note(report, "Q99,5 (q995) é a referência principal para sinalização de anomalias e comparação AE × IF. Nesta avaliação semântica, após escolher a configuração pela PR-AUC, o programa compara Q99, Q99,5 e Q99,9 pelo F1 na validação pré-Pectra. As matrizes abaixo usam o limiar então selecionado, que pode ser diferente de Q99,5.")
        show_label_coverage(report, selected)
        final = collect(selected, "05-analise-resultados/resultados-rotulos-semanticos/06_selecao_validacao_teste.csv",
                        ["policy", "method", "selected_configuration", "validation_rows", "validation_pr_auc",
                         "selected_threshold", "threshold_selection_status", "validation_selected_threshold_f1",
                         "test_rows", "test_roc_auc", "test_pr_auc", "test_q995_f1",
                         "test_selected_threshold_precision", "test_selected_threshold_recall",
                         "test_selected_threshold_f1"])
        if not final.empty:
            final["Política"] = final["policy"].map({"strict": "Estrita", "sensitivity": "Sensibilidade"})
            final = final.drop(columns=["policy"])
            threshold_view = final[["Execução", "Política", "method", "selected_configuration",
                                    "selected_threshold", "threshold_selection_status",
                                    "validation_selected_threshold_f1", "test_q995_f1",
                                    "test_selected_threshold_f1"]].copy()
            threshold_view["threshold_selection_status"] = threshold_view["threshold_selection_status"].replace({
                "selected_by_validation_f1": "maior F1 na validação",
                "no_positive_predictions_in_validation": "sem positivos na validação; empate",
            })
            table(report, "Limiar selecionado na validação × referência Q99,5",
                  threshold_view, key="avaliacao_limiares")
            selected_quantiles = ", ".join(sorted(final["selected_threshold"].dropna().unique()))
            note(report, f"Limiar(es) selecionado(s) nas execuções exibidas: {selected_quantiles}. Q99 = q990; Q99,5 = q995. Os valores numéricos dos cortes são calibrados separadamente para cada modelo e execução.")
            ties = final[final["threshold_selection_status"] == "no_positive_predictions_in_validation"]
            if not ties.empty:
                affected = ", ".join(sorted({f"{row['method']} de {row['Execução']}" for _, row in ties.iterrows()}))
                note(report, f"Atenção: {affected} não produziu positivos na validação semântica com nenhum dos três cortes. Q99 aparece por desempate com F1 igual a zero, não como evidência de um limiar operacional eficaz.", warning=True)
            performance = final[["Execução", "Política", "method", "selected_configuration",
                                 "validation_rows", "validation_pr_auc", "test_rows", "test_roc_auc",
                                 "test_pr_auc", "test_selected_threshold_precision",
                                 "test_selected_threshold_recall", "test_selected_threshold_f1"]]
            table(report, "Modelos selecionados e desempenho no teste", performance, key="avaliacao_modelos")
            pr = final[final["Política"] == "Estrita"].copy()
            pr["Modelo"] = pr["method"]
            bar_chart(report, "PR-AUC no teste · política estrita", pr, category="Modelo", value="test_pr_auc")
        show_semantic_by_type(report, selected)
        unavailable = [f"{pct(run)}%" for run in selected if not stage_complete(run, "semantic")]
        if unavailable:
            note(report, "Sem métricas finais para " + ", ".join(unavailable) + " enquanto a execução estiver incompleta.", warning=True)
        note(report, "ROC-AUC mede ordenação entre classes; PR-AUC resume precisão × recall. F1 depende do limiar. Os exemplos avaliados são eventos candidatos e negativos contextuais, logo não representam prevalência global de front-running.")
        figure_gallery(report, selected, "Curvas ROC/PR · teste · política estrita",
                       "05-analise-resultados/resultados-rotulos-semanticos/07_curvas_roc_pr_strict_teste_final_2025.png",
                       "Compare a forma das curvas e a prevalência da amostra, não apenas a área. A avaliação de teste não deve orientar nova escolha de modelo ou limiar.")
        show_semantic_lift(report, selected)
        figure_gallery(report, selected, "Superioridade pareada · política estrita",
                       "05-analise-resultados/resultados-rotulos-semanticos/11_heatmap_superioridade_pareada_strict.png",
                       "Cada coluna é uma janela temporal e cada linha é uma configuração AE ou IF. A célula mostra a fração dos pares atacante–vítima do mesmo evento confirmado em que o atacante recebeu escore de anomalia maior. Valores próximos de 1 indicam ordenação favorável; perto de 0,5, pouca preferência; abaixo de 0,5, tendência inversa. O cálculo usa escores, não os cortes Q99/Q99,5, e não mede quantos ataques foram sinalizados. Em inserção, as duas pernas do atacante podem gerar dois pares com a vítima; consulte o número de pares antes de comparar células.")
        paired = selected_semantic_pairwise(selected)
        if not paired.empty:
            paired_view = paired[["Execução", "policy", "method", "configuration", "events",
                                  "attacker_victim_pairs", "attacker_score_higher",
                                  "pairwise_superiority_rate", "events_all_attackers_above_victim",
                                  "event_superiority_rate"]].copy()
            paired_view["policy"] = paired_view["policy"].replace({
                "strict": "Estrita", "sensitivity": "Sensibilidade"})
            paired_view = paired_view.rename(columns={
                "policy": "Política", "method": "Modelo", "configuration": "Configuração",
                "events": "Eventos", "attacker_victim_pairs": "Pares atacante–vítima",
                "attacker_score_higher": "Pares com atacante acima",
                "pairwise_superiority_rate": "Superioridade por par",
                "events_all_attackers_above_victim": "Eventos com todos atacantes acima",
                "event_superiority_rate": "Superioridade por evento",
            })
            table(report, "Superioridade dos modelos selecionados · teste final", paired_view,
                  key="avaliacao_superioridade_pareada")
            note(report, "A taxa por par e a taxa por evento têm denominadores diferentes: a primeira conta cada comparação atacante–vítima; a segunda exige que todas as pernas atacantes de um evento superem a vítima. Mesmo uma boa ordenação pareada pode coexistir com baixo recall no limiar binário selecionado.")
        figure_gallery(report, selected, "Matrizes de confusão · limiar selecionado",
                       "05-analise-resultados/resultados-rotulos-semanticos/10_matrizes_confusao_teste_selecionado_strict.png",
                       "Estas matrizes usam o corte escolhido pelo maior F1 na validação semântica, não necessariamente a referência Q99,5. Leia VN, FP, FN e VP; alto ROC-AUC pode coexistir com baixo recall nesse corte.")

    st.divider()
    st.caption("PDF da página inclui textos, tabelas resumidas e figuras exibidas. Para dados tabulares completos, use os CSVs; os PDFs originais das figuras estão disponíveis junto a cada imagem.")
    if report.notes or report.tables or report.figures:
        try:
            pdf = export_pdf(report, selected)
            st.download_button("Exportar esta página em PDF", pdf,
                               file_name=f"painel_{PAGES.index(page):02d}_{'-'.join(str(pct(run)) for run in selected)}pct.pdf",
                               mime="application/pdf")
        except Exception as exc:
            st.error(f"Não foi possível gerar o PDF desta página: {exc}")


if __name__ == "__main__":
    main()
