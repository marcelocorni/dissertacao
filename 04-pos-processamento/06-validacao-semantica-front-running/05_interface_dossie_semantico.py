# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.4.3,<2", "streamlit>=1.49,<2"]
# ///

"""Interface Streamlit somente leitura para consulta do dossiê semântico."""

from __future__ import annotations

import csv
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import streamlit as st


HERE = Path(__file__).resolve().parent
RESULTS_ROOT = Path(os.environ.get("SEMANTIC_RESULTS_ROOT", HERE / "resultados")).resolve()
SPLIT_ORDER = (
    "estresse_pre_dencun_2024",
    "transicao_dencun_2024",
    "treino_2024_pos_dencun",
    "validacao_2025_pre_pectra",
    "transicao_pectra_2025",
    "teste_final_2025",
    "estresse_fusaka_2025",
)

def sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


@st.cache_data(show_spinner=False)
def read_parquet(path: str) -> list[dict[str, Any]]:
    connection = duckdb.connect()
    cursor = connection.execute(f"SELECT * FROM read_parquet({sql_path(Path(path))})")
    columns = [item[0] for item in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    connection.close()
    return rows


@st.cache_data(show_spinner=False)
def read_csv_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["audit_id"]: row for row in csv.DictReader(stream)}


@st.cache_data(show_spinner=False)
def parquet_count(path: str) -> int:
    connection = duckdb.connect()
    count = int(connection.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_path(Path(path))})"
    ).fetchone()[0])
    connection.close()
    return count


def normalized_deltas(event: dict[str, Any], metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    deltas = json.loads(event.get("attacker_token_deltas_raw_json") or "{}")
    rows = []
    for token, raw in deltas.items():
        meta = metadata.get(token, {})
        decimals = meta.get("decimals")
        normalized = None
        if decimals is not None:
            normalized = format(Decimal(str(raw)) / (Decimal(10) ** int(decimals)), "f")
        rows.append(
            {
                "token": token,
                "símbolo": meta.get("symbol"),
                "decimais": decimals,
                "delta_bruto": str(raw),
                "delta_normalizado": normalized,
                "metadados": meta.get("metadata_status", "não_coletado"),
            }
        )
    return sorted(rows, key=lambda row: (str(row["símbolo"]), row["token"]))


def transaction_panel(title: str, tx_hash: str | None, transactions: dict[str, dict[str, Any]]) -> None:
    st.subheader(title)
    if not tx_hash:
        st.info("Papel não aplicável a este detector.")
        return
    transaction = transactions.get(tx_hash.lower())
    st.link_button("Abrir no Etherscan", f"https://etherscan.io/tx/{tx_hash}")
    if not transaction:
        st.warning("Transação ausente do enriquecimento RPC.")
        return
    input_data = transaction.get("input_data") or "0x"
    st.write(
        {
            "hash": tx_hash,
            "from": transaction.get("from_address"),
            "to": transaction.get("to_address"),
            "índice": int(str(transaction.get("transaction_index_hex")), 16),
            "status": int(str(transaction.get("receipt_status_hex")), 16),
            "seletor": input_data[:10] if len(input_data) >= 10 else None,
            "gas usado": int(str(transaction.get("gas_used_hex")), 16),
        }
    )
    with st.expander("Input Data bruto"):
        st.code(input_data, language=None, wrap_lines=True)
    with st.expander("Logs do receipt"):
        try:
            st.json(json.loads(transaction.get("logs_json") or "[]"))
        except json.JSONDecodeError:
            st.code(transaction.get("logs_json") or "[]", language="json")


def main() -> None:
    st.set_page_config(page_title="Dossiê semântico de front-running", layout="wide")
    st.title("Auditoria semântica dos candidatos de front-running")
    split_dirs = sorted(
        (
            path for path in RESULTS_ROOT.iterdir()
            if path.is_dir() and (path / "08_dossie_auditoria.parquet").is_file()
        ),
        key=lambda path: SPLIT_ORDER.index(path.name)
        if path.name in SPLIT_ORDER else len(SPLIT_ORDER),
    ) if RESULTS_ROOT.is_dir() else []
    if not split_dirs:
        st.error("Nenhuma janela com dossiê foi encontrada.")
        st.stop()
    split_names = [path.name for path in split_dirs]
    global_total = sum(parquet_count(str(path / "08_dossie_auditoria.parquet")) for path in split_dirs)
    global_adjudicated = sum(len(load_decisions(path / "15_decisoes_dossie.csv")) for path in split_dirs)
    st.caption(
        f"Cobertura da adjudicação assistida: {global_adjudicated:,}/{global_total:,} eventos "
        f"({100 * global_adjudicated / max(global_total, 1):.1f}%)."
    )
    st.progress(global_adjudicated / max(global_total, 1))

    split_state = "selected_split"
    if st.session_state.get(split_state) not in split_names:
        st.session_state[split_state] = split_names[0]
    split_index = split_names.index(st.session_state[split_state])

    def move_window(offset: int) -> None:
        current = split_names.index(st.session_state[split_state])
        target = max(0, min(len(split_names) - 1, current + offset))
        st.session_state[split_state] = split_names[target]

    window_prev, window_selector, window_next = st.columns([1, 8, 1])
    window_prev.button(
        "← Janela", disabled=split_index == 0, width="stretch",
        on_click=move_window, args=(-1,),
    )
    selected_split = window_selector.selectbox(
        "Janela temporal", split_names, key=split_state
    )
    split_index = split_names.index(selected_split)
    window_next.button(
        "Janela →", disabled=split_index == len(split_names) - 1, width="stretch",
        on_click=move_window, args=(1,),
    )
    results = RESULTS_ROOT / selected_split
    dossier_path = results / "08_dossie_auditoria.parquet"
    flows_path = results / "09_fluxos_tokens_atacante.parquet"
    transactions_path = results / "01_transacoes_rpc.parquet"
    metadata_path = results / "12_metadados_tokens.parquet"
    assisted_path = results / "15_decisoes_dossie.csv"
    validation_path = results / "17_validacao_adjudicacoes.csv"
    required = [dossier_path, flows_path, transactions_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        st.error("Artefatos obrigatórios ausentes:\n" + "\n".join(missing))
        st.stop()

    dossier = read_parquet(str(dossier_path))
    flows = read_parquet(str(flows_path))
    transaction_rows = read_parquet(str(transactions_path))
    transactions = {str(row["tx_hash"]).lower(): row for row in transaction_rows}
    metadata_rows = read_parquet(str(metadata_path)) if metadata_path.is_file() else []
    metadata = {str(row["token_address"]).lower(): row for row in metadata_rows}
    decisions = load_decisions(assisted_path)
    validations = {
        str(row["audit_id"]): row for row in read_csv_rows(str(validation_path))
    } if validation_path.is_file() else {}

    st.caption(
        f"{len(decisions)} adjudicações assistidas para {len(dossier)} eventos nesta janela."
    )
    st.progress(len(decisions) / max(len(dossier), 1))
    if not metadata_rows:
        st.warning("Metadados dos tokens ainda não foram coletados; deltas normalizados podem ficar indisponíveis.")

    c1, c2, c3, c4 = st.columns(4)
    detector_filter = c1.selectbox("Detector", ["todos"] + sorted({str(row["detector"]) for row in dossier}))
    semantic_filter = c2.selectbox("Validação semântica", ["todos"] + sorted({str(row["validation_status"]) for row in dossier}))
    economic_filter = c3.selectbox("Situação econômica", ["todos"] + sorted({str(row["economic_status"]) for row in dossier}))
    validation_options = ["todos"] + sorted({str(row.get("validation_result")) for row in validations.values()})
    validation_filter = c4.selectbox("Validação automática", validation_options)
    c5, c6, c7 = st.columns(3)
    decision_options = ["todos"] + sorted({str(row.get("manual_label")) for row in decisions.values()})
    decision_filter = c5.selectbox("Decisão assistida", decision_options)
    quality_options = ["todos"] + sorted({str(row.get("evidence_quality")) for row in decisions.values()})
    quality_filter = c6.selectbox("Qualidade", quality_options)
    attention_filter = c7.selectbox(
        "Fila de atenção", ["todos", "prioritários", "conflitos amplos"]
    )
    search_text = st.text_input(
        "Busca direta",
        placeholder="audit_id, detection_event_id, hash, endereço atacante ou pool",
    ).strip().lower()

    conflicts_path = RESULTS_ROOT / "consolidado" / "04_conflitos_rotulos.csv"
    conflict_hashes = {
        row["tx_hash"].lower() for row in read_csv_rows(str(conflicts_path))
        if row.get("split") == selected_split
    } if conflicts_path.is_file() else set()

    def matches_search(row: dict[str, Any]) -> bool:
        if not search_text:
            return True
        values = (
            row.get("audit_id"), row.get("detection_event_id"),
            row.get("attacker_front_hash"), row.get("attacker_back_hash"),
            row.get("victim_hash"), row.get("attacker_address"),
            row.get("attacker_executor_address"), row.get("pool_addresses"),
        )
        return any(search_text in str(value).lower() for value in values if value is not None)

    def requires_attention(row: dict[str, Any]) -> bool:
        aid = str(row["audit_id"])
        decision = decisions.get(aid, {})
        validation = validations.get(aid, {})
        return (
            row.get("validation_status") == "probable"
            or decision.get("evidence_quality") == "baixa"
            or validation.get("validation_result") not in (None, "valid")
            or row.get("economic_status") in {"incomplete", "unknown", "sem_valoração"}
        )

    def has_conflict(row: dict[str, Any]) -> bool:
        hashes = {
            str(value).lower() for value in (
                row.get("attacker_front_hash"), row.get("attacker_back_hash"), row.get("victim_hash")
            ) if value
        }
        return bool(hashes & conflict_hashes)

    filtered = [
        row for row in dossier
        if (detector_filter == "todos" or row["detector"] == detector_filter)
        and (semantic_filter == "todos" or row["validation_status"] == semantic_filter)
        and (economic_filter == "todos" or row["economic_status"] == economic_filter)
        and (validation_filter == "todos" or validations.get(str(row["audit_id"]), {}).get("validation_result") == validation_filter)
        and (decision_filter == "todos" or decisions.get(str(row["audit_id"]), {}).get("manual_label") == decision_filter)
        and (quality_filter == "todos" or decisions.get(str(row["audit_id"]), {}).get("evidence_quality") == quality_filter)
        and (attention_filter != "prioritários" or requires_attention(row))
        and (attention_filter != "conflitos amplos" or has_conflict(row))
        and matches_search(row)
    ]
    filtered.sort(
        key=lambda row: (
            int(row.get("audit_priority") or 999),
            int(row["detection_event_id"]),
        )
    )
    if not filtered:
        st.info("Nenhum evento corresponde aos filtros.")
        st.stop()

    event_ids = [str(row["audit_id"]) for row in filtered]
    state_key = f"selected_event__{selected_split}"
    if st.session_state.get(state_key) not in event_ids:
        st.session_state[state_key] = event_ids[0]
    current_index = event_ids.index(st.session_state[state_key])
    st.caption(f"Evento {current_index + 1:,} de {len(event_ids):,} na seleção atual.")
    previous_col, selector_col, next_col = st.columns([1, 8, 1])
    if previous_col.button("← Anterior", disabled=current_index == 0, width="stretch"):
        st.session_state[state_key] = event_ids[current_index - 1]; st.rerun()
    selected_id = selector_col.selectbox(
        "Evento", event_ids, index=current_index,
        format_func=lambda value: (
            f"{value} · prioridade "
            f"{next(row['audit_priority'] for row in filtered if row['audit_id'] == value)}"
        ),
    )
    st.session_state[state_key] = selected_id
    current_index = event_ids.index(selected_id)
    if next_col.button("Próximo →", disabled=current_index == len(event_ids)-1, width="stretch"):
        st.session_state[state_key] = event_ids[current_index + 1]; st.rerun()
    event = next(row for row in filtered if row["audit_id"] == selected_id)
    existing = decisions.get(selected_id, {})
    validation = validations.get(selected_id, {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prioridade", int(event["audit_priority"]))
    m2.metric("Detector", str(event["detector"]))
    m3.metric("Validação semântica", str(event["validation_status"]))
    m4.metric("Situação econômica", str(event["economic_status"]))
    st.info(str(event["semantic_evidence"]))
    st.write({
        "decisão assistida": existing.get("manual_label"),
        "tipo adjudicado": existing.get("manual_type"),
        "qualidade da evidência": existing.get("evidence_quality"),
        "validação automática": validation.get("validation_result", "não executada"),
        "alertas": validation.get("issues") or "nenhum",
    })
    if has_conflict(event):
        st.warning(
            "Este evento contém hash com papéis contraditórios no cenário de sensibilidade. "
            "A consolidação preservou o caso, mas não atribuiu rótulo transacional amplo ao hash."
        )
    st.write(
        {
            "protocolo": event.get("protocols"),
            "pool": event.get("pool_addresses"),
            "EOA atacante": event.get("attacker_address"),
            "executor atacante": event.get("attacker_executor_address"),
            "custo de gás (ETH)": event.get("attacker_gas_cost_eth"),
            "WETH após gás": event.get("weth_after_gas_eth"),
        }
    )

    st.subheader("Deltas de tokens observáveis")
    delta_rows = normalized_deltas(event, metadata)
    if delta_rows:
        st.dataframe(delta_rows, width="stretch", hide_index=True)
    else:
        st.info("Nenhum delta ERC-20 diretamente associado à entidade atacante.")

    event_flows = [row for row in flows if int(row["detection_event_id"]) == int(event["detection_event_id"])]
    with st.expander(f"Transferências ERC-20 das pernas externas ({len(event_flows)})"):
        st.dataframe(event_flows, width="stretch", hide_index=True)

    tabs = st.tabs(["Atacante — frente", "Vítima", "Atacante — retorno"])
    with tabs[0]:
        transaction_panel("Perna frontal", event.get("attacker_front_hash"), transactions)
    with tabs[1]:
        transaction_panel("Vítima", event.get("victim_hash"), transactions)
    with tabs[2]:
        transaction_panel("Perna de retorno", event.get("attacker_back_hash"), transactions)

    st.subheader("Adjudicação assistida — somente leitura")
    st.write(
        {
            "conclusão": existing.get("manual_label"),
            "tipo": existing.get("manual_type"),
            "qualidade": existing.get("evidence_quality"),
            "evidências": existing.get("reason_codes"),
            "responsável registrado": existing.get("reviewer"),
            "data UTC": existing.get("reviewed_at_utc"),
        }
    )
    st.text_area(
        "Justificativa produzida pelo adjudicador",
        value=existing.get("notes", ""),
        height=160,
        disabled=True,
        key=f"notes__{selected_split}__{selected_id}",
    )


if __name__ == "__main__":
    main()
