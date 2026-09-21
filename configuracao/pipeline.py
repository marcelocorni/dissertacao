"""Utilitários para a configuração central do experimento."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Sequence


class ConfiguracaoPipelineError(ValueError):
    """Indica configuração ausente ou inválida."""


def _carregar_documento(caminho: Path) -> dict[str, object]:
    if not caminho.is_file():
        raise ConfiguracaoPipelineError(
            f"Arquivo de configuração não encontrado: {caminho}"
        )
    try:
        documento = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfiguracaoPipelineError(
            f"Arquivo de configuração inválido: {caminho}"
        ) from exc
    if not isinstance(documento, dict):
        raise ConfiguracaoPipelineError(
            f"A raiz da configuração deve ser um objeto JSON: {caminho}"
        )
    return documento


def _caminho_solicitado(
    caminho_padrao: Path,
    argumentos: Sequence[str] | None = None,
) -> Path:
    tokens = list(sys.argv[1:] if argumentos is None else argumentos)
    for indice, token in enumerate(tokens):
        if token == "--config":
            if indice + 1 >= len(tokens) or tokens[indice + 1].startswith("--"):
                raise ConfiguracaoPipelineError("Falta um valor para --config.")
            return Path(tokens[indice + 1]).expanduser().resolve()
        if token.startswith("--config="):
            valor = token.partition("=")[2].strip()
            if not valor:
                raise ConfiguracaoPipelineError("Falta um valor para --config.")
            return Path(valor).expanduser().resolve()
    return caminho_padrao.resolve()


def carregar_root_template(
    caminho_padrao: Path,
    argumentos: Sequence[str] | None = None,
) -> tuple[Path, str]:
    """Retorna o arquivo usado e o modelo da raiz dos Parquets."""
    caminho = _caminho_solicitado(caminho_padrao, argumentos)
    try:
        documento = _carregar_documento(caminho)
        root_template = documento["dados"]["root_template"]
    except (KeyError, TypeError) as exc:
        raise ConfiguracaoPipelineError(
            f"Configuração inválida em {caminho}: esperado dados.root_template."
        ) from exc
    if not isinstance(root_template, str) or not root_template.strip():
        raise ConfiguracaoPipelineError(
            f"dados.root_template deve ser um texto não vazio em {caminho}."
        )
    if "{year}" not in root_template:
        raise ConfiguracaoPipelineError(
            f"dados.root_template deve conter {{year}} em {caminho}."
        )
    return caminho, root_template


def carregar_amostragem_blocos(caminho: Path) -> tuple[float, int]:
    """Retorna o percentual configurado e o módulo determinístico equivalente.

    A regra histórica seleciona ``block_number % modulo = 0``. Para preservar
    exatamente essa regra, o percentual deve ser representável como ``100/N``.
    """
    try:
        documento = _carregar_documento(caminho.resolve())
        percentual = documento["amostragem"]["percentual_blocos"]
    except (KeyError, TypeError) as exc:
        raise ConfiguracaoPipelineError(
            f"Configuração inválida em {caminho}: esperado "
            "amostragem.percentual_blocos."
        ) from exc
    if isinstance(percentual, bool) or not isinstance(percentual, (int, float)):
        raise ConfiguracaoPipelineError(
            "amostragem.percentual_blocos deve ser numérico."
        )
    percentual = float(percentual)
    if not math.isfinite(percentual) or not 0 < percentual <= 100:
        raise ConfiguracaoPipelineError(
            "amostragem.percentual_blocos deve estar no intervalo (0, 100]."
        )
    modulo_exato = 100.0 / percentual
    modulo = round(modulo_exato)
    if not math.isclose(modulo_exato, modulo, rel_tol=0, abs_tol=1e-9):
        raise ConfiguracaoPipelineError(
            "amostragem.percentual_blocos deve ser representável como 100/N "
            "para preservar a seleção determinística (ex.: 0.5, 1, 2, 5, 10, "
            "20, 25, 50 ou 100)."
        )
    return percentual, int(modulo)
