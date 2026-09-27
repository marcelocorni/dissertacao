"""Utilitários para a configuração central do experimento."""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class ConfiguracaoPipelineError(ValueError):
    """Indica configuração ausente ou inválida."""


@dataclass(frozen=True)
class ConfiguracaoExecucao:
    """Diretórios isolados da execução e cache reutilizável do pipeline."""

    identificador: str
    raiz_projeto: Path
    diretorio_execucao: Path
    diretorio_historico: Path
    diretorio_cache: Path
    cache_rpc_blocks: Path
    cache_rpc_transactions: Path
    cache_token_metadata: Path
    proteger_resultados_existentes: bool


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


def carregar_documento_pipeline(caminho: Path) -> dict[str, object]:
    """Retorna a configuração JSON validada como objeto."""
    return _carregar_documento(caminho.resolve())


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


def _texto_obrigatorio(documento: dict[str, object], *chaves: str) -> str:
    valor: object = documento
    try:
        for chave in chaves:
            valor = valor[chave]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        caminho = ".".join(chaves)
        raise ConfiguracaoPipelineError(
            f"Configuração inválida: esperado {caminho}."
        ) from exc
    if not isinstance(valor, str) or not valor.strip():
        caminho = ".".join(chaves)
        raise ConfiguracaoPipelineError(f"{caminho} deve ser um texto não vazio.")
    return valor.strip()


def _resolver_caminho_interno(raiz: Path, valor: str, campo: str) -> Path:
    caminho = Path(valor)
    if caminho.is_absolute():
        raise ConfiguracaoPipelineError(
            f"{campo} deve ser relativo à raiz do projeto: {valor}"
        )
    resolvido = (raiz / caminho).resolve()
    try:
        resolvido.relative_to(raiz)
    except ValueError as exc:
        raise ConfiguracaoPipelineError(
            f"{campo} aponta para fora da raiz do projeto: {valor}"
        ) from exc
    return resolvido


def carregar_configuracao_execucao(caminho: Path) -> ConfiguracaoExecucao:
    """Carrega os destinos isolados da execução e o cache compartilhado.

    A função apenas valida e resolve caminhos; nenhum diretório é criado e
    nenhum resultado existente é alterado.
    """
    caminho = caminho.resolve()
    documento = _carregar_documento(caminho)
    raiz_projeto = caminho.parent.parent.resolve()

    identificador = _texto_obrigatorio(documento, "execucao", "id")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identificador):
        raise ConfiguracaoPipelineError(
            "execucao.id deve usar somente letras minúsculas, números e hífens."
        )

    raiz_execucoes = _resolver_caminho_interno(
        raiz_projeto,
        _texto_obrigatorio(documento, "execucao", "raiz"),
        "execucao.raiz",
    )
    diretorio_historico = _resolver_caminho_interno(
        raiz_projeto,
        _texto_obrigatorio(documento, "execucao", "historico_raiz"),
        "execucao.historico_raiz",
    )
    diretorio_cache = _resolver_caminho_interno(
        raiz_projeto,
        _texto_obrigatorio(documento, "cache_compartilhado", "raiz"),
        "cache_compartilhado.raiz",
    )

    try:
        proteger = documento["execucao"]["proteger_resultados_existentes"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ConfiguracaoPipelineError(
            "Configuração inválida: esperado "
            "execucao.proteger_resultados_existentes."
        ) from exc
    if not isinstance(proteger, bool):
        raise ConfiguracaoPipelineError(
            "execucao.proteger_resultados_existentes deve ser true ou false."
        )

    def cache(nome: str) -> Path:
        return _resolver_caminho_interno(
            diretorio_cache,
            _texto_obrigatorio(documento, "cache_compartilhado", nome),
            f"cache_compartilhado.{nome}",
        )

    return ConfiguracaoExecucao(
        identificador=identificador,
        raiz_projeto=raiz_projeto,
        diretorio_execucao=(raiz_execucoes / identificador).resolve(),
        diretorio_historico=diretorio_historico,
        diretorio_cache=diretorio_cache,
        cache_rpc_blocks=cache("rpc_blocks"),
        cache_rpc_transactions=cache("rpc_transactions"),
        cache_token_metadata=cache("token_metadata"),
        proteger_resultados_existentes=proteger,
    )


def validar_destino_execucao(configuracao: ConfiguracaoExecucao) -> None:
    """Recusa reutilizar uma execução não vazia quando a proteção está ativa."""
    destino = configuracao.diretorio_execucao
    if (
        configuracao.proteger_resultados_existentes
        and destino.is_dir()
        and next(destino.iterdir(), None) is not None
    ):
        raise ConfiguracaoPipelineError(
            f"A execução '{configuracao.identificador}' já possui resultados em "
            f"{destino}. Escolha outro execucao.id para preservar o histórico."
        )
