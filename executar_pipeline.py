# /// script
# requires-python = ">=3.11"
# ///

"""Executa o experimento em uma raiz isolada e retomável."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from configuracao import (
    ConfiguracaoPipelineError,
    carregar_amostragem_blocos,
    carregar_configuracao_execucao,
    carregar_documento_pipeline,
    validar_destino_execucao,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configuracao" / "pipeline.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def config_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nested(document: dict[str, object], section: str, field: str) -> object:
    try:
        return document[section][field]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ConfiguracaoPipelineError(
            f"Configuração inválida: esperado {section}.{field}."
        ) from exc


def relative_project_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfiguracaoPipelineError(f"{field} deve ser um texto não vazio.")
    path = Path(value)
    if path.is_absolute():
        raise ConfiguracaoPipelineError(f"{field} deve ser relativo à raiz do projeto.")
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ConfiguracaoPipelineError(f"{field} aponta para fora do projeto.") from exc
    return resolved


@dataclass(frozen=True)
class Context:
    config: Path
    run: Path
    history: Path
    cache: Path
    temp: Path
    memory: str
    threads: int
    percentage: float
    modulus: int
    labelcloud_csv: Path
    rpc_regression: Path
    uv: str
    dotnet: str

    @property
    def features(self) -> Path:
        return self.run / "03-machine-learning" / "01-features"

    @property
    def ae(self) -> Path:
        return self.run / "03-machine-learning" / "02-ae" / "resultados"

    @property
    def isolation_forest(self) -> Path:
        return self.run / "03-machine-learning" / "03-if" / "resultados"

    @property
    def analysis(self) -> Path:
        return self.run / "05-analise-resultados" / "resultados"

    @property
    def labelcloud(self) -> Path:
        return self.run / "04-pos-processamento" / "resultados-labelcloud"

    @property
    def csharp(self) -> Path:
        return (
            self.run
            / "04-pos-processamento"
            / "05-rotulador-front-running-csharp"
            / "resultados-v3"
        )

    @property
    def csharp_windows(self) -> Path:
        return self.csharp / "sampled" / "independent_sampled"

    @property
    def semantic(self) -> Path:
        return (
            self.run
            / "04-pos-processamento"
            / "06-validacao-semantica-front-running"
            / "resultados"
        )


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: tuple[str, ...]
    marker: Path
    needs_rpc: bool = False
    always_run: bool = False


def build_context(config_path: Path) -> Context:
    execution = carregar_configuracao_execucao(config_path)
    document = carregar_documento_pipeline(config_path)
    percentage, modulus = carregar_amostragem_blocos(config_path)

    memory = nested(document, "recursos", "memory_limit")
    threads = nested(document, "recursos", "threads")
    temp_dir = nested(document, "recursos", "temp_dir")
    if not isinstance(memory, str) or not memory.strip():
        raise ConfiguracaoPipelineError("recursos.memory_limit deve ser um texto.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ConfiguracaoPipelineError("recursos.threads deve ser um inteiro positivo.")
    if not isinstance(temp_dir, str) or not temp_dir.strip():
        raise ConfiguracaoPipelineError("recursos.temp_dir deve ser um texto.")
    temp_path = Path(temp_dir)
    if temp_path.is_absolute() or ".." in temp_path.parts:
        raise ConfiguracaoPipelineError(
            "recursos.temp_dir deve ser relativo à raiz da execução."
        )

    uv = shutil.which("uv")
    dotnet = shutil.which("dotnet")
    if not uv:
        raise ConfiguracaoPipelineError("Executável uv não encontrado no PATH.")
    if not dotnet:
        raise ConfiguracaoPipelineError("Executável dotnet não encontrado no PATH.")

    return Context(
        config=config_path,
        run=execution.diretorio_execucao,
        history=execution.diretorio_historico,
        cache=execution.diretorio_cache,
        temp=(execution.diretorio_execucao / temp_path).resolve(),
        memory=memory.strip(),
        threads=threads,
        percentage=percentage,
        modulus=modulus,
        labelcloud_csv=relative_project_path(
            nested(document, "dados_auxiliares", "labelcloud_csv"),
            "dados_auxiliares.labelcloud_csv",
        ),
        rpc_regression=relative_project_path(
            nested(document, "dados_auxiliares", "regressao_rpc"),
            "dados_auxiliares.regressao_rpc",
        ),
        uv=uv,
        dotnet=dotnet,
    )


def stages(context: Context) -> list[Stage]:
    py = (context.uv, "run")
    config = str(context.config)
    temp = str(context.temp)
    memory = context.memory
    threads = str(context.threads)
    features = context.features
    split_manifest = features / "resultados-splits" / "00_manifest_splits_temporais.json"
    preprocessing = features / "resultados-preprocessamento"
    common = ("--temp-dir", temp, "--memory-limit", memory, "--threads", threads)
    semantic_scripts = ROOT / "04-pos-processamento" / "06-validacao-semantica-front-running"
    providers_config = ROOT / "configuracao" / "rpc_providers.json"
    provider_audit = context.semantic / "auditoria-rpc" / "01_manifest_provedores_rpc.json"

    return [
        Stage(
            "auditoria_features",
            "Auditoria de esquema, features e correlações",
            (*py, str(ROOT / "03-machine-learning/01-features/01_auditoria_selecao_features.py"), "--config", config, "--output-dir", str(features / "resultados"), *common),
            features / "resultados" / "15_resultado_execucao.json",
        ),
        Stage(
            "figuras_correlacao",
            "Figuras vetoriais das correlações",
            (*py, str(ROOT / "03-machine-learning/01-features/04_gerar_figuras_correlacao_pdf.py"), "--results-dir", str(features / "resultados")),
            features / "resultados" / "18_matrizes_correlacao_vetorial.pdf",
        ),
        Stage(
            "matrizes",
            "Matrizes A, B e C do treinamento",
            (*py, str(ROOT / "03-machine-learning/01-features/02_gerar_matrizes_features.py"), "--config", config, "--start-date", "2024-04-01", "--end-date", "2024-12-31", "--dataset-name", "treino_2024_pos_dencun", "--output-dir", str(features / "resultados-matrizes"), *common),
            features / "resultados-matrizes" / "10_manifest_execucao.json",
        ),
        Stage(
            "splits",
            "Sete janelas temporais",
            (*py, str(ROOT / "03-machine-learning/01-features/03_gerar_splits_temporais.py"), "--config", config, "--output-root", str(features / "resultados-splits"), "--training-dir", str(features / "resultados-matrizes"), *common),
            split_manifest,
        ),
        Stage(
            "preprocessamento",
            "RobustScaler ajustado somente no treinamento",
            (*py, str(ROOT / "03-machine-learning/01-features/05_preprocessar_matrizes.py"), "--split-manifest", str(split_manifest), "--output-dir", str(preprocessing), *common),
            preprocessing / "05_manifest_preprocessamento.json",
        ),
        Stage(
            "drift",
            "Análise de drift temporal",
            (*py, str(ROOT / "03-machine-learning/01-features/06_analisar_drift_temporal.py"), "--split-manifest", str(split_manifest), "--preprocessing-manifest", str(preprocessing / "05_manifest_preprocessamento.json"), "--transformation-dictionary", str(preprocessing / "02_dicionario_transformacoes.csv"), "--output-dir", str(features / "resultados-drift"), *common),
            features / "resultados-drift" / "10_manifest_drift.json",
        ),
        Stage(
            "isolation_forest",
            "Treinamento e inferência do Isolation Forest",
            (*py, str(ROOT / "03-machine-learning/03-if/01_treinar_isolation_forest.py"), "--split-manifest", str(split_manifest), "--preprocessing-manifest", str(preprocessing / "05_manifest_preprocessamento.json"), "--output-dir", str(context.isolation_forest), *common),
            context.isolation_forest / "10_manifest_isolation_forest.json",
        ),
        Stage(
            "autoencoder",
            "Treinamento e inferência do Autoencoder",
            (*py, str(ROOT / "03-machine-learning/02-ae/01_treinar_autoencoder.py"), "--split-manifest", str(split_manifest), "--preprocessing-manifest", str(preprocessing / "05_manifest_preprocessamento.json"), "--if-results-dir", str(context.isolation_forest), "--output-dir", str(context.ae), "--device", "auto", *common),
            context.ae / "13_manifest_autoencoder.json",
        ),
        Stage(
            "analise_ae_if",
            "Integração dos escores AE e IF",
            (*py, str(ROOT / "05-analise-resultados/01_analisar_ae_if.py"), "--split-manifest", str(split_manifest), "--ae-results-dir", str(context.ae), "--if-results-dir", str(context.isolation_forest), "--output-dir", str(context.analysis), *common),
            context.analysis / "09_manifest_analise_integrada.json",
        ),
        Stage(
            "labelcloud_preparar",
            "Normalização do snapshot Label Cloud",
            (*py, str(ROOT / "04-pos-processamento/02_preparar_labelcloud_historico.py"), "--input-csv", str(context.labelcloud_csv), "--output-dir", str(context.labelcloud)),
            context.labelcloud / "04_manifest_labelcloud_historico.json",
        ),
        Stage(
            "labelcloud_vincular",
            "Vínculo do Label Cloud aos candidatos",
            (*py, str(ROOT / "04-pos-processamento/03_vincular_labelcloud_candidatos.py"), "--tags", str(context.labelcloud / "01_labelcloud_historico_normalizado.parquet"), "--candidates-dir", str(context.analysis / "candidatos"), "--output-dir", str(context.labelcloud / "vinculos-candidatos"), *common),
            context.labelcloud / "vinculos-candidatos" / "00_manifest_vinculos.json",
        ),
        Stage(
            "labelcloud_rotulos",
            "Rótulos hipotéticos do Label Cloud",
            (*py, str(ROOT / "04-pos-processamento/04_gerar_rotulos_hipoteticos_labelcloud.py"), "--tags", str(context.labelcloud / "01_labelcloud_historico_normalizado.parquet"), "--split-manifest", str(split_manifest), "--output-dir", str(context.labelcloud), *common),
            context.labelcloud / "07_manifest_rotulos_hipoteticos.json",
        ),
        Stage(
            "labelcloud_avaliar",
            "Métricas hipotéticas com Label Cloud",
            (*py, str(ROOT / "05-analise-resultados/03_avaliar_labelcloud_hipotetico.py"), "--labels", str(context.labelcloud / "05_rotulos_binarios_hipoteticos_labelcloud.parquet"), "--ae-results-dir", str(context.ae), "--if-results-dir", str(context.isolation_forest), "--split-manifest", str(split_manifest), "--output-dir", str(context.run / "05-analise-resultados/resultados-labelcloud-hipotetico"), *common),
            context.run / "05-analise-resultados/resultados-labelcloud-hipotetico/04_manifest_avaliacao.json",
        ),
        Stage(
            "candidatos_csharp",
            "Candidatos estruturais de front-running",
            (context.dotnet, "run", "--project", str(ROOT / "04-pos-processamento/05-rotulador-front-running-csharp"), "--configuration", "Release", "--", "--input", str(features), "--input-mode", "sampled", "--config", config, "--output-dir", str(context.csharp), "--detectors", "all", "--temp-dir", temp, "--memory-limit", memory, "--threads", threads),
            context.csharp_windows / "teste_final_2025" / "04_manifest_run.json",
        ),
        Stage(
            "auditoria_csharp",
            "Auditoria das sete saídas do rotulador C#",
            (*py, str(semantic_scripts / "07_auditar_saidas_csharp.py"), "--csharp-root", str(context.csharp_windows), "--output-dir", str(context.semantic / "auditoria-csharp")),
            context.semantic / "auditoria-csharp" / "02_manifest_auditoria_csharp.json",
        ),
        Stage(
            "auditoria_rpc",
            "Homologação dos provedores RPC configurados",
            (*py, str(semantic_scripts / "15_testar_provedores_rpc.py"), "--providers-config", str(providers_config), "--output", str(provider_audit)),
            provider_audit,
            always_run=True,
        ),
        Stage(
            "validacao_semantica",
            "Enriquecimento RPC concorrente, validação e adjudicação",
            (*py, str(semantic_scripts / "16_executar_pipeline_janelas_paralelo.py"), "--csharp-root", str(context.csharp_windows), "--results-root", str(context.semantic), "--cache-root", str(context.cache), "--audit-manifest", str(context.semantic / "auditoria-csharp/02_manifest_auditoria_csharp.json"), "--regression-manifest", str(context.rpc_regression), "--provider-audit", str(provider_audit), "--providers-config", str(providers_config)),
            context.semantic / "00_manifest_pipeline_paralelo.json",
        ),
        Stage(
            "consolidacao_semantica",
            "Consolidação dos rótulos semânticos",
            (*py, str(semantic_scripts / "12_consolidar_rotulos_semanticos.py"), "--results-root", str(context.semantic), "--output-dir", str(context.semantic / "consolidado")),
            context.semantic / "consolidado" / "06_manifest_consolidacao.json",
        ),
        Stage(
            "avaliacao_semantica",
            "Avaliação final de AE e IF com os rótulos semânticos",
            (*py, str(ROOT / "05-analise-resultados/02_avaliar_rotulos_semanticos.py"), "--labels", str(context.semantic / "consolidado/03_rotulos_transacao.parquet"), "--event-roles", str(context.semantic / "consolidado/02_papeis_evento.parquet"), "--ae-results-dir", str(context.ae), "--if-results-dir", str(context.isolation_forest), "--split-manifest", str(split_manifest), "--output-dir", str(context.run / "05-analise-resultados/resultados-rotulos-semanticos"), *common),
            context.run / "05-analise-resultados/resultados-rotulos-semanticos/15_manifest_avaliacao.json",
        ),
    ]


def select_stages(all_stages: list[Stage], start: str | None, end: str | None) -> list[Stage]:
    names = [stage.name for stage in all_stages]
    first = names.index(start) if start else 0
    last = names.index(end) if end else len(names) - 1
    if first > last:
        raise ConfiguracaoPipelineError("--de deve anteceder ou coincidir com --ate.")
    return all_stages[first : last + 1]


def initial_state(context: Context, fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_id": context.run.name,
        "configuration_sha256": fingerprint,
        "sampling_percentage": context.percentage,
        "block_modulus": context.modulus,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "running",
        "stages": {},
    }


def load_or_create_state(context: Context, dry_run: bool) -> tuple[dict[str, Any], Path]:
    state_path = context.run / "00_manifest_multiexecucao.json"
    fingerprint = config_fingerprint(context.config)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration_sha256") != fingerprint:
            raise ConfiguracaoPipelineError(
                "A configuração mudou após o início desta execução. Use um novo "
                "execucao.id para preservar a comparabilidade."
            )
        return state, state_path
    if context.run.exists() and any(context.run.iterdir()):
        if dry_run:
            return initial_state(context, fingerprint), state_path
        validar_destino_execucao(carregar_configuracao_execucao(context.config))
    state = initial_state(context, fingerprint)
    if not dry_run:
        context.run.mkdir(parents=True, exist_ok=False)
        context.temp.mkdir(parents=True, exist_ok=True)
        context.cache.mkdir(parents=True, exist_ok=True)
        write_json_atomic(state_path, state)
    return state, state_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--listar", action="store_true", help="Lista as etapas e encerra.")
    parser.add_argument("--de", dest="start", help="Primeira etapa da faixa.")
    parser.add_argument("--ate", dest="end", help="Última etapa da faixa.")
    parser.add_argument("--dry-run", action="store_true", help="Exibe comandos sem gravar arquivos.")
    args = parser.parse_args()

    try:
        context = build_context(args.config.expanduser().resolve())
        all_stages = stages(context)
        names = [stage.name for stage in all_stages]
        if args.start and args.start not in names:
            raise ConfiguracaoPipelineError(f"Etapa desconhecida em --de: {args.start}")
        if args.end and args.end not in names:
            raise ConfiguracaoPipelineError(f"Etapa desconhecida em --ate: {args.end}")
        selected = select_stages(all_stages, args.start, args.end)
    except ConfiguracaoPipelineError as exc:
        parser.error(str(exc))

    if args.listar:
        for index, stage in enumerate(all_stages, start=1):
            print(f"{index:02d}. {stage.name}: {stage.description}")
        return 0

    try:
        state, state_path = load_or_create_state(context, args.dry_run)
    except (OSError, ValueError, ConfiguracaoPipelineError) as exc:
        parser.error(str(exc))

    print(f"Execução: {context.run.name}")
    print(f"Amostragem: {context.percentage:g}% (módulo {context.modulus})")
    print(f"Resultados: {context.run}")
    print(f"Cache compartilhado: {context.cache}")

    for stage in selected:
        previous = state["stages"].get(stage.name, {})
        if previous.get("status") == "completed" and not stage.always_run:
            if not stage.marker.is_file():
                parser.error(
                    f"A etapa {stage.name} consta como concluída, mas seu marcador "
                    f"não existe: {stage.marker}"
                )
            print(f"[IGNORADA] {stage.name}: já concluída e verificada.")
            continue
        if (
            stage.needs_rpc
            and not args.dry_run
            and not os.environ.get("ETH_RPC_URL", "").strip()
        ):
            parser.error(
                "ETH_RPC_URL deve estar definida antes da etapa validacao_semantica."
            )
        print(f"\n[{stage.name}] {stage.description}")
        print(subprocess.list2cmdline(stage.command))
        if args.dry_run:
            continue

        started = time.monotonic()
        state["stages"][stage.name] = {
            "status": "running",
            "started_at_utc": utc_now(),
            "command": list(stage.command),
            "marker": str(stage.marker),
        }
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        try:
            subprocess.run(stage.command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            state["stages"][stage.name].update(
                {
                    "status": "failed",
                    "return_code": exc.returncode,
                    "completed_at_utc": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            state["status"] = "failed"
            state["updated_at_utc"] = utc_now()
            write_json_atomic(state_path, state)
            return exc.returncode or 1
        if not stage.marker.is_file():
            state["stages"][stage.name].update(
                {
                    "status": "failed_missing_marker",
                    "completed_at_utc": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
            )
            state["status"] = "failed"
            state["updated_at_utc"] = utc_now()
            write_json_atomic(state_path, state)
            print(f"Marcador esperado não encontrado: {stage.marker}", file=sys.stderr)
            return 1
        state["stages"][stage.name].update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        state["status"] = "running"
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)

    if not args.dry_run and selected[-1].name == all_stages[-1].name:
        state["status"] = "completed"
        state["completed_at_utc"] = utc_now()
        state["updated_at_utc"] = utc_now()
        write_json_atomic(state_path, state)
        summary = {
            "execution_id": context.run.name,
            "status": state["status"],
            "sampling_percentage": context.percentage,
            "block_modulus": context.modulus,
            "configuration_sha256": state["configuration_sha256"],
            "completed_at_utc": state["completed_at_utc"],
            "execution_manifest": str(state_path.relative_to(ROOT)),
        }
        write_json_atomic(context.history / f"{context.run.name}.json", summary)
    print("\nExecução planejada concluída.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
