"""Entrada independente da configuração canônica e do escalonador RPC."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .core import PROTOCOL, Store, atomic_json, code_provenance, file_hash, grids, validate_destination

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prepare", "search", "fit", "score", "export", "complements", "descriptive", "figures", "reports")


def source_files(source):
    from .dados import RAW, SEMANTIC, SPLIT_FILE, load_source, metadata
    _, splits = load_source(source)
    files = [source / SPLIT_FILE]
    for split in splits:
        directory = Path(split["output_directory"])
        files += [metadata(split), directory / "10_manifest_execucao.json", *[directory / name for name in RAW.values()]]
        files.append(source / "04-pos-processamento/06-validacao-semantica-front-running/resultados" / split["key"] / "07_manifest_validacao.json")
    files += [source / SEMANTIC / name for name in ("02_papeis_evento.parquet", "03_rotulos_transacao.parquet")]
    files += [source / "04-pos-processamento/resultados-labelcloud/05_rotulos_binarios_hipoteticos_labelcloud.parquet"]
    files += [source / "03-machine-learning/01-features/resultados" / name for name in
              ("09_correlacao_pearson.csv", "10_correlacao_spearman.csv", "15_resultado_execucao.json")]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Entrada local ausente; não recuperar por RPC automaticamente: {path}")
    return sorted(set(files))


@contextmanager
def execution_lock(root):
    path = root / ".execution.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "all", *STAGES), default="plan")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--execute", action="store_true", help="Executa a etapa; omitir para somente planejar")
    args = parser.parse_args(argv)
    source = args.source_run.resolve()
    destination = validate_destination(args.output, [source, ROOT / "cache-compartilhado", ROOT / "configuracao"])
    if not source.is_relative_to(ROOT / "execucoes") or destination.parent != ROOT / "execucoes":
        parser.error("Origem e nova saída devem ser execuções distintas sob a raiz execucoes do projeto")
    if destination == source or destination.name == "amostra-05pct":
        parser.error("A execução histórica amostra-05pct é protegida")
    if args.stage != "plan" and not args.execute:
        parser.error("Esta etapa exige --execute; nenhum processamento iniciado")
    files = source_files(source)
    plan = {"source_run": str(source), "output": str(destination), "stage": args.stage,
            "protocol": PROTOCOL, "grid": grids(), "input_files": len(files),
            "input_bytes": sum(p.stat().st_size for p in files), "rpc": "disabled"}
    if args.stage == "plan":
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    # Freeze content, code and environment before any calculation or destination write.
    frozen = {"protocol": PROTOCOL, "grid": grids(), "source_run": str(source),
              "device_requested": args.device,
              "inputs": {str(p): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in files},
              "code": code_provenance(ROOT),
              "versions": {p: importlib.metadata.version(p) for p in
                           ("torch", "numpy", "duckdb", "pyarrow", "scikit-learn", "scipy", "matplotlib", "joblib", "threadpoolctl", "pandas", "pypdf")}}
    store = Store(destination, frozen)
    os.environ["MPLCONFIGDIR"] = str(destination / ".tmp" / "matplotlib")
    from .dados import prepare
    from .fluxo import final_fit, score_all, search
    from .exportacao import export, legacy_reports
    from .complementos import count_reuse_and_types, evaluate
    from .figuras import generate
    from threadpoolctl import threadpool_limits
    functions = {"prepare": lambda: prepare(source, store), "search": lambda: search(source, store, args.device),
                 "fit": lambda: final_fit(source, store, args.device), "score": lambda: score_all(source, store, args.device),
                 "export": lambda: export(source, store), "complements": lambda: evaluate(source, store),
                 "descriptive": lambda: count_reuse_and_types(source, store), "figures": lambda: generate(source, store),
                 "reports": lambda: legacy_reports(source, store)}
    with execution_lock(destination), threadpool_limits(limits=PROTOCOL["threads"]):
        for stage in STAGES if args.stage == "all" else (args.stage,):
            print(f"[revisao] {stage}", flush=True)
            functions[stage]()
            # Not a pipeline-completion flag: a single successful stage is not the whole experiment.
            atomic_json(destination / "last_successful_stage.json", {"stage": stage, "protocol_signature": store.identity})
        if args.stage == "all":
            required = ("prepare", "selection", "exports", "complements", "descriptive", "figures-semantic",
                        "figures-agreement", "external-labelcloud", "drift", "correlation-figures")
            if not all(store.complete(name) for name in required):
                raise ValueError("Há etapas incompletas; não declarar conclusão")
            atomic_json(destination / "execution_completed.json", {"status": "completed", "protocol_signature": store.identity,
                        "audit_human": "pending_separate_workflow", "review_intervals":
                        any(r["status"] != "completed" for r in json.loads((destination / "complements/clustered_intervals.json").read_text(encoding="utf-8")))})
    return 0
