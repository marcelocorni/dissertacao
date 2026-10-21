"""Protocolo, partições e persistência segura (somente biblioteca padrão)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

CONFIGURATIONS = ("a_completa", "b_completa", "c_completa",
                  "b_sem_taxas_absolutas", "c_sem_taxas_absolutas")
PROTOCOL = {
    "version": 1, "sampling_seed": 20261018, "model_seed": 42,
    "block_modulus": 20, "internal_modulus": 10, "search_modulus": 10,
    "if_target_rows": 500000, "max_epochs": 200, "batch_size": 65536,
    "patience": 12, "min_delta": 1e-5, "threads": 4,
    "bootstrap_repetitions": 2000, "bootstrap_seed": 20261018,
    "quantiles": {"q990": .99, "q995": .995, "q999": .999},
    "selection": "mean_five_AP_validation_strict",
    "test_previously_inspected": True, "rpc_allowed": False,
    "paired_conflicts": "exclude_whole_event_approved",
}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def block_priority(number, purpose, seed=20261018):
    if purpose not in ("internal", "ae-search", "if-fit"):
        raise ValueError("Finalidade de amostragem desconhecida")
    text = f"ethereum-mainnet|{int(number)}|{purpose}|{seed}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")


def partition_blocks(block_counts, target=500000):
    if len(block_counts) != len(set(block_counts)) or not block_counts:
        raise ValueError("Lista de blocos vazia ou duplicada")
    if any(int(n) <= 0 for n in block_counts.values()):
        raise ValueError("Contagem de transações inválida")
    internal = sorted(b for b in block_counts if block_priority(b, "internal") % 10 == 0)
    fit = sorted(set(block_counts) - set(internal))
    search = sorted(b for b in block_counts if block_priority(b, "ae-search") % 10 == 0)
    if not internal or not fit or not (set(search) & set(internal)) or not (set(search) & set(fit)):
        raise ValueError("Partição vazia; não trocar a semente automaticamente")
    selected, total = [], 0
    for block in sorted(fit, key=lambda b: (block_priority(b, "if-fit"), b)):
        selected.append(block)
        total += block_counts[block]
        if total >= target:
            break
    if total < target:
        raise ValueError("Treinamento insuficiente para o alvo do IF")
    return {"internal": internal, "fit": fit, "ae_search": search, "if_fit": sorted(selected)}


def grids():
    ae = [dict(hidden_dims=list(hidden), latent_dim=latent, learning_rate=lr, weight_decay=wd)
          for (hidden, latent), lr, wd in product(
              [((16, 8), 2), ((16, 8), 4), ((32, 16), 8)], [.0003, .001], [0., .00001])]
    forest = [dict(n_estimators=n, max_samples=s) for n, s in product([100, 200, 400], [2048, 4096, 8192, 16384])]
    historical = {"AE": dict(hidden_dims=[16, 8], latent_dim=4, learning_rate=.001, weight_decay=.00001),
                  "IF": dict(n_estimators=200, max_samples=8192)}
    result = {}
    for method, candidates in (("AE", ae), ("IF", forest)):
        candidates.sort(key=lambda c: (c != historical[method], json.dumps(c, sort_keys=True)))
        result[method] = [dict(id=f"{method.lower()}-{i:02d}", parameters=c) for i, c in enumerate(candidates)]
    return result


def select_candidate(method, completed):
    expected = grids()[method]
    if set(completed) != {c["id"] for c in expected}:
        raise ValueError("Grade incompleta: nenhum vencedor pode ser publicado")
    ranked = []
    for order, candidate in enumerate(expected):
        values = completed[candidate["id"]]
        if set(values) != set(CONFIGURATIONS):
            raise ValueError("Candidato sem as cinco representações")
        scores = [float(values[key]) for key in CONFIGURATIONS]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in scores):
            raise ValueError("AP inválida")
        ranked.append((sum(scores) / 5, -order, candidate))
    score, _, candidate = max(ranked, key=lambda r: (r[0], r[1]))
    return {**candidate, "mean_ap": score}


def validate_destination(destination, protected):
    destination = Path(destination).resolve()
    for source in protected:
        source = Path(source).resolve()
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError(f"Destino sobrepõe entrada protegida: {source}")
    if destination.exists() and any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction())
                                    for p in destination.rglob("*")):
        raise ValueError("Destino contém links ou junctions; usar diretório isolado")
    return destination


def code_provenance(root):
    root = Path(root)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    files = sorted((root / "revisao_experimental").glob("*.py"))
    files += [root / "executar_revisao.py"]
    files += sorted((root / "03-machine-learning/01-features").glob("*.py"))
    files += sorted((root / "03-machine-learning/02-ae").glob("*.py"))
    files += sorted((root / "03-machine-learning/03-if").glob("*.py"))
    files += sorted((root / "05-analise-resultados").glob("*.py"))
    return {"git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain", "--", "revisao_experimental", "executar_revisao.py"),
            "python": platform.python_version(),
            "code_hashes": {str(p.relative_to(root)): file_hash(p) for p in files}}


class Store:
    """Cada etapa possui assinatura e hashes; parciais nunca são reutilizados."""
    def __init__(self, root, frozen):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.frozen = frozen
        self.identity = signature(frozen)
        path = self.root / "protocol.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != frozen:
                raise ValueError("Protocolo/entradas/código mudaram; usar nova saída")
        else:
            if any(self.root.iterdir()):
                raise ValueError("Destino não vazio e sem protocolo da revisão")
            atomic_json(path, frozen)

    def destination(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Saída fora do diretório da revisão")
        return path

    def complete(self, name):
        path = self.destination(name) / "manifest.json"
        if not path.exists():
            return False
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("status") != "completed":
            return False
        if doc.get("protocol_signature") != self.identity:
            raise ValueError("Assinatura incompatível")
        for relative, digest in doc["outputs"].items():
            output = self.destination(f"{name}/{relative}")
            if not output.is_file() or file_hash(output) != digest:
                raise ValueError(f"Artefato incompleto ou modificado: {output}")
        return True

    @contextmanager
    def stage(self, name):
        directory = self.destination(name)
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / ".lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        start = datetime.now(timezone.utc).isoformat()
        try:
            if self.complete(name):
                raise ValueError("Etapa concluída; reutilize, não sobrescreva")
            atomic_json(directory / "manifest.json", {"status": "running", "started_at": start,
                        "protocol_signature": self.identity})
            yield directory
            outputs = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*")
                       if p.is_file() and p.name not in ("manifest.json", ".lock")}
            if not outputs:
                raise ValueError("Etapa sem saídas")
            atomic_json(directory / "manifest.json", {"status": "completed", "started_at": start,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "protocol_signature": self.identity, "outputs": outputs})
        except BaseException as exc:
            # Do not leak credentials or arbitrary exception data to manifests.
            if not self.complete(name):
                atomic_json(directory / "manifest.json", {"status": "failed", "started_at": start,
                            "protocol_signature": self.identity, "error_type": type(exc).__name__})
            raise
        finally:
            lock.unlink()
