"""Motores de ajuste compartilhados pela busca e pelo ajuste final."""
from __future__ import annotations

import copy
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from .core import PROTOCOL


class Autoencoder(nn.Module):
    def __init__(self, dimension, hidden_dims, latent_dim):
        super().__init__()
        sizes = [dimension, *hidden_dims, latent_dim, *reversed(hidden_dims), dimension]
        layers = []
        for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(sizes) - 2:
                layers.append(nn.LeakyReLU(.1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values)


class Checkpoint:
    def __init__(self, min_delta, patience):
        self.min_delta, self.patience = min_delta, patience
        self.loss, self.epoch, self.stale = float("inf"), None, 0

    def update(self, loss, epoch):
        if not np.isfinite(loss):
            raise ValueError("Perda não finita")
        improved = loss < self.loss - self.min_delta
        if improved:
            self.loss, self.epoch, self.stale = loss, epoch, 0
        else:
            self.stale += 1
        return improved


def fit_model(method, parameters, values, internal, device="cpu", budget=None):
    p = {**PROTOCOL, **(budget or {})}
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Entradas de ajuste inválidas")
    random.seed(p["model_seed"])
    np.random.seed(p["model_seed"])
    start = time.perf_counter()
    if method == "IF":
        if parameters["max_samples"] > len(values):
            raise ValueError("Amostra menor que max_samples; não reduzir automaticamente")
        model = IsolationForest(**parameters, max_features=1., bootstrap=False,
                                contamination="auto", random_state=p["model_seed"], n_jobs=p["threads"])
        model.fit(values)
        return model, {"fit_seconds": time.perf_counter() - start, "fit_rows": len(values),
                       "max_samples_effective": int(model.max_samples_), "history": []}
    if method != "AE":
        raise ValueError("Método desconhecido")
    internal = np.asarray(internal, dtype=bool)
    if internal.shape != (len(values),) or not internal.any() or internal.all():
        raise ValueError("Partição interna vazia ou inválida")
    torch.set_num_threads(p["threads"])
    torch.manual_seed(p["model_seed"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(p["model_seed"])
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    train = torch.from_numpy(values[~internal]).to(device)
    validation = torch.from_numpy(values[internal]).to(device)
    model = Autoencoder(values.shape[1], parameters["hidden_dims"], parameters["latent_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint = Checkpoint(p["min_delta"], p["patience"])
    history, state = [], None
    for epoch in range(1, p["max_epochs"] + 1):
        model.train()
        total = 0.
        order = torch.randperm(len(train), device=device)
        for offset in range(0, len(train), p["batch_size"]):
            batch = train[order[offset:offset + p["batch_size"]]]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = torch.mean((model(batch) - batch) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("Perda de ajuste não finita")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(batch)
        model.eval()
        validation_total = 0.
        with torch.inference_mode():
            for offset in range(0, len(validation), p["batch_size"]):
                batch = validation[offset:offset + p["batch_size"]]
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = torch.mean((model(batch) - batch) ** 2)
                validation_total += float(loss) * len(batch)
        loss_value = validation_total / len(validation)
        improved = checkpoint.update(loss_value, epoch)
        if improved:
            state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        history.append({"epoch": epoch, "train_loss": total / len(train),
                        "internal_validation_loss": loss_value, "improved": improved})
        if checkpoint.stale >= checkpoint.patience:
            break
    if state is None:
        raise ValueError("Nenhum checkpoint válido")
    model.load_state_dict(state)
    model.to(device).eval()
    # The accepted checkpoint may differ from the absolute minimum by min_delta.
    return model, {"fit_seconds": time.perf_counter() - start, "fit_rows": len(train),
                   "internal_rows": len(validation), "epochs": len(history),
                   "checkpoint_epoch": checkpoint.epoch, "checkpoint_loss": checkpoint.loss,
                   "raw_minimum_epoch": int(np.argmin([h["internal_validation_loss"] for h in history])) + 1,
                   "stop_reason": "patience" if checkpoint.stale >= checkpoint.patience else "max_epochs",
                   "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "device": str(device), "history": history}


def score_model(method, model, values, batch_size=262144):
    if not np.isfinite(values).all():
        raise ValueError("Entradas de inferência não finitas")
    if method == "IF":
        scores = -model.score_samples(values)
    else:
        device = next(model.parameters()).device
        model.eval()
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(values), batch_size):
                batch = torch.from_numpy(values[offset:offset + batch_size]).to(device)
                scores.extend(torch.mean((model(batch).float() - batch.float()) ** 2, dim=1).cpu().numpy())
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Escores não finitos")
    return scores
