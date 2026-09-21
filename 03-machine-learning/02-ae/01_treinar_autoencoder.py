# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "duckdb>=1.4.3,<2",
#   "matplotlib>=3.10,<4",
#   "numpy>=2.1,<3",
#   "pyarrow>=18,<24",
#   "torch>=2.11,<3",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///

"""Treina Autoencoders na GPU e pontua os recortes temporais sem vazamento."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn


MATRIX_FILES = {
    "A": "12_matriz_A_preprocessada.parquet",
    "B": "13_matriz_B_preprocessada.parquet",
    "C": "14_matriz_C_preprocessada.parquet",
}
ABSOLUTE_FEE_FEATURES = {
    "effective_gas_price_log",
    "max_fee_per_gas_log",
    "max_priority_fee_log",
    "transaction_fee_paid_log",
}


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int], latent_dim: int):
        super().__init__()
        first, second = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.LeakyReLU(0.1),
            nn.Linear(first, second),
            nn.LeakyReLU(0.1),
            nn.Linear(second, latent_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(latent_dim, second),
            nn.LeakyReLU(0.1),
            nn.Linear(second, first),
            nn.LeakyReLU(0.1),
            nn.Linear(first, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    src_root = here.parents[1]
    feature_root = here.parent / "01-features"
    parser = argparse.ArgumentParser(
        description="Treina Autoencoders com CUDA e avalia os recortes temporais."
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=feature_root / "resultados-splits" / "00_manifest_splits_temporais.json",
    )
    parser.add_argument(
        "--preprocessing-manifest", type=Path,
        default=feature_root / "resultados-preprocessamento" / "05_manifest_preprocessamento.json",
    )
    parser.add_argument(
        "--if-results-dir", type=Path, default=here.parent / "03-if" / "resultados"
    )
    parser.add_argument("--output-dir", type=Path, default=here / "resultados")
    parser.add_argument("--temp-dir", type=Path, default=src_root / ".tmp" / "duckdb")
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--inference-batch-size", type=int, default=262_144)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=4)
    args = parser.parse_args()
    for path in (args.split_manifest, args.preprocessing_manifest):
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")
    for name in ("threads", "batch_size", "inference_batch_size", "max_epochs", "patience", "latent_dim"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} deve ser positivo")
    return args


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sql_quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_duckdb(args: argparse.Namespace) -> duckdb.DuckDBPyConnection:
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = {sql_quote(args.memory_limit)}")
    con.execute(f"SET temp_directory = {sql_quote(args.temp_dir.as_posix())}")
    con.execute("SET preserve_insertion_order = false")
    return con


def input_path(split: dict[str, object], matrix: str) -> Path:
    return Path(str(split["output_directory"])) / MATRIX_FILES[matrix]


def metadata_path(split: dict[str, object]) -> Path:
    return Path(str(split["output_directory"])) / "05_metadados.parquet"


def validate_inputs(splits: list[dict[str, object]], preprocessing: dict[str, object]) -> None:
    expected = str(preprocessing["preprocessing_signature"])
    for split in splits:
        directory = Path(str(split["output_directory"]))
        local_manifest = directory / "15_manifest_preprocessamento.json"
        if not local_manifest.is_file():
            raise FileNotFoundError(f"Manifesto ausente: {local_manifest}")
        if str(load_json(local_manifest).get("preprocessing_signature")) != expected:
            raise RuntimeError(f"Pré-processamento incompatível em {split['key']}.")
        for matrix in MATRIX_FILES:
            if not input_path(split, matrix).is_file():
                raise FileNotFoundError(f"Matriz ausente: {input_path(split, matrix)}")


def build_configurations(preprocessing: dict[str, object]) -> list[dict[str, object]]:
    matrices = preprocessing["matrices"]
    configs: list[dict[str, object]] = []
    for matrix in ("A", "B", "C"):
        configs.append({
            "key": f"{matrix.lower()}_completa", "matrix": matrix,
            "family": "completa", "features": list(matrices[matrix]),
            "excluded_features": [],
        })
    for matrix in ("B", "C"):
        original = list(matrices[matrix])
        configs.append({
            "key": f"{matrix.lower()}_sem_taxas_absolutas", "matrix": matrix,
            "family": "sem_taxas_absolutas",
            "features": [feature for feature in original if feature not in ABSOLUTE_FEE_FEATURES],
            "excluded_features": [feature for feature in original if feature in ABSOLUTE_FEE_FEATURES],
        })
    return configs


def select_device(requested: str) -> torch.device:
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA foi solicitada, mas o PyTorch não detectou a GPU.")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if available else "cpu")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_training_superset(path: Path, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["row_id", *features])
    row_ids = table.column("row_id").combine_chunks().to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
    values = np.column_stack([
        table.column(feature).combine_chunks().to_numpy(zero_copy_only=False)
        for feature in features
    ]).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError("A matriz de treinamento contém valores inválidos.")
    return row_ids, values


def train_autoencoder(
    values: np.ndarray,
    row_ids: np.ndarray,
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
    configuration: str,
) -> tuple[Autoencoder, list[dict[str, object]], int, float]:
    internal_validation = (row_ids % 10) == 0
    # A matriz inteira ocupa menos de 300 MB na configuração mais larga. Mantê-la
    # na RTX 3060 evita o custo do DataLoader linha a linha e aproveita a VRAM.
    train_values = torch.from_numpy(values[~internal_validation]).to(device)
    validation_values = torch.from_numpy(values[internal_validation]).to(device)
    model = Autoencoder(input_dim, (16, 8), args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss(reduction="mean")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, object]] = []
    start = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_sum = 0.0
        train_rows = 0
        permutation = torch.randperm(len(train_values), device=device)
        for offset in range(0, len(train_values), args.batch_size):
            batch = train_values[permutation[offset : offset + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * len(batch)
            train_rows += len(batch)
        model.eval()
        validation_sum = 0.0
        validation_rows_count = 0
        with torch.inference_mode():
            validation_batch_size = args.batch_size * 2
            for offset in range(0, len(validation_values), validation_batch_size):
                batch = validation_values[offset : offset + validation_batch_size]
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = criterion(model(batch), batch)
                validation_sum += float(loss) * len(batch)
                validation_rows_count += len(batch)
        train_loss = train_sum / train_rows
        validation_loss = validation_sum / validation_rows_count
        improved = validation_loss < best_loss - args.min_delta
        history.append({
            "configuration": configuration, "epoch": epoch,
            "train_loss": train_loss, "internal_validation_loss": validation_loss,
            "improved": int(improved), "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if epoch == 1 or epoch % 5 == 0 or not improved:
            print(f"    época {epoch:03d}: treino={train_loss:.7f} validação_interna={validation_loss:.7f}")
        if improved:
            best_loss = validation_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("O Autoencoder não produziu um estado válido.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, history, int(np.argmin([float(row["internal_validation_loss"]) for row in history])) + 1, time.perf_counter() - start


def score_batches(
    model: Autoencoder,
    path: Path,
    features: list[str],
    device: torch.device,
    batch_size: int,
    output_path: Path | None,
    thresholds: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    parquet = pq.ParquetFile(path)
    writer: pq.ParquetWriter | None = None
    ids_list: list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["row_id", *features]):
                row_ids = batch.column(0).to_numpy(zero_copy_only=False).astype(np.uint64, copy=False)
                values = np.column_stack([
                    batch.column(index).to_numpy(zero_copy_only=False)
                    for index in range(1, len(features) + 1)
                ]).astype(np.float32, copy=False)
                tensor = torch.from_numpy(values).to(device, non_blocking=True)
                reconstructed = model(tensor)
                errors = torch.mean((reconstructed.float() - tensor.float()) ** 2, dim=1)
                scores = errors.cpu().numpy().astype(np.float32, copy=False)
                ids_list.append(row_ids)
                scores_list.append(scores)
                if output_path is not None and thresholds is not None:
                    table = pa.table({
                        "row_id": row_ids,
                        "reconstruction_error": scores,
                        "is_anomaly_q990": scores >= thresholds["q990"],
                        "is_anomaly_q995": scores >= thresholds["q995"],
                        "is_anomaly_q999": scores >= thresholds["q999"],
                    })
                    if writer is None:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                    writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    return np.concatenate(ids_list), np.concatenate(scores_list)


def score_statistics(scores: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(scores, [0.25, 0.50, 0.75, 0.95, 0.99, 0.995, 0.999])
    return {
        "score_min": float(np.min(scores)), "score_q250": float(quantiles[0]),
        "score_median": float(quantiles[1]), "score_mean": float(np.mean(scores)),
        "score_q750": float(quantiles[2]), "score_q950": float(quantiles[3]),
        "score_q990": float(quantiles[4]), "score_q995": float(quantiles[5]),
        "score_q999": float(quantiles[6]), "score_max": float(np.max(scores)),
        "score_std": float(np.std(scores)),
    }


def plot_history(path_pdf: Path, path_png: Path, history: list[dict[str, object]], configs: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, config in zip(axes.flat, configs):
        rows = [row for row in history if row["configuration"] == config["key"]]
        axis.plot([row["epoch"] for row in rows], [row["train_loss"] for row in rows], label="treino")
        axis.plot([row["epoch"] for row in rows], [row["internal_validation_loss"] for row in rows], label="validação interna")
        axis.set_title(str(config["key"])); axis.set_xlabel("Época"); axis.set_ylabel("MSE")
        axis.grid(alpha=0.25); axis.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Autoencoder: curvas de reconstrução", fontsize=14)
    fig.savefig(path_pdf, bbox_inches="tight"); fig.savefig(path_png, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_rates(path_pdf: Path, path_png: Path, rates: list[dict[str, object]], splits: list[dict[str, object]], configs: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    keys = [str(split["key"]) for split in splits]
    lookup = {(str(row["configuration"]), str(row["split"])): float(row["anomaly_rate_q995"]) * 100 for row in rates}
    fig, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for config in configs:
        key = str(config["key"])
        axis.plot(range(len(keys)), [lookup[(key, split)] for split in keys], marker="o", linewidth=1.8, label=key)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="calibração 0,5%")
    axis.set_xticks(range(len(keys)), [key.replace("_", "\n") for key in keys], fontsize=8)
    axis.set_ylabel("Transações sinalizadas (%)"); axis.set_xlabel("Recorte temporal")
    axis.set_title("Autoencoder: taxa de anomalias pelo limiar Q99,5 da validação")
    axis.grid(axis="y", alpha=0.25); axis.legend(fontsize=8, ncol=2)
    fig.savefig(path_pdf, bbox_inches="tight"); fig.savefig(path_png, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    args = parse_args()
    set_reproducibility(args.random_state)
    device = select_device(args.device)
    if device.type == "cuda":
        print(f"GPU ativa: {torch.cuda.get_device_name(0)} | CUDA PyTorch {torch.version.cuda}")
    else:
        print("AVISO: treinamento em CPU; CUDA não está ativa.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir, score_root = args.output_dir / "modelos", args.output_dir / "escores"
    model_dir.mkdir(parents=True, exist_ok=True); score_root.mkdir(parents=True, exist_ok=True)

    split_document, preprocessing = load_json(args.split_manifest), load_json(args.preprocessing_manifest)
    splits = list(split_document["splits"]); validate_inputs(splits, preprocessing)
    train = next(split for split in splits if split["role"] == "train")
    validation = next(split for split in splits if split["role"] == "validation")
    configs = build_configurations(preprocessing)
    all_features = list(preprocessing["matrices"]["C"])
    con = configure_duckdb(args)

    device_info = {
        "requested": args.device, "selected": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
        "torch_cuda_version": torch.version.cuda,
    }
    experiment = {
        "generated_at": now_iso(), "method": "Autoencoder",
        "fit_split": train["key"], "threshold_split": validation["key"],
        "internal_validation": "row_id modulo 10 igual a zero, somente dentro do treino",
        "architecture": {"hidden_dims": [16, 8], "latent_dim": args.latent_dim, "activation": "LeakyReLU(0.1)", "output": "linear"},
        "optimization": {"loss": "MSE", "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience, "min_delta": args.min_delta, "mixed_precision_cuda": True},
        "threshold_quantiles_validation": [0.990, 0.995, 0.999],
        "random_state": args.random_state, "configurations": configs,
        "preprocessing_signature": preprocessing["preprocessing_signature"], "device": device_info,
        "software_versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pyarrow": pa.__version__, "duckdb": duckdb.__version__, "matplotlib": matplotlib.__version__},
    }
    signature_payload = {key: value for key, value in experiment.items() if key != "generated_at"}
    experiment_signature = hashlib.sha256(json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
    experiment["experiment_signature"] = experiment_signature
    (args.output_dir / "01_configuracoes_experimento.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Carregando a matriz C de treinamento uma única vez...")
    train_ids, train_superset = load_training_superset(input_path(train, "C"), all_features)
    feature_indexes = {feature: index for index, feature in enumerate(all_features)}
    training_rows: list[dict[str, object]] = []; history_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []; statistics_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []; type4_rows: list[dict[str, object]] = []

    for config_index, config in enumerate(configs, start=1):
        key, matrix_name, features = str(config["key"]), str(config["matrix"]), list(config["features"])
        print(f"\n[{config_index}/{len(configs)}] Treinando {key} ({len(features)} features)...")
        indexes = [feature_indexes[feature] for feature in features]
        values = np.ascontiguousarray(train_superset[:, indexes], dtype=np.float32)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        model, history, best_epoch, fit_seconds = train_autoencoder(values, train_ids, len(features), device, args, key)
        history_rows.extend(history)
        peak_memory = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        print("  Calibrando quantis exclusivamente na validação...")
        _, validation_scores = score_batches(model, input_path(validation, matrix_name), features, device, args.inference_batch_size, None, None)
        thresholds = {"q990": float(np.quantile(validation_scores, 0.990)), "q995": float(np.quantile(validation_scores, 0.995)), "q999": float(np.quantile(validation_scores, 0.999))}
        threshold_rows.append({"configuration": key, "calibration_split": validation["key"], "validation_rows": len(validation_scores), **thresholds})
        del validation_scores
        model_path = model_dir / f"{key}.pt"
        torch.save({"state_dict": model.state_dict(), "input_dim": len(features), "hidden_dims": [16, 8], "latent_dim": args.latent_dim, "features": features, "configuration": config, "thresholds": thresholds, "experiment_signature": experiment_signature, "preprocessing_signature": preprocessing["preprocessing_signature"]}, model_path)
        training_rows.append({"configuration": key, "matrix": matrix_name, "family": config["family"], "features": len(features), "training_rows_total": len(values), "training_rows_fit": int(np.sum((train_ids % 10) != 0)), "internal_validation_rows": int(np.sum((train_ids % 10) == 0)), "best_epoch": best_epoch, "epochs_executed": len(history), "best_internal_validation_loss": min(float(row["internal_validation_loss"]) for row in history), "fit_seconds": fit_seconds, "device": device.type, "peak_gpu_memory_bytes": peak_memory, "model_path": str(model_path.resolve()), "model_size_bytes": model_path.stat().st_size})
        del values

        for split_index, split in enumerate(splits, start=1):
            split_key = str(split["key"]); print(f"  [{split_index}/{len(splits)}] Pontuando {split_key}...")
            output_path = score_root / key / f"{int(split['order']):02d}_{split_key}.parquet"
            _, scores = score_batches(model, input_path(split, matrix_name), features, device, args.inference_batch_size, output_path, thresholds)
            stats = score_statistics(scores); flags = {name: scores >= threshold for name, threshold in thresholds.items()}
            statistics_rows.append({"split_order": split["order"], "split": split_key, "role": split["role"], "configuration": key, "matrix": matrix_name, "rows": len(scores), **stats})
            rate_rows.append({"split_order": split["order"], "split": split_key, "role": split["role"], "configuration": key, "matrix": matrix_name, "rows": len(scores), "anomalies_q990": int(np.sum(flags["q990"])), "anomaly_rate_q990": float(np.mean(flags["q990"])), "anomalies_q995": int(np.sum(flags["q995"])), "anomaly_rate_q995": float(np.mean(flags["q995"])), "anomalies_q999": int(np.sum(flags["q999"])), "anomaly_rate_q999": float(np.mean(flags["q999"])), "score_path": str(output_path.resolve()), "score_size_bytes": output_path.stat().st_size})
            groups = con.execute(f"""SELECT m.is_type_4, COUNT(*)::BIGINT, SUM(CASE WHEN s.is_anomaly_q995 THEN 1 ELSE 0 END)::BIGINT, AVG(CASE WHEN s.is_anomaly_q995 THEN 1.0 ELSE 0.0 END)::DOUBLE, AVG(s.reconstruction_error)::DOUBLE, QUANTILE_CONT(s.reconstruction_error, 0.50)::DOUBLE, QUANTILE_CONT(s.reconstruction_error, 0.995)::DOUBLE FROM read_parquet({sql_quote(output_path.as_posix())}) s INNER JOIN read_parquet({sql_quote(metadata_path(split).as_posix())}) m USING (row_id) GROUP BY m.is_type_4 ORDER BY m.is_type_4""").fetchall()
            for is_type4, rows, anomalies, rate, mean_score, median_score, q995_score in groups:
                type4_rows.append({"split_order": split["order"], "split": split_key, "role": split["role"], "configuration": key, "is_type_4": int(is_type4), "rows": int(rows), "anomalies_q995": int(anomalies), "anomaly_rate_q995": float(rate), "score_mean": float(mean_score), "score_median": float(median_score), "score_q995": float(q995_score)})
            del scores, flags
        del model
        if device.type == "cuda": torch.cuda.empty_cache()

    del train_superset, train_ids
    write_csv(args.output_dir / "02_resumo_treinamento.csv", training_rows)
    write_csv(args.output_dir / "03_historico_treinamento.csv", history_rows)
    write_csv(args.output_dir / "04_limiares_validacao.csv", threshold_rows)
    write_csv(args.output_dir / "05_estatisticas_escores.csv", statistics_rows)
    write_csv(args.output_dir / "06_taxas_anomalias_recortes.csv", rate_rows)
    write_csv(args.output_dir / "07_taxas_anomalias_tipo4.csv", type4_rows)

    validation_sets: dict[str, set[int]] = {}
    for config in configs:
        key = str(config["key"]); path = score_root / key / f"{int(validation['order']):02d}_{validation['key']}.parquet"
        validation_sets[key] = {int(row[0]) for row in con.execute(f"SELECT row_id FROM read_parquet({sql_quote(path.as_posix())}) WHERE is_anomaly_q995").fetchall()}
    agreement_rows: list[dict[str, object]] = []
    for left, right in combinations(validation_sets, 2):
        intersection = len(validation_sets[left] & validation_sets[right]); union = len(validation_sets[left] | validation_sets[right])
        agreement_rows.append({"calibration_split": validation["key"], "configuration_left": left, "configuration_right": right, "anomalies_left": len(validation_sets[left]), "anomalies_right": len(validation_sets[right]), "intersection": intersection, "union": union, "jaccard": intersection / union if union else 1.0})
    write_csv(args.output_dir / "08_concordancia_ae_validacao.csv", agreement_rows)

    cross_rows: list[dict[str, object]] = []
    if args.if_results_dir.is_dir():
        for split in splits:
            for config in configs:
                key = str(config["key"]); prefix = f"{int(split['order']):02d}_{split['key']}.parquet"
                ae_path, if_path = score_root / key / prefix, args.if_results_dir / "escores" / key / prefix
                if if_path.is_file():
                    row = con.execute(f"""SELECT SUM(CASE WHEN a.is_anomaly_q995 THEN 1 ELSE 0 END)::BIGINT, SUM(CASE WHEN i.is_anomaly_q995 THEN 1 ELSE 0 END)::BIGINT, SUM(CASE WHEN a.is_anomaly_q995 AND i.is_anomaly_q995 THEN 1 ELSE 0 END)::BIGINT, SUM(CASE WHEN a.is_anomaly_q995 OR i.is_anomaly_q995 THEN 1 ELSE 0 END)::BIGINT FROM read_parquet({sql_quote(ae_path.as_posix())}) a INNER JOIN read_parquet({sql_quote(if_path.as_posix())}) i USING (row_id)""").fetchone()
                    cross_rows.append({"split_order": split["order"], "split": split["key"], "role": split["role"], "configuration": key, "ae_anomalies": int(row[0]), "if_anomalies": int(row[1]), "intersection": int(row[2]), "union": int(row[3]), "jaccard": int(row[2]) / int(row[3]) if row[3] else 1.0})
    write_csv(args.output_dir / "09_concordancia_ae_if.csv", cross_rows)
    plot_history(args.output_dir / "10_curvas_treinamento.pdf", args.output_dir / "10_curvas_treinamento.png", history_rows, configs)
    plot_rates(args.output_dir / "11_taxas_anomalias_recortes.pdf", args.output_dir / "11_taxas_anomalias_recortes.png", rate_rows, splits, configs)

    methodology = f"""# Metodologia do Autoencoder

Os modelos foram ajustados exclusivamente em `{train['key']}`. Dentro desse recorte,
90% das linhas treinam os pesos e 10% formam uma validação interna determinística
para early stopping. Essa divisão interna não é a validação temporal de 2025.

A arquitetura simétrica é `entrada -> 16 -> 8 -> {args.latent_dim} -> 8 -> 16 ->
saída`, com LeakyReLU e saída linear. O treinamento minimiza MSE com AdamW.
O escore por transação é o erro quadrático médio de reconstrução.

Os limiares Q99, Q99,5 e Q99,9 são obtidos somente em
`{validation['key']}`. O Q99,5 é a análise principal. As cinco configurações são
idênticas às do Isolation Forest e foram congeladas antes da avaliação.

Dispositivo usado: `{device_info['selected']}`; GPU: `{device_info['gpu_name']}`.
Resultados do teste não devem orientar novo ajuste de arquitetura, features ou limiar.
"""
    (args.output_dir / "12_metodologia_autoencoder.md").write_text(methodology, encoding="utf-8")
    outputs = ["01_configuracoes_experimento.json", "02_resumo_treinamento.csv", "03_historico_treinamento.csv", "04_limiares_validacao.csv", "05_estatisticas_escores.csv", "06_taxas_anomalias_recortes.csv", "07_taxas_anomalias_tipo4.csv", "08_concordancia_ae_validacao.csv", "09_concordancia_ae_if.csv", "10_curvas_treinamento.pdf", "10_curvas_treinamento.png", "11_taxas_anomalias_recortes.pdf", "11_taxas_anomalias_recortes.png", "12_metodologia_autoencoder.md", "13_manifest_autoencoder.json"]
    manifest = {"generated_at": now_iso(), "experiment_signature": experiment_signature, "preprocessing_signature": preprocessing["preprocessing_signature"], "fit_split": train["key"], "threshold_split": validation["key"], "device": device_info, "configurations": [config["key"] for config in configs], "outputs": outputs, "models_directory": str(model_dir.resolve()), "scores_directory": str(score_root.resolve())}
    (args.output_dir / "13_manifest_autoencoder.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    con.close(); print(f"\nAutoencoder concluído: {args.output_dir.resolve()}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
