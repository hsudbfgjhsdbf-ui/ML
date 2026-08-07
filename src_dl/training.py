"""Unified PyTorch training loops, checkpoints, and epoch telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluation.metrics import compute_metrics
from src_dl.data import make_loader
from src_dl.models import build_model
from src_dl.utils import atomic_json, parameter_count, seed_everything


@dataclass(frozen=True)
class TrainingConfig:
    """Training controls shared by all five architectures."""

    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    reconstruction_weight: float
    device: torch.device


def _forward(model: nn.Module, batch: torch.Tensor, autoencoder: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Call a model through the common auxiliary-output contract."""
    output = model(batch, return_aux=True)
    if autoencoder:
        logits, aux = output
        return logits, aux
    logits, aux = output
    return logits, aux


def train_model(
    key: str,
    input_dim: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    model_config: dict[str, Any],
    cfg: TrainingConfig,
    seed: int,
    checkpoint_dir: Path,
    log_dir: Path,
) -> dict[str, Any]:
    """Train one architecture with BCE weighting, clipping, and early stopping.

    Args:
        key: Deep model key.
        input_dim: Frozen feature width.
        x_train/y_train: Training arrays.
        x_validation/y_validation: Validation arrays.
        model_config: Architecture settings.
        cfg: Shared optimization settings.
        seed: Seed for initialization and loader order.
        checkpoint_dir: Directory for best state dictionaries.
        log_dir: Directory for epoch CSV logs.
    Returns:
        Dictionary containing model, validation probabilities, epoch telemetry,
        best epoch, training time, and parameter count.
    """
    seed_everything(seed)
    model = build_model(key, input_dim, model_config).to(cfg.device)
    train_loader = make_loader(x_train, y_train, cfg.batch_size, True, seed)
    validation_loader = make_loader(x_validation, y_validation, cfg.batch_size * 2, False, seed)
    positives = max(1.0, float(y_train.sum()))
    negatives = max(1.0, float(len(y_train) - y_train.sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    autoencoder = key == "dl_d_autoencoder"
    best_score = -np.inf
    best_epoch = 0
    stale = 0
    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{key}_s{seed}.pt"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        gradient_norms = []
        epoch_start = time.perf_counter()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits, aux = _forward(model, batch_x, autoencoder)
            loss = criterion(logits.unsqueeze(-1), batch_y.unsqueeze(-1))
            if autoencoder:
                # Reconstruction is auxiliary; the supervised logit still uses
                # the same fraud-positive target as every other architecture.
                reconstruction = aux["reconstruction"]
                loss = loss + cfg.reconstruction_weight * nn.functional.mse_loss(reconstruction, batch_x)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN or infinite loss in {key} seed {seed} epoch {epoch}")
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            gradient_norms.append(gradient_norm)
        scheduler.step()
        model.eval()
        val_losses = []
        probabilities = []
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                batch_x = batch_x.to(cfg.device)
                batch_y = batch_y.to(cfg.device)
                logits, aux = _forward(model, batch_x, autoencoder)
                loss = criterion(logits.unsqueeze(-1), batch_y.unsqueeze(-1))
                if autoencoder:
                    loss = loss + cfg.reconstruction_weight * nn.functional.mse_loss(aux["reconstruction"], batch_x)
                val_losses.append(float(loss.detach().cpu()))
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        val_prob = np.concatenate(probabilities)
        quick_metrics = compute_metrics(y_validation.astype(int), val_prob, threshold=0.5)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": float(np.mean(val_losses)),
            "validation_pr_auc": quick_metrics["pr_auc"],
            "validation_roc_auc": quick_metrics["roc_auc"],
            "validation_f2_at_0_5": quick_metrics["f2"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        rows.append(row)
        score = float(quick_metrics["pr_auc"])
        if score > best_score + 1e-7:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_key": key,
                    "seed": seed,
                    "best_epoch": epoch,
                    "input_dim": input_dim,
                    "parameter_count": parameter_count(model),
                },
                checkpoint,
            )
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device, weights_only=True)["model_state"])
    model.eval()
    val_probabilities = []
    with torch.no_grad():
        for batch_x, _ in validation_loader:
            logits, _ = _forward(model, batch_x.to(cfg.device), autoencoder)
            val_probabilities.append(torch.sigmoid(logits).cpu().numpy())
    elapsed = time.perf_counter() - start_time
    telemetry = pd.DataFrame(rows)
    telemetry["best_epoch"] = best_epoch
    telemetry["seed"] = seed
    telemetry["model_key"] = key
    telemetry.to_csv(log_dir / f"{key}_s{seed}_epoch_log.csv", index=False)
    atomic_json(
        log_dir / f"{key}_s{seed}_training.json",
        {
            "key": key,
            "seed": seed,
            "best_epoch": best_epoch,
            "epochs_run": len(rows),
            "best_validation_pr_auc": best_score,
            "training_seconds": elapsed,
            "parameter_count": parameter_count(model),
            "device": str(cfg.device),
        },
    )
    return {
        "model": model,
        "validation_probabilities": np.concatenate(val_probabilities),
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "parameter_count": parameter_count(model),
        "telemetry": telemetry,
        "checkpoint": str(checkpoint),
        "seed": seed,
    }
