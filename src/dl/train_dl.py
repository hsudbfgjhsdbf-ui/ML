"""Unified deep learning training pipeline (Approach 2).

Provides a single Trainer class that trains any of the ten architectures with
early stopping, checkpointing, loss logging and evaluation, plus helper to
prepare the tabular data into numpy/torch tensors.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import setup_logging

logger = setup_logging()


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(y: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights from labels.

    Args:
        y: Binary labels.

    Returns:
        np.ndarray: [weight_neg, weight_pos] where weight_pos is pos_weight.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return np.array([1.0, n_neg / max(n_pos, 1)])


class Trainer:
    """Train and evaluate a single deep learning architecture."""

    def __init__(self, model, criterion: str = "bce",
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 128, epochs: int = 60,
                 patience: int = 15, clip: float = 1.0,
                 device: str = "cpu", class_w: np.ndarray | None = None,
                 seed: int = 42):
        self.model = model.to(device)
        self.criterion_name = criterion
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.clip = clip
        self.device = device
        self.class_w = class_w
        self.history = {"train_loss": [], "val_loss": [], "val_f2": []}
        set_seed(seed)

    def _make_batches(self, X, y=None, shuffle=True):
        n = len(X)
        idx = np.arange(n)
        rng = np.random.default_rng(0)
        if shuffle:
            rng.shuffle(idx)
        for i in range(0, n, self.batch_size):
            b = idx[i:i + self.batch_size]
            if y is None:
                yield torch.tensor(X[b], dtype=torch.float32)
            else:
                yield (torch.tensor(X[b], dtype=torch.float32),
                       torch.tensor(y[b], dtype=torch.float32).view(-1, 1))

    def _loss_fn(self):
        if self.criterion_name == "weighted_bce":
            pos_w = torch.tensor([self.class_w[1]], dtype=torch.float32)
            return nn.BCEWithLogitsLoss(pos_weight=pos_w)
        return nn.BCEWithLogitsLoss()

    def fit(self, X_train, y_train, X_val, y_val, checkpoint_path: Path):
        """Run the training loop with early stopping and checkpointing."""
        logger.info("Training %s (%d epochs)", type(self.model).__name__, self.epochs)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                      weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        loss_fn = self._loss_fn()

        best_f2, best_state, bad = -1.0, None, 0
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, n = 0.0, 0
            for Xb, yb in self._make_batches(X_train, y_train):
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                if self.model.anomaly:
                    out = self.model(Xb)
                    if isinstance(out, tuple):   # VAE -> (recon, mu, logvar)
                        recon, mu, logvar = out
                        recon_loss = nn.functional.mse_loss(recon, Xb)
                        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                        loss = recon_loss + self.model.beta * kl
                    else:                        # Autoencoder -> recon only
                        loss = nn.functional.mse_loss(out, Xb)
                else:
                    logits = self.model(Xb)
                    loss = loss_fn(logits, yb)
                loss.backward()
                if self.clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)
                n += len(Xb)
            scheduler.step()

            val_loss, val_f2 = self.evaluate(X_val, y_val)
            self.history["train_loss"].append(epoch_loss / n)
            self.history["val_loss"].append(val_loss)
            self.history["val_f2"].append(val_f2)
            if val_f2 > best_f2:
                best_f2, bad = val_f2, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
            if bad >= self.patience:
                logger.info("Early stopping at epoch %d (best val F2=%.4f)",
                            epoch, best_f2)
                break
            if epoch % 10 == 0:
                logger.info("Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_f2=%.4f",
                            epoch, self.history["train_loss"][-1], val_loss, val_f2)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        return best_f2

    def evaluate(self, X_val, y_val):
        """Return (validation loss, validation F2)."""
        self.model.eval()
        loss_fn = self._loss_fn()
        total, n = 0.0, 0
        all_prob = []
        with torch.no_grad():
            for Xb, yb in self._make_batches(X_val, y_val, shuffle=False):
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                if self.model.anomaly:
                    out = self.model(Xb)
                    if isinstance(out, tuple):
                        recon, mu, logvar = out
                        loss = nn.functional.mse_loss(recon, Xb) + self.model.beta * (
                            -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
                    else:
                        loss = nn.functional.mse_loss(out, Xb)
                    prob = torch.sigmoid(torch.zeros_like(yb))
                else:
                    logits = self.model(Xb)
                    loss = loss_fn(logits, yb)
                    prob = torch.sigmoid(logits)
                total += loss.item() * len(Xb)
                n += len(Xb)
                all_prob.append(prob.cpu().numpy())
        prob = np.concatenate(all_prob).ravel()
        from sklearn.metrics import fbeta_score
        f2 = fbeta_score(y_val, (prob >= 0.5).astype(int), beta=2, zero_division=0)
        return total / n, float(f2)

    def predict_proba(self, X_np):
        """Return fraud probabilities for a feature matrix."""
        return self.model.predict_proba(X_np)

    def predict_anomaly(self, X_np, threshold):
        """For anomaly models, predict using reconstruction error threshold."""
        if isinstance(self.model, type(self.model)):
            pass
        return self.model.reconstruction_error(X_np) if hasattr(self.model, "reconstruction_error") \
            else self.model.anomaly_score(X_np)
