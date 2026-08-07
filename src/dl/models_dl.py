"""Deep Learning architectures (Approach 2).

Implements ten neural architectures for tabular fraud detection, all producing
a binary (fraud) probability. A unified base class provides the training and
prediction interface so the training pipeline can treat every model uniformly.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import setup_logging

logger = setup_logging()


def _focal_loss(logits, y, gamma: float = 2.0, alpha: float = 0.75):
    """Focal loss: down-weights easy examples, focuses on hard/rare cases."""
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = torch.where(y == 1, p, 1 - p)
    w = alpha * y + (1 - alpha) * (1 - y)
    return (w * (1 - pt) ** gamma * bce).mean()


class BaseNet(nn.Module, ABC):
    """Base class for all deep learning architectures.

    Subclasses implement `forward(X) -> logits`. The base class exposes
    `predict_proba`, `fit` and `evaluate` used by the training pipeline.
    """

    def __init__(self, input_dim: int, **kw):
        super().__init__()
        self.input_dim = input_dim
        self.criterion = "bce"          # bce | focal | weighted_bce
        self.anomaly = False            # True for AE/VAE (reconstruction-based)

    @abstractmethod
    def forward(self, X):
        """Forward pass returning logits (or anomaly score)."""

    def predict_proba(self, X_np: np.ndarray) -> np.ndarray:
        """Return fraud probabilities for a numpy feature matrix.

        Args:
            X_np: Feature matrix.

        Returns:
            np.ndarray: Fraud probabilities (n,).
        """
        self.eval()
        with torch.no_grad():
            logits = self(torch.tensor(X_np, dtype=torch.float32))
            return torch.sigmoid(logits).numpy().ravel()

    def compute_loss(self, logits, y, class_weights=None):
        """Compute the configured loss given logits and targets."""
        if self.criterion == "focal":
            return _focal_loss(logits, y, gamma=2.0, alpha=0.75)
        if self.criterion == "weighted_bce":
            if class_weights is not None:
                w = torch.tensor(class_weights, dtype=torch.float32)
                return F.binary_cross_entropy_with_logits(logits, y, pos_weight=w)
            return F.binary_cross_entropy_with_logits(logits, y)
        return F.binary_cross_entropy_with_logits(logits, y)


# --------------------------------------------------------------------------
# Architecture 1 - MLP
# --------------------------------------------------------------------------
class MLP(BaseNet):
    """Multi-layer perceptron with batch norm + dropout."""

    def __init__(self, input_dim, hidden=(256, 128, 64), dropout=0.3, **kw):
        super().__init__(input_dim, **kw)
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


# --------------------------------------------------------------------------
# Architecture 2 - Wide and Deep
# --------------------------------------------------------------------------
class WideAndDeep(BaseNet):
    """Linear (wide) + deep network with concatenated output."""

    def __init__(self, input_dim, deep=(128, 64, 32), dropout=0.2, **kw):
        super().__init__(input_dim, **kw)
        self.wide = nn.Linear(input_dim, 1)
        layers, prev = [], input_dim
        for h in deep:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.out = nn.Linear(prev + 1, 1)

    def forward(self, X):
        w = self.wide(X)
        d = self.deep(X)
        return self.out(torch.cat([w, d], dim=1))


# --------------------------------------------------------------------------
# Architecture 3 - DCN (Deep & Cross Network)
# --------------------------------------------------------------------------
class CrossLayer(nn.Module):
    """Explicit feature-interaction cross layer."""

    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        return x0 * self.w(x) + self.b + x


class DCN(BaseNet):
    """Deep & Cross network: cross network in parallel with a deep network."""

    def __init__(self, input_dim, cross_layers=3, deep=(128, 64), dropout=0.2, **kw):
        super().__init__(input_dim, **kw)
        self.cross_layers = nn.ModuleList(
            [CrossLayer(input_dim) for _ in range(cross_layers)])
        layers, prev = [], input_dim
        for h in deep:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.out = nn.Linear(input_dim + prev, 1)

    def forward(self, X):
        x0 = X
        for layer in self.cross_layers:
            X = layer(x0, X)
        d = self.deep(x0)
        return self.out(torch.cat([X, d], dim=1))


# --------------------------------------------------------------------------
# Architecture 4 - TabNet (simplified attention-based)
# --------------------------------------------------------------------------
class TabNet(BaseNet):
    """Attention-based feature selection network (TabNet-style)."""

    def __init__(self, input_dim, hidden=64, n_steps=3, gamma=1.3, **kw):
        super().__init__(input_dim, **kw)
        self.n_steps = n_steps
        self.gamma = gamma
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.attn = nn.Linear(hidden, input_dim)
        self.decision_transformer = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.out = nn.Linear(hidden * n_steps, 1)

    def forward(self, X):
        prior = torch.ones_like(X)
        feat = self.feature_transformer(X)        # [B, hidden]
        steps = []
        for _ in range(self.n_steps):
            mask = torch.softmax(self.attn(feat) * prior, dim=1)   # [B, input_dim]
            prior = prior * (self.gamma - mask)
            masked = X * mask
            step_out = self.decision_transformer(masked)          # [B, hidden]
            steps.append(step_out)
        concat = torch.cat(steps, dim=1)
        return self.out(concat)


# --------------------------------------------------------------------------
# Architecture 5 - Transformer
# --------------------------------------------------------------------------
class TransformerModel(BaseNet):
    """Self-attention transformer treating features as tokens."""

    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, **kw):
        super().__init__(input_dim, **kw)
        self.d_model = d_model
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, X):
        x = self.proj(X).unsqueeze(1)          # [B, 1, d]
        x = x + self.pos
        x = self.encoder(x)
        return self.head(x[:, 0, :])


# --------------------------------------------------------------------------
# Architecture 6 - ResNet (pre-activation residual blocks)
# --------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-activation residual block (BN -> ReLU -> Linear)."""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim), nn.Dropout(dropout),
            nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNet(BaseNet):
    """Deep residual network with skip connections."""

    def __init__(self, input_dim, hidden=128, blocks=3, dropout=0.2, **kw):
        super().__init__(input_dim, **kw)
        self.in_layer = nn.Linear(input_dim, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, X):
        x = F.relu(self.in_layer(X))
        return self.out(self.blocks(x))


# --------------------------------------------------------------------------
# Architecture 7 - NODE (simplified differentiable oblivious trees)
# --------------------------------------------------------------------------
class NODE(BaseNet):
    """Differentiable oblivious decision trees as a single neural layer."""

    def __init__(self, input_dim, n_trees=8, tree_depth=3, hidden=32, **kw):
        super().__init__(input_dim, **kw)
        self.n_trees = n_trees
        self.depth = tree_depth
        self.n_leaf = 2 ** tree_depth
        # per split: a learned linear combination of input features
        self.split_weights = nn.Parameter(
            torch.randn(n_trees, tree_depth, input_dim) * 0.1)
        self.split_bias = nn.Parameter(torch.zeros(n_trees, tree_depth))
        self.leaf_vals = nn.Parameter(
            torch.randn(n_trees, self.n_leaf, hidden) * 0.1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, X):
        # soft split probabilities per (tree, depth)  -> [B, n_trees, depth]
        scores = torch.tensordot(X, self.split_weights, dims=([1], [2])) + self.split_bias
        prob = torch.sigmoid(scores)
        B = X.size(0)
        leaf_prob = torch.ones(B, self.n_trees, self.n_leaf)
        leaf_idx = torch.arange(self.n_leaf).view(1, 1, -1).to(X.device)
        for d in range(self.depth):
            # leaf l is in the LEFT child of its depth-d node iff bit (depth-d-1) is 0
            left = ((leaf_idx >> (self.depth - d - 1)) & 1) == 0
            left = left.float()
            p = prob[:, :, d].unsqueeze(-1)           # [B, t, 1]
            leaf_prob = leaf_prob * ((1 - p) * left + p * (1 - left))
        out = torch.einsum("btl,tlh->bh", leaf_prob, self.leaf_vals)
        return self.out(out)


# --------------------------------------------------------------------------
# Architecture 8 - LSTM (sequential policyholder claim modelling)
# --------------------------------------------------------------------------
class LSTMNet(BaseNet):
    """LSTM over windowed feature sequences (padding-based)."""

    def __init__(self, input_dim, hidden=32, layers=1, **kw):
        super().__init__(input_dim, **kw)
        self.hidden = hidden
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, X):
        # treat batch as a single time step with feature dim -> unsqueeze
        x = X.unsqueeze(1)  # [B, 1, input_dim]
        out, _ = self.lstm(x)
        return self.out(out[:, -1, :])


# --------------------------------------------------------------------------
# Architecture 9 - Autoencoder (unsupervised anomaly detection)
# --------------------------------------------------------------------------
class Autoencoder(BaseNet):
    """Undercomplete autoencoder for reconstruction-based anomaly detection."""

    def __init__(self, input_dim, hidden=64, bottleneck=16, **kw):
        super().__init__(input_dim, **kw)
        self.anomaly = True
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, X):
        return self.decoder(self.encoder(X))

    def reconstruction_error(self, X_np: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X_np, dtype=torch.float32)
            recon = self(X)
            err = ((X - recon) ** 2).mean(dim=1).numpy()
        return err


# --------------------------------------------------------------------------
# Architecture 10 - VAE (probabilistic latent anomaly detection)
# --------------------------------------------------------------------------
class VAE(BaseNet):
    """Variational autoencoder (beta-VAE) for reconstruction + KL loss."""

    def __init__(self, input_dim, hidden=64, latent=16, beta=1.0, **kw):
        super().__init__(input_dim, **kw)
        self.anomaly = True
        self.beta = beta
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.mu_lin = nn.Linear(hidden, latent)
        self.logvar_lin = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X):
        h = self.enc(X)
        mu, logvar = self.mu_lin(h), self.logvar_lin(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def anomaly_score(self, X_np: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X_np, dtype=torch.float32)
            h = self.enc(X)
            mu, logvar = self.mu_lin(h), self.logvar_lin(h)
            z = mu
            recon = self.decoder(z)
            recon_err = ((X - recon) ** 2).mean(dim=1)
            kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=1)
            return (recon_err + self.beta * kl).numpy()


# Registry of model constructors and display names
ARCHITECTURES = {
    "MLP": (MLP, {}),
    "WideAndDeep": (WideAndDeep, {}),
    "DCN": (DCN, {}),
    "TabNet": (TabNet, {}),
    "Transformer": (TransformerModel, {}),
    "ResNet": (ResNet, {}),
    "NODE": (NODE, {}),
    "LSTM": (LSTMNet, {}),
    "Autoencoder": (Autoencoder, {}),
    "VAE": (VAE, {}),
}


def build_arch(name: str, input_dim: int, criterion: str = "bce", **overrides):
    """Build a fresh architecture instance.

    Args:
        name: Architecture key.
        input_dim: Number of input features.
        criterion: bce | weighted_bce | focal.
        overrides: Optional constructor overrides.

    Returns:
        BaseNet: A fresh model.
    """
    ctor, defaults = ARCHITECTURES[name]
    params = {**defaults, **overrides}
    model = ctor(input_dim=input_dim, **params)
    model.criterion = criterion
    return model
