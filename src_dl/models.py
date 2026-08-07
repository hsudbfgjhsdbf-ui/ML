"""Five deep tabular architectures required by Approach 2.

All models expose a single fraud logit for binary classification. The TabNet-
style masks, transformer tokens, and autoencoder reconstruction are auxiliary
signals for XAI and ablation; the common evaluator scores the same probability
contract for every architecture.
"""

from __future__ import annotations

from typing import Any
import warnings

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Batch-normalized feed-forward baseline for transformed tabular rows."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits and optionally an empty auxiliary dictionary."""
        logits = self.network(x).squeeze(-1)
        return (logits, {}) if return_aux else logits


class TabNetStyle(nn.Module):
    """Lightweight attentive decision-step model inspired by TabNet semantics."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, steps: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.steps = steps
        self.attention = nn.ModuleList(
            [nn.Sequential(nn.Linear(input_dim, input_dim), nn.LayerNorm(input_dim)) for _ in range(steps)]
        )
        self.transformers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                for _ in range(steps)
            ]
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply soft feature masks at each decision step."""
        prior = torch.ones_like(x)
        representations = []
        masks = []
        for attention, transform in zip(self.attention, self.transformers):
            mask = torch.softmax(attention(x) * prior, dim=-1)
            masked = x * mask
            representations.append(transform(masked))
            masks.append(mask)
            prior = prior * (1.0 - mask.detach())
        representation = torch.stack(representations, dim=0).sum(dim=0)
        logits = self.head(representation).squeeze(-1)
        aux = {"masks": torch.stack(masks, dim=1)}
        return (logits, aux) if return_aux else logits


class TabularCNN1D(nn.Module):
    """One-dimensional convolutional probe over the frozen feature order."""

    def __init__(self, input_dim: int, channels: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels // 2, 1))

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Treat the transformed columns as a one-dimensional sequence."""
        representation = self.convolution(x.unsqueeze(1))
        logits = self.head(representation).squeeze(-1)
        return (logits, {"representation": representation}) if return_aux else logits


class AutoencoderHybrid(nn.Module):
    """Legitimate-manifold reconstruction plus a supervised anomaly head."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 96) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
        self.head = nn.Linear(1, 1)

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a supervised logit based on per-row reconstruction error."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        error = (x - reconstruction).pow(2).mean(dim=1, keepdim=True)
        logits = self.head(error).squeeze(-1)
        aux = {"reconstruction": reconstruction, "reconstruction_error": error.squeeze(-1), "latent": latent}
        return (logits, aux) if return_aux else logits


class FeatureTransformer(nn.Module):
    """Feature-token transformer with mean pooling and pre-norm encoders."""

    def __init__(
        self, input_dim: int, d_model: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.15
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor")
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Project each scalar feature to a token and aggregate attention output."""
        tokens = self.input_projection(x.unsqueeze(-1)) + self.feature_embedding
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        logits = self.head(pooled).squeeze(-1)
        aux = {"tokens": encoded, "pooled": pooled}
        return (logits, aux) if return_aux else logits


def build_model(key: str, input_dim: int, config: dict[str, Any]) -> nn.Module:
    """Construct one of the five configured deep architectures.

    Args:
        key: Stable model key.
        input_dim: Frozen transformed feature width.
        config: `models` subsection of YAML.
    Returns:
        Newly initialized PyTorch module.
    Raises:
        ValueError: If an unknown key is requested.
    """
    hidden = int(config.get("hidden_dim", 128))
    dropout = float(config.get("dropout", 0.25))
    if key == "dl_a_mlp":
        return MLPClassifier(input_dim, hidden, dropout)
    if key == "dl_b_tabnet":
        return TabNetStyle(
            input_dim, hidden_dim=max(32, hidden // 2), steps=int(config.get("tabnet_steps", 4)), dropout=dropout
        )
    if key == "dl_c_cnn1d":
        return TabularCNN1D(input_dim, channels=int(config.get("cnn_channels", 64)), dropout=dropout)
    if key == "dl_d_autoencoder":
        return AutoencoderHybrid(
            input_dim, latent_dim=int(config.get("autoencoder_latent_dim", 16)), hidden_dim=max(32, hidden // 2)
        )
    if key == "dl_e_transformer":
        return FeatureTransformer(
            input_dim,
            d_model=int(config.get("transformer_dim", 32)),
            heads=int(config.get("transformer_heads", 4)),
            layers=int(config.get("transformer_layers", 2)),
            dropout=dropout,
        )
    raise ValueError(f"Unknown deep model key: {key}")


MODEL_SPECS = [
    {"key": "dl_a_mlp", "display_name": "MLP", "family": "dense"},
    {"key": "dl_b_tabnet", "display_name": "TabNet-style attentive network", "family": "attention"},
    {"key": "dl_c_cnn1d", "display_name": "1D convolutional tabular network", "family": "convolution"},
    {"key": "dl_d_autoencoder", "display_name": "Autoencoder anomaly hybrid", "family": "anomaly"},
    {"key": "dl_e_transformer", "display_name": "Feature-token transformer", "family": "transformer"},
]
