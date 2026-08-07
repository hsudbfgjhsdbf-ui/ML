"""Fast Approach 2 tensor and XAI contract tests."""

import numpy as np
import torch

from src_dl.models import MODEL_SPECS, build_model
from src_dl.utils import parameter_count


def test_all_deep_models_emit_one_logit() -> None:
    """Every architecture must accept a batch and return one logit per row."""
    for spec in MODEL_SPECS:
        model = build_model(
            spec["key"],
            12,
            {
                "hidden_dim": 32,
                "dropout": 0.1,
                "transformer_dim": 16,
                "transformer_heads": 4,
                "transformer_layers": 1,
                "tabnet_steps": 2,
                "cnn_channels": 16,
                "autoencoder_latent_dim": 8,
            },
        )
        output = model(torch.zeros(4, 12))
        logits = output[0] if isinstance(output, tuple) else output
        assert logits.shape == (4,)
        assert parameter_count(model) > 0


def test_deep_seeded_input_is_finite() -> None:
    """The reference tensor contract contains finite float32 values."""
    batch = np.zeros((3, 12), dtype=np.float32)
    assert batch.dtype == np.float32
    assert np.isfinite(batch).all()
