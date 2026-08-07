# Training protocol

Seeds: `[42, 43, 44]`. Optimizer: AdamW. Loss: weighted
BCE-with-logits; autoencoder adds a documented reconstruction term. Gradient
clipping norm: 1.0. Checkpoints
restore the best validation PR-AUC epoch. Test rows are not used for early
stopping, model selection, or XAI selection.
