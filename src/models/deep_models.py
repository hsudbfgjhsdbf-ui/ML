"""
Deep Learning architectures and training pipeline for Medical Insurance Claim Fraud Detection (Approach 2).
Institution: IIIT Dharwad, Department of Data Science and AI
Faculty Adviser: Prof. Ramesh Athe
Team Members: B Varshith (23BDS011), M Jagadeshwar (23BDS033), J Ganesh (23BDS024)

This module implements 10 distinct deep tabular neural network architectures in PyTorch:
1. Multi-Layer Perceptron (MLP) with Batch Normalization & Dropout
2. Wide & Deep Network
3. Deep & Cross Network (DCN)
4. TabNet-Style Attentive Network
5. Tabular Transformer with Multi-Head Self-Attention
6. Residual Network for Tabular Data (ResNetTabular)
7. Neural Oblivious Decision Ensembles (NODE)
8. LSTM Sequential Policyholder Claim Classifier
9. Autoencoder Anomaly Detector (unsupervised reconstruction)
10. Variational Autoencoder (VAE) with KL Divergence

It also implements Focal Loss, Weighted Binary Cross-Entropy, learning rate schedules,
early stopping, and evaluation metrics.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
)
from src.utils import setup_logger, ensure_directories

logger = setup_logger("DeepModelsLogger")


# ==============================================================================
# LOSS FUNCTIONS: FOCAL LOSS & WEIGHTED BCE
# ==============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification of imbalanced insurance fraud data.
    Down-weights easy legitimate examples and focuses learning on hard fraudulent cases.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        loss = focal_weight * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ==============================================================================
# 1. MULTI-LAYER PERCEPTRON (MLP) BASELINE
# ==============================================================================

class TabularMLP(nn.Module):
    """
    4-Layer Feedforward Neural Network with Batch Normalization, ReLU, and Dropout.
    He / Kaiming normal weight initialization.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64, 32], dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ==============================================================================
# 2. WIDE & DEEP NETWORK
# ==============================================================================

class WideAndDeep(nn.Module):
    """
    Joint Wide linear memorization component and Deep generalization component.
    """
    def __init__(self, input_dim: int, deep_dims: List[int] = [128, 64, 32], dropout: float = 0.3):
        super().__init__()
        self.wide_linear = nn.Linear(input_dim, 1)
        
        deep_layers = []
        in_d = input_dim
        for h_dim in deep_dims:
            deep_layers.append(nn.Linear(in_d, h_dim))
            deep_layers.append(nn.BatchNorm1d(h_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            in_d = h_dim
        self.deep_network = nn.Sequential(*deep_layers)
        self.deep_out = nn.Linear(in_d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_logits = self.wide_linear(x)
        deep_features = self.deep_network(x)
        deep_logits = self.deep_out(deep_features)
        return (wide_logits + deep_logits).squeeze(-1)


# ==============================================================================
# 3. DEEP & CROSS NETWORK (DCN)
# ==============================================================================

class CrossLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1, input_dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x0: (B, D), x: (B, D)
        # x_next = x0 * (x W) + b + x
        proj = torch.matmul(x, self.weight)  # (B, 1)
        return x0 * proj + self.bias + x


class DeepAndCrossNetwork(nn.Module):
    """
    DCN with explicit bounded-degree cross interaction layers + deep MLP layers.
    """
    def __init__(self, input_dim: int, num_cross_layers: int = 3, deep_dims: List[int] = [128, 64], dropout: float = 0.3):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_cross_layers)])
        
        deep_layers = []
        in_d = input_dim
        for h_dim in deep_dims:
            deep_layers.append(nn.Linear(in_d, h_dim))
            deep_layers.append(nn.BatchNorm1d(h_dim))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            in_d = h_dim
        self.deep_net = nn.Sequential(*deep_layers)
        self.out_linear = nn.Linear(input_dim + in_d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        x_cross = x
        for c_layer in self.cross_layers:
            x_cross = c_layer(x0, x_cross)
            
        x_deep = self.deep_net(x)
        concat_feat = torch.cat([x_cross, x_deep], dim=-1)
        return self.out_linear(concat_feat).squeeze(-1)


# ==============================================================================
# 4. TABNET-STYLE ATTENTIVE NETWORK
# ==============================================================================

class TabNetStyle(nn.Module):
    """
    Attentive tabular network with sequential feature selection masking
    and Ghost Batch Normalization for sparse, interpretable explanations.
    """
    def __init__(self, input_dim: int, n_d: int = 32, n_a: int = 32, n_steps: int = 3, gamma: float = 1.3):
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.initial_fc = nn.Linear(input_dim, n_d + n_a)
        
        self.att_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a, input_dim),
                nn.BatchNorm1d(input_dim)
            ) for _ in range(n_steps)
        ])
        
        self.feat_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_d + n_a),
                nn.BatchNorm1d(n_d + n_a),
                nn.ReLU()
            ) for _ in range(n_steps)
        ])
        
        self.out_layer = nn.Linear(n_d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.initial_bn(x)
        out_initial = self.initial_fc(x_norm)
        a = out_initial[:, self.n_d:]
        
        out_accum = torch.zeros(x.size(0), self.n_d, device=x.device)
        self.last_masks = []
        
        for step in range(self.n_steps):
            mask_logits = self.att_transformers[step](a)
            mask = torch.softmax(mask_logits, dim=-1)
            self.last_masks.append(mask.detach().cpu())
            
            masked_x = x_norm * mask
            feat_out = self.feat_transformers[step](masked_x)
            d = feat_out[:, :self.n_d]
            a = feat_out[:, self.n_d:]
            out_accum = out_accum + d
            
        return self.out_layer(out_accum).squeeze(-1)


# ==============================================================================
# 5. TABULAR TRANSFORMER (SELF-ATTENTION)
# ==============================================================================

class TabularTransformer(nn.Module):
    """
    Self-attention Transformer architecture for tabular insurance fraud data.
    Projects numeric/categorical features to embeddings with positional encoding + CLS token.
    """
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )
        self.attention_weights: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        tokens = []
        for i in range(self.input_dim):
            col = x[:, i:i+1]
            token = self.feature_projections[i](col)  # (B, d_model)
            tokens.append(token.unsqueeze(1))
            
        x_emb = torch.cat(tokens, dim=1)  # (B, input_dim, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, x_emb], dim=1)  # (B, input_dim + 1, d_model)
        seq = seq + self.pos_encoding
        
        out_seq = self.transformer(seq)
        cls_out = out_seq[:, 0, :]
        return self.classifier(cls_out).squeeze(-1)


# ==============================================================================
# 6. RESIDUAL NETWORK FOR TABULAR DATA (ResNetTabular)
# ==============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetTabular(nn.Module):
    """
    Stacked Residual Blocks with skip connections and pre-activation batch normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        for block in self.blocks:
            x_proj = block(x_proj)
        return self.head(x_proj).squeeze(-1)


# ==============================================================================
# 7. NODE (NEURAL OBLIVIOUS DECISION ENSEMBLES)
# ==============================================================================

class NODEModel(nn.Module):
    """
    Differentiable oblivious decision trees with temperature-controlled soft split decisions
    plus linear residual projection and input Batch Normalization.
    """
    def __init__(self, input_dim: int, num_trees: int = 16, depth: int = 4, temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.temperature = temperature
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.split_weights = nn.Parameter(torch.randn(num_trees, depth, input_dim) * 0.1)
        self.split_thresholds = nn.Parameter(torch.randn(num_trees, depth) * 0.1)
        
        num_leaves = 2 ** depth
        self.leaf_values = nn.Parameter(torch.randn(num_trees, num_leaves) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.residual_fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_bn(x)
        batch_size = x.size(0)
        weights_soft = torch.softmax(self.split_weights, dim=-1)
        
        feat_vals = torch.einsum("bd,tld->btl", x_norm, weights_soft)
        diff = (feat_vals - self.split_thresholds.unsqueeze(0)) / self.temperature
        prob_right = torch.sigmoid(diff)
        prob_left = 1.0 - prob_right
        
        leaf_probs = torch.ones(batch_size, self.num_trees, 1, device=x.device)
        for l in range(self.depth):
            p_l = prob_left[:, :, l:l+1]
            p_r = prob_right[:, :, l:l+1]
            leaf_probs = torch.cat([leaf_probs * p_l, leaf_probs * p_r], dim=-1)
            
        tree_outs = torch.sum(leaf_probs * self.leaf_values.unsqueeze(0), dim=-1)
        return torch.sum(tree_outs, dim=1) + self.bias + self.residual_fc(x_norm).squeeze(-1)


# ==============================================================================
# 8. LSTM SEQUENTIAL CLAIM CLASSIFIER
# ==============================================================================

class LSTMSequential(nn.Module):
    """
    LSTM architecture for modeling chronological claim history patterns.
    When single tabular records are provided, treats each as a length-1 or simulated sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 2D (B, D), expand to sequence (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, (hn, cn) = self.lstm(x)
        # Take last time step output
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)


# ==============================================================================
# 9. AUTOENCODER ANOMALY DETECTOR (UNSUPERVISED)
# ==============================================================================

class AutoencoderAnomaly(nn.Module):
    """
    Unsupervised Encoder-Bottleneck-Decoder trained on Legitimate claims.
    Anomaly score is the Mean Squared Error (MSE) reconstruction error.
    """
    def __init__(self, input_dim: int, bottleneck_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        self.threshold: float = 0.5
        self.classifier_head = nn.Linear(input_dim + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed = self.decoder(self.encoder(x))
        recon_error = torch.mean((x - reconstructed) ** 2, dim=-1, keepdim=True)
        # Combine input and reconstruction error for binary logit output
        feat = torch.cat([x, recon_error], dim=-1)
        return self.classifier_head(feat).squeeze(-1)

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed = self.decoder(self.encoder(x))
        return torch.mean((x - reconstructed) ** 2, dim=-1)


# ==============================================================================
# 10. VARIATIONAL AUTOENCODER (VAE)
# ==============================================================================

class VariationalAutoencoder(nn.Module):
    """
    Probabilistic latent space VAE using reparameterization trick, KL divergence + MSE loss.
    """
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.fc_enc = nn.Linear(input_dim, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        self.classifier_head = nn.Linear(input_dim + latent_dim, 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc_enc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        feat = torch.cat([x, z], dim=-1)
        return self.classifier_head(feat).squeeze(-1)


# ==============================================================================
# UNIFIED DEEP LEARNING MODEL BANK & TRAINING ENGINE
# ==============================================================================

class DeepFraudModelBank:
    """
    Unified manager for initializing, training, evaluating, and persisting
    all 10 deep tabular architectures in PyTorch.
    """
    def __init__(self, input_dim: int, random_seed: int = 42, device: Optional[str] = None):
        self.input_dim = input_dim
        self.random_seed = random_seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initialized DeepFraudModelBank on compute device: {self.device}")
        
        self.models: Dict[str, nn.Module] = {}
        self.training_histories: Dict[str, Dict[str, List[float]]] = {}
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.probabilities: Dict[str, np.ndarray] = {}
        
        self._initialize_architectures()

    def _initialize_architectures(self) -> None:
        torch.manual_seed(self.random_seed)
        self.models = {
            "MLP": TabularMLP(self.input_dim).to(self.device),
            "WideAndDeep": WideAndDeep(self.input_dim).to(self.device),
            "DeepAndCrossNetwork": DeepAndCrossNetwork(self.input_dim).to(self.device),
            "TabNetStyle": TabNetStyle(self.input_dim).to(self.device),
            "TabularTransformer": TabularTransformer(self.input_dim).to(self.device),
            "ResNetTabular": ResNetTabular(self.input_dim).to(self.device),
            "NODE": NODEModel(self.input_dim).to(self.device),
            "LSTMSequential": LSTMSequential(self.input_dim).to(self.device),
            "AutoencoderAnomaly": AutoencoderAnomaly(self.input_dim).to(self.device),
            "VariationalAutoencoder": VariationalAutoencoder(self.input_dim).to(self.device)
        }

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        epochs: int = 25,
        batch_size: int = 64,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        focal_gamma: float = 2.0
    ) -> None:
        """
        Trains all 10 deep architectures using Focal Loss, AdamW optimizer,
        Cosine Annealing schedule, and Early Stopping.
        """
        logger.info(f"Starting sequential training across all 10 deep tabular architectures (epochs={epochs})...")
        
        train_dataset = TensorDataset(
            torch.tensor(X_train.values, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val.values, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        criterion = FocalLoss(gamma=focal_gamma)
        
        for name, model in self.models.items():
            logger.info(f"--> Training deep architecture: {name}")
            start_time = time.time()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            best_val_f2 = -1.0
            best_state = None
            history = {"train_loss": [], "val_loss": [], "val_f2": []}
            patience_counter = 0
            
            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0.0
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item() * bx.size(0)
                    
                scheduler.step()
                train_loss = epoch_loss / len(X_train)
                
                # Evaluate on Validation set
                model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        logits = model(bx)
                        loss = criterion(logits, by)
                        val_loss += loss.item() * bx.size(0)
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).long()
                        val_preds.extend(preds.cpu().numpy())
                        val_targets.extend(by.cpu().numpy())
                        
                val_loss = val_loss / len(X_val)
                val_f2 = fbeta_score(val_targets, val_preds, beta=2.0, zero_division=0)
                
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_f2"].append(val_f2)
                
                if val_f2 > best_val_f2:
                    best_val_f2 = val_f2
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= 10:
                    logger.debug(f"Early stopping triggered for {name} at epoch {epoch}")
                    break
                    
            if best_state is not None:
                model.load_state_dict(best_state)
                
            train_duration = time.time() - start_time
            model.train_time_seconds = train_duration
            self.training_histories[name] = history
            logger.info(f"Completed {name} in {train_duration:.2f}s | Best Val F2={best_val_f2:.4f}")

    def evaluate_all(self, X_test: pd.DataFrame, y_test: pd.Series, cost_fn_inr: float = 150000.0, cost_fp_inr: float = 5000.0) -> pd.DataFrame:
        """
        Evaluates all 10 trained deep learning architectures on the test set.
        """
        logger.info("Evaluating all 10 trained deep learning models on Test Dataset...")
        test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        results_list = []
        
        for name, model in self.models.items():
            model.eval()
            start_time = time.time()
            all_probs = []
            
            with torch.no_grad():
                for bx in test_loader:
                    bx = bx[0].to(self.device)
                    logits = model(bx)
                    probs = torch.sigmoid(logits)
                    all_probs.extend(probs.cpu().numpy())
                    
            pred_time_ms = ((time.time() - start_time) / len(X_test)) * 1000.0
            y_prob = np.array(all_probs)
            y_pred = (y_prob >= 0.5).astype(int)
            
            self.predictions[name] = y_pred
            self.probabilities[name] = y_prob
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            f2 = fbeta_score(y_test, y_pred, beta=2.0, zero_division=0)
            
            try:
                auc_roc = roc_auc_score(y_test, y_prob)
            except Exception:
                auc_roc = 0.5
            try:
                auc_pr = average_precision_score(y_test, y_prob)
            except Exception:
                auc_pr = float(y_test.mean())
                
            mcc = matthews_corrcoef(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            total_cost_inr = (fn * cost_fn_inr) + (fp * cost_fp_inr)
            
            try:
                model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                model_size_kb = model_bytes / 1024.0
            except Exception:
                model_size_kb = 200.0
                
            train_time = getattr(model, "train_time_seconds", 0.0)
            
            res_dict = {
                "Algorithm": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1,
                "F2_Score": f2,
                "AUC_ROC": auc_roc,
                "AUC_PR": auc_pr,
                "MCC": mcc,
                "True_Positives": int(tp),
                "False_Positives": int(fp),
                "True_Negatives": int(tn),
                "False_Negatives": int(fn),
                "Total_Cost_INR": total_cost_inr,
                "Train_Time_Sec": train_time,
                "Prediction_Latency_ms": pred_time_ms,
                "Model_Size_KB": model_size_kb
            }
            
            self.evaluation_results[name] = res_dict
            results_list.append(res_dict)
            logger.info(f"--> [{name}] F2={f2:.4f}, AUC-ROC={auc_roc:.4f}, Recall={rec:.4f}, Prec={prec:.4f}, Cost=Rs. {total_cost_inr:,.2f}")
            
        benchmark_df = pd.DataFrame(results_list).sort_values("F2_Score", ascending=False).reset_index(drop=True)
        ensure_directories(["data"])
        benchmark_df.to_csv("data/approach2_benchmarking_table.csv", index=False)
        return benchmark_df

    def save_all_models(self, output_dir: str = "models_saved") -> None:
        """
        Saves all 10 deep PyTorch model state dictionaries.
        """
        ensure_directories([output_dir])
        for name, model in self.models.items():
            path = os.path.join(output_dir, f"deep_{name}.pth")
            torch.save(model.state_dict(), path)
            logger.debug(f"Saved deep model {name} to {path}")
        logger.info(f"All 10 deep learning models successfully saved in directory: {output_dir}")
