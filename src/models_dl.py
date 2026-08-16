"""
Deep Learning Architectures for Tabular and Sequential Claim Verification.
Implements 10 neural architectures in PyTorch:
1. MLP with Skip Connections
2. Wide and Deep Network
3. Deep & Cross Network (DCN)
4. TabNet Sequential Attention
5. Tabular FT-Transformer
6. Tabular ResNet
7. NODE (Neural Oblivious Decision Ensembles)
8. BiLSTM with Temporal Attention
9. Autoencoder Anomaly Detector
10. Variational Autoencoder (VAE)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional

# -------------------------------------------------------------
# 1. Multi-Layer Perceptron (MLP with Residual Connections)
# -------------------------------------------------------------
class TabularMLP(nn.Module):
    """Deep Multi-Layer Perceptron with Batch Normalization, Dropout, and Skip Connections."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout_rate: float = 0.3,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        
        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))
            self.norms.append(nn.BatchNorm1d(h_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            if use_residual and prev_dim != h_dim:
                self.skip_projections.append(nn.Linear(prev_dim, h_dim))
            elif use_residual:
                self.skip_projections.append(nn.Identity())
            prev_dim = h_dim
            
        self.head = nn.Linear(prev_dim, 1)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        curr = x
        for i, layer in enumerate(self.layers):
            out = layer(curr)
            out = self.norms[i](out)
            out = F.relu(out)
            out = self.dropouts[i](out)
            if self.use_residual:
                out = out + self.skip_projections[i](curr)
            curr = out
        logits = self.head(curr)
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 2. Wide and Deep Network
# -------------------------------------------------------------
class WideAndDeep(nn.Module):
    """Wide linear model for memorization combined with deep dense layers for generalization."""
    def __init__(self, input_dim: int, deep_dims: List[int] = [128, 64, 32], dropout: float = 0.25):
        super().__init__()
        # Wide component
        self.wide_linear = nn.Linear(input_dim, 1)
        
        # Deep component
        deep_layers = []
        prev = input_dim
        for d in deep_dims:
            deep_layers.append(nn.Linear(prev, d))
            deep_layers.append(nn.BatchNorm1d(d))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout))
            prev = d
        self.deep_net = nn.Sequential(*deep_layers)
        self.deep_head = nn.Linear(prev, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_out = self.wide_linear(x)
        deep_repr = self.deep_net(x)
        deep_out = self.deep_head(deep_repr)
        logits = wide_out + deep_out
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 3. Deep & Cross Network (DCN)
# -------------------------------------------------------------
class CrossNetworkLayer(nn.Module):
    """Computes explicit bounded-degree feature interactions: x_{l+1} = x_0 * (W_l * x_l) + b_l + x_l."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # xl: (batch, dim), weight: (dim, 1) -> xl @ weight: (batch, 1)
        xl_w = torch.matmul(xl, self.weight)
        return x0 * xl_w + self.bias + xl

class DeepAndCrossNetwork(nn.Module):
    """DCN combining explicit cross layers with deep non-linear MLP."""
    def __init__(self, input_dim: int, num_cross_layers: int = 3, deep_dims: List[int] = [128, 64]):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossNetworkLayer(input_dim) for _ in range(num_cross_layers)])
        
        deep_layers = []
        prev = input_dim
        for d in deep_dims:
            deep_layers.append(nn.Linear(prev, d))
            deep_layers.append(nn.BatchNorm1d(d))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(0.2))
            prev = d
        self.deep_net = nn.Sequential(*deep_layers)
        
        # Combine cross output and deep output
        combined_dim = input_dim + prev
        self.final_head = nn.Linear(combined_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for cross in self.cross_layers:
            xl = cross(x0, xl)
            
        deep_out = self.deep_net(x)
        combined = torch.cat([xl, deep_out], dim=-1)
        logits = self.final_head(combined)
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 4. TabNet Sequential Attention Network
# -------------------------------------------------------------
class GhostBatchNorm(nn.Module):
    """Ghost Batch Normalization for virtual mini-batch regularization."""
    def __init__(self, num_features: int, virtual_batch_size: int = 32):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.size(0) > self.virtual_batch_size:
            chunks = x.chunk(int(math.ceil(x.size(0) / self.virtual_batch_size)), dim=0)
            res = [self.bn(c) for c in chunks]
            return torch.cat(res, dim=0)
        return self.bn(x)

class TabNetModel(nn.Module):
    """Sequential Attention TabNet Architecture for Interpretable Tabular Learning."""
    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 32,
        output_dim: int = 1,
        num_steps: int = 3,
        gamma: float = 1.3
    ):
        super().__init__()
        self.num_steps = num_steps
        self.gamma = gamma
        self.input_dim = input_dim
        
        # Initial transformation
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # Step components
        self.attention_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                GhostBatchNorm(input_dim),
                nn.Sigmoid()
            ) for _ in range(num_steps)
        ])
        
        self.feature_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                GhostBatchNorm(feature_dim),
                nn.GLU(dim=-1),
                nn.Linear(feature_dim // 2, feature_dim),
                GhostBatchNorm(feature_dim),
                nn.ReLU()
            ) for _ in range(num_steps)
        ])
        
        self.final_head = nn.Linear(feature_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_norm = self.initial_bn(x)
        batch_size = x.size(0)
        
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)
        aggregated_steps = 0.0
        mask_history = []
        
        for step in range(self.num_steps):
            # Attention masking
            mask = self.attention_transformers[step](x_norm) * prior_scales
            mask_history.append(mask)
            
            # Feature transformation
            masked_features = x_norm * mask
            step_repr = self.feature_transformers[step](masked_features)
            aggregated_steps = aggregated_steps + step_repr
            
            # Update prior scale
            prior_scales = prior_scales * (self.gamma - mask)
            
        logits = self.final_head(aggregated_steps)
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 5. Tabular FT-Transformer (Feature Tokenizer Transformer)
# -------------------------------------------------------------
class TabularTransformer(nn.Module):
    """Treats tabular feature subsets as tokens with learned multi-head self-attention and CLS token."""
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Fast grouped linear projection: (batch, input_dim) -> (batch, input_dim, embed_dim)
        self.dense_proj = nn.Linear(input_dim, 16 * embed_dim)
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer Encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Fast reshape to sequence of 16 tokens each of embed_dim
        proj = self.dense_proj(x).view(batch_size, 16, self.embed_dim)
        
        # Prepend [CLS] token
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_expanded, proj], dim=1) # (batch, 17, embed_dim)
        
        out = self.transformer(seq)
        cls_rep = out[:, 0, :] # Extract [CLS] token output
        logits = self.head(cls_rep)
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 6. Tabular ResNet
# -------------------------------------------------------------
class ResNetBlock(nn.Module):
    """Pre-activation Tabular Residual Block: BN -> ReLU -> Dropout -> Linear -> BN -> ReLU -> Linear."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.fc1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.fc2(out)
        return residual + out

class TabularResNet(nn.Module):
    """Deep Tabular ResNet with pre-activation blocks."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.final_bn = nn.BatchNorm1d(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        out = F.relu(self.final_bn(out))
        logits = self.head(out)
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 7. Neural Oblivious Decision Ensembles (NODE)
# -------------------------------------------------------------
class ObliviousTreeLayer(nn.Module):
    """Fast differentiable oblivious tree layer with soft routing."""
    def __init__(self, input_dim: int, tree_depth: int = 4, num_trees: int = 8, temperature: float = 0.1):
        super().__init__()
        self.tree_depth = tree_depth
        self.num_trees = num_trees
        self.temperature = temperature
        
        self.split_linear = nn.Linear(input_dim, num_trees * tree_depth)
        self.leaves = nn.Parameter(torch.randn(num_trees, 2 ** tree_depth) * 0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        splits = self.split_linear(x).view(batch_size, self.num_trees, self.tree_depth)
        split_probs = torch.sigmoid(splits / self.temperature)
        
        p0 = split_probs[:, :, 0:1]
        leaf_weights = torch.cat([1.0 - p0, p0], dim=-1)
        
        for d in range(1, self.tree_depth):
            pd = split_probs[:, :, d:d+1]
            left = leaf_weights * (1.0 - pd)
            right = leaf_weights * pd
            leaf_weights = torch.cat([left, right], dim=-1)
            
        tree_outputs = (leaf_weights * self.leaves.unsqueeze(0)).sum(dim=-1)
        return tree_outputs.mean(dim=1, keepdim=True)

class NODEClassifier(nn.Module):
    """Neural Oblivious Decision Ensemble with hierarchical differentiable tree layers."""
    def __init__(self, input_dim: int, num_layers: int = 2, tree_depth: int = 4, num_trees: int = 20):
        super().__init__()
        self.tree_layers = nn.ModuleList([
            ObliviousTreeLayer(input_dim, tree_depth, num_trees) for _ in range(num_layers)
        ])
        self.head = nn.Linear(num_layers, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        for layer in self.tree_layers:
            layer_outputs.append(layer(x))
        combined = torch.cat(layer_outputs, dim=-1)
        logits = self.head(combined)
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 8. LSTM with Temporal Attention (Sequential Claim Modeler)
# -------------------------------------------------------------
class BiLSTMTemporalAttention(nn.Module):
    """BiLSTM with Self-Attention over sequential claimant history."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.head = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If 2D (batch, features), treat as sequence of length 1
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        lstm_out, _ = self.lstm(x) # (batch, seq_len, 2 * hidden_dim)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # (batch, seq_len, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1) # (batch, 2 * hidden_dim)
        logits = self.head(context)
        return torch.sigmoid(logits).squeeze(-1)

# -------------------------------------------------------------
# 9. Autoencoder Anomaly Detector
# -------------------------------------------------------------
class TabularAutoencoder(nn.Module):
    """Unsupervised Autoencoder trained on legitimate claims; anomaly detection via reconstruction error."""
    def __init__(self, input_dim: int, bottleneck_dim: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
        
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample MSE reconstruction error as fraud anomaly score."""
        with torch.no_grad():
            recon = self.forward(x)
            mse = torch.mean((x - recon) ** 2, dim=-1)
            # Map MSE to 0-1 probability via sigmoid
            prob = torch.sigmoid((mse - mse.mean()) / (mse.std() + 1e-5))
            return prob

# -------------------------------------------------------------
# 10. Variational Autoencoder (VAE)
# -------------------------------------------------------------
class TabularVAE(nn.Module):
    """Variational Autoencoder with reparameterization trick and ELBO loss for generation and anomaly detection."""
    def __init__(self, input_dim: int, latent_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Shared Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
        
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        return recon_loss + beta * kl_div
