"""
Deep Learning Training Engine with Focal Loss, Temperature Scaling, and MC Dropout.
Trains and evaluates all 10 tabular neural architectures.
"""

import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Tuple, List, Optional

from src.config import config, RANDOM_SEED
from src.utils import (
    logger, compute_all_metrics, find_optimal_threshold_f2,
    save_model_artifacts
)
from src.models_dl import (
    TabularMLP, WideAndDeep, DeepAndCrossNetwork, TabNetModel,
    TabularTransformer, TabularResNet, NODEClassifier,
    BiLSTMTemporalAttention, TabularAutoencoder, TabularVAE
)

# -------------------------------------------------------------
# Focal Loss Implementation
# -------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss down-weights easy examples and focuses on hard ambiguous fraud cases:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred_prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(pred_prob, 1e-7, 1.0 - 1e-7)
        target = target.float()
        
        # Binary focal loss
        loss = - (
            self.alpha * ((1.0 - p) ** self.gamma) * target * torch.log(p) +
            (1.0 - self.alpha) * (p ** self.gamma) * (1.0 - target) * torch.log(1.0 - p)
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# -------------------------------------------------------------
# Temperature Scaling for Model Calibration
# -------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Learned temperature parameter T > 0 on validation set logits for post-hoc calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=0.01)

def apply_mixup_augmentation(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates convex combinations of random pairs of training samples."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1.0 - lam) * x[index]
    mixed_y = lam * y + (1.0 - lam) * y[index]
    return mixed_x, mixed_y

def train_single_dl_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.0015,
    weight_decay: float = 1e-4,
    use_focal_loss: bool = False,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Executes standard PyTorch training with Cosine Annealing, Early Stopping, and Gradient Clipping.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCELoss()
    
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience = 12
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if isinstance(model, TabularAutoencoder):
                recon = model(batch_x)
                loss = F.mse_loss(recon, batch_x)
            elif isinstance(model, TabularVAE):
                recon, mu, logvar = model(batch_x)
                loss = model.loss_function(recon, batch_x, mu, logvar)
            else:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = total_train_loss / max(1, len(train_loader))
        
        # Validation Step
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                if isinstance(model, TabularAutoencoder):
                    vrecon = model(vx)
                    vloss = F.mse_loss(vrecon, vx)
                elif isinstance(model, TabularVAE):
                    vrecon, vmu, vlogvar = model(vx)
                    vloss = model.loss_function(vrecon, vx, vmu, vlogvar)
                else:
                    vpreds = model(vx)
                    vloss = criterion(vpreds, vy)
                total_val_loss += vloss.item()
                
        avg_val_loss = total_val_loss / max(1, len(val_loader))
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.debug(f"{model_name}: Early stopping triggered at epoch {epoch+1}.")
                break
                
    model.load_state_dict(best_weights)
    return model, history

def evaluate_dl_uncertainty_mc_dropout(
    model: nn.Module,
    x: torch.Tensor,
    num_passes: int = 40,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes mean prediction and epistemic uncertainty (variance) using Monte Carlo Dropout.
    """
    model.train() # Enable dropout during evaluation
    x = x.to(device)
    passes = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            if isinstance(model, TabularAutoencoder):
                prob = model.get_anomaly_score(x)
            elif isinstance(model, TabularVAE):
                recon, _, _ = model(x)
                mse = torch.mean((x - recon) ** 2, dim=-1)
                prob = torch.sigmoid((mse - mse.mean()) / (mse.std() + 1e-5))
            else:
                prob = model(x)
            passes.append(prob.cpu().numpy())
            
    passes_arr = np.stack(passes, axis=0) # (num_passes, N)
    mean_prob = np.mean(passes_arr, axis=0)
    var_uncertainty = np.var(passes_arr, axis=0)
    return mean_prob, var_uncertainty

def evaluate_adversarial_fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    device: str = "cpu"
) -> float:
    """
    Evaluates adversarial robustness using Fast Gradient Sign Method (FGSM).
    Returns accuracy degradation under perturbation.
    """
    if isinstance(model, (TabularAutoencoder, TabularVAE)):
        return 0.0
        
    model.eval()
    x = x.clone().detach().to(device).requires_grad_(True)
    y = y.to(device).float()
    
    preds = model(x)
    loss = F.binary_cross_entropy(preds, y)
    model.zero_grad()
    loss.backward()
    
    # Perturb
    x_adv = x + epsilon * x.grad.sign()
    with torch.no_grad():
        adv_preds = model(x_adv)
        adv_labels = (adv_preds >= 0.5).long()
        clean_labels = (preds >= 0.5).long()
        accuracy_drop = float((clean_labels != adv_labels).float().mean().item())
        
    return accuracy_drop

def train_and_evaluate_dl_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Master pipeline training and evaluating all 10 Deep Learning tabular architectures.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Initiating Deep Learning benchmark with device: {device}")
    
    # Prepare PyTorch DataLoaders with optimized batch size
    batch_size = 256
    t_x = torch.tensor(X_train, dtype=torch.float32)
    t_y = torch.tensor(y_train, dtype=torch.float32)
    v_x = torch.tensor(X_val, dtype=torch.float32)
    v_y = torch.tensor(y_val, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)
    
    # For autoencoder, train predominantly on legitimate claims (class 0)
    legit_mask = (y_train == 0)
    ae_train_x = t_x[legit_mask]
    
    train_loader = DataLoader(TensorDataset(t_x, t_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(v_x, v_y), batch_size=batch_size, shuffle=False)
    ae_loader = DataLoader(TensorDataset(ae_train_x, torch.zeros(len(ae_train_x))), batch_size=batch_size, shuffle=True)
    
    input_dim = X_train.shape[1]
    
    # 10 Deep Learning Model Definitions
    dl_catalog = {
        "Tabular_MLP": TabularMLP(input_dim=input_dim),
        "Wide_and_Deep": WideAndDeep(input_dim=input_dim),
        "Deep_and_Cross_DCN": DeepAndCrossNetwork(input_dim=input_dim),
        "TabNet_Attention": TabNetModel(input_dim=input_dim),
        "Tabular_FT_Transformer": TabularTransformer(input_dim=input_dim),
        "Tabular_ResNet": TabularResNet(input_dim=input_dim),
        "NODE_Differentiable_Trees": NODEClassifier(input_dim=input_dim, num_trees=8),
        "BiLSTM_Temporal_Attention": BiLSTMTemporalAttention(input_dim=input_dim),
        "Autoencoder_Anomaly_Detector": TabularAutoencoder(input_dim=input_dim),
        "Variational_Autoencoder_VAE": TabularVAE(input_dim=input_dim)
    }
    
    dl_results = {}
    test_probabilities = {}
    test_predictions = {}
    uncertainties = {}
    robustness_scores = {}
    
    for name, model in dl_catalog.items():
        logger.info(f"--- Training Deep Learning Architecture: {name} ---")
        t0 = time.time()
        
        loader_to_use = ae_loader if "Autoencoder" in name or "VAE" in name else train_loader
        trained_model, history = train_single_dl_model(
            model=model,
            model_name=name,
            train_loader=loader_to_use,
            val_loader=val_loader,
            epochs=10,
            device=device
        )
        train_time = time.time() - t0
        
        # Validation threshold tuning
        trained_model.eval()
        with torch.no_grad():
            if isinstance(trained_model, TabularAutoencoder):
                val_probs = trained_model.get_anomaly_score(v_x.to(device)).cpu().numpy()
            elif isinstance(trained_model, TabularVAE):
                vrecon, _, _ = trained_model(v_x.to(device))
                vmse = torch.mean((v_x.to(device) - vrecon) ** 2, dim=-1)
                val_probs = torch.sigmoid((vmse - vmse.mean()) / (vmse.std() + 1e-5)).cpu().numpy()
            else:
                val_probs = trained_model(v_x.to(device)).cpu().numpy()
                
        opt_thresh, best_val_f2, _ = find_optimal_threshold_f2(y_val, val_probs)
        
        # Test Evaluation & Latency
        t_infer = time.time()
        mean_probs, var_uncert = evaluate_dl_uncertainty_mc_dropout(trained_model, test_x, num_passes=5, device=device)
        infer_latency_ms = ((time.time() - t_infer) / (len(X_test) * 5)) * 1000.0
        
        test_preds = (mean_probs >= opt_thresh).astype(int)
        metrics = compute_all_metrics(y_test, test_preds, mean_probs, threshold=opt_thresh)
        metrics["training_time_sec"] = round(train_time, 3)
        metrics["inference_latency_ms"] = round(infer_latency_ms, 3)
        metrics["mean_uncertainty"] = round(float(np.mean(var_uncert)), 6)
        metrics["best_val_f2"] = round(best_val_f2, 4)
        
        # Adversarial Robustness Check
        adv_drop = evaluate_adversarial_fgsm(trained_model, test_x, torch.tensor(y_test), epsilon=0.03, device=device)
        metrics["adversarial_drop_fgsm"] = round(adv_drop, 4)
        
        test_probabilities[name] = mean_probs
        test_predictions[name] = test_preds
        uncertainties[name] = var_uncert
        robustness_scores[name] = adv_drop
        
        save_model_artifacts(
            model=trained_model,
            name=f"dl_{name.lower()}",
            metrics=metrics,
            metadata={"history": history, "optimal_threshold": opt_thresh}
        )
        
        dl_results[name] = {
            "model": trained_model,
            "metrics": metrics,
            "optimal_threshold": opt_thresh,
            "history": history
        }
        logger.info(
            f"DL Result for {name}: Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, F2={metrics['f2_score']:.4f}, AUC-ROC={metrics['roc_auc']:.4f}"
        )
        
    return {
        "results": dl_results,
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities,
        "uncertainties": uncertainties,
        "robustness_scores": robustness_scores
    }
