"""
03_anomaly_detection.py — Unsupervised and semi-supervised anomaly detection for suspicious claims.

Includes:
- Isolation Forest, LOF, One-Class SVM, Robust Covariance
- Optional autoencoder
- Training only on non-fraud where appropriate
- Distinction between anomaly score vs fraud probability
- Precision@k, Recall@k, ranking evaluation, threshold analysis, visualizations
- Limitations documented

"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config
from common.logging_utils import get_logger
from common.seed import set_global_seed
from common.dataset_loader import load_claims_dataset, get_feature_types
from common.preprocessing import build_preprocessor, engineer_date_features
from common.artifacts import save_json, save_model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

logger = get_logger("03_anomaly_detection")

def load_data(config, override=None):
    raw_path = Path(override) if override else PROJECT_ROOT / config.get("dataset",{}).get("raw_path","data/raw/Health_Insurance_Fraud_Claims.xlsx")
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    if not raw_path.exists():
        alt = PROJECT_ROOT.parent / "Health Insurance Fraud Claims.xlsx"
        if alt.exists():
            raw_path = alt
    df = load_claims_dataset(raw_path)
    return df

def prepare_features(df, config):
    target_col = config.get("dataset",{}).get("target_column","ClaimLegitimacy")
    num_feats, cat_feats, date_feats, drop_feats = get_feature_types(df, config)
    df_engineered = engineer_date_features(df.drop(columns=[target_col]), date_feats)
    y = df[target_col].map({"Legitimate":0,"Fraud":1}).astype(int)
    if y.isna().any():
        y = df[target_col].astype(str).str.lower().map({"legitimate":0,"fraud":1})
    X = df_engineered
    engineered_date_cols = [c for c in X.columns if "ClaimDate" in c]
    full_num = list(set(num_feats + engineered_date_cols + ["Cluster"]))
    full_num = [c for c in full_num if c in X.columns]
    full_cat = [c for c in cat_feats if c in X.columns]
    return X, y, full_num, full_cat

def compute_precision_recall_at_k(y_true, scores, k):
    """scores higher means more anomalous"""
    order = np.argsort(scores)[::-1]
    topk = order[:k]
    y_arr = np.array(y_true)
    prec = y_arr[topk].mean() if len(topk)>0 else 0.0
    total_pos = y_arr.sum()
    rec = y_arr[topk].sum()/total_pos if total_pos>0 else 0.0
    return float(prec), float(rec)

def evaluate_anomaly(y_true, anomaly_scores, contamination=None):
    """Anomaly scores: higher = more anomalous.
    Convert to pseudo probabilities via ranking? But distinguish.
    """
    # For metric, we treat anomaly_score as decision score (not calibrated probability)
    # Compute PR-AUC and ROC-AUC using score
    try:
        pr_auc = average_precision_score(y_true, anomaly_scores)
    except:
        pr_auc = float('nan')
    try:
        roc_auc = roc_auc_score(y_true, anomaly_scores)
    except:
        roc_auc = float('nan')

    # Precision at top k
    ks = [10, 50, 100, 200]
    precisions = {}
    recalls = {}
    for k in ks:
        if k <= len(y_true):
            p,r = compute_precision_recall_at_k(y_true, anomaly_scores, k)
            precisions[f"prec@{k}"] = p
            recalls[f"rec@{k}"] = r

    return {"pr_auc": pr_auc, "roc_auc": roc_auc, **precisions, **recalls}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else PROJECT_ROOT/"config.yaml")
    set_global_seed(config.get("dataset",{}).get("random_state",42))

    df = load_data(config, args.data_path)
    X, y, full_num, full_cat = prepare_features(df, config)

    rs = config.get("dataset",{}).get("random_state",42)
    test_size = config.get("dataset",{}).get("test_size",0.2)
    # Split normally but for anomaly we train only on legit
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)
    # Further split train into train legit only
    # For semi-supervised: train_only_on_legit = True
    train_only_legit = config.get("anomaly_detection",{}).get("train_only_on_legit", True)
    if train_only_legit:
        mask_legit = (y_train_full==0)
        X_train = X_train_full[mask_legit]
        y_train_legit = y_train_full[mask_legit]
        logger.info(f"Training on legit only: {len(X_train)} out of {len(X_train_full)}")
    else:
        X_train = X_train_full

    preprocessor = build_preprocessor(full_num, full_cat, [], config)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    X_train_full_trans = preprocessor.transform(X_train_full)

    contamination = config.get("anomaly_detection",{}).get("contamination",0.06)

    models = {}

    # Isolation Forest
    try:
        iso = IsolationForest(contamination=contamination, random_state=rs, n_jobs=-1)
        iso.fit(X_train_trans)
        # Decision function: lower = more anomalous, we invert to get anomaly_score higher = more anomalous
        train_scores = -iso.decision_function(X_train_trans)
        test_scores = -iso.decision_function(X_test_trans)
        models["IsolationForest"] = (iso, test_scores, X_train_trans, X_test_trans)
        logger.info("IsolationForest trained")
    except Exception as e:
        logger.warning(f"IsolationForest failed: {e}")

    # LOF - note LOF is transductive, need to use fit_predict? We'll use novelty mode
    try:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True, n_jobs=-1)
        lof.fit(X_train_trans)
        test_scores = -lof.decision_function(X_test_trans)  # more positive = outlier?
        # For LOF, negative_outlier_factor_ is opposite: lower = more abnormal
        # decision_function already gives shift; invert for consistency
        models["LOF"] = (lof, test_scores, X_train_trans, X_test_trans)
        logger.info("LOF trained")
    except Exception as e:
        logger.warning(f"LOF failed: {e}")

    # One-Class SVM
    try:
        oc_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
        oc_svm.fit(X_train_trans)
        test_scores = -oc_svm.decision_function(X_test_trans)
        models["OneClassSVM"] = (oc_svm, test_scores, X_train_trans, X_test_trans)
        logger.info("OneClassSVM trained")
    except Exception as e:
        logger.warning(f"OneClassSVM failed: {e}")

    # EllipticEnvelope (robust covariance)
    try:
        ee = EllipticEnvelope(contamination=contamination, random_state=rs)
        ee.fit(X_train_trans)
        test_scores = -ee.decision_function(X_test_trans)
        models["EllipticEnvelope"] = (ee, test_scores, X_train_trans, X_test_trans)
        logger.info("EllipticEnvelope trained")
    except Exception as e:
        logger.warning(f"EllipticEnvelope failed: {e}")

    # Autoencoder optional
    autoencoder_model = None
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        # Simple autoencoder
        input_dim = X_train_trans.shape[1]
        class AE(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, 64),
                    nn.ReLU(),
                    nn.Linear(64, dim)
                )
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)
        ae = AE(input_dim)
        # Train only on legit
        X_t = torch.tensor(X_train_trans, dtype=torch.float32)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        ae.train()
        for epoch in range(30):
            losses=[]
            for (xb,) in loader:
                opt.zero_grad()
                recon = ae(xb)
                loss = crit(recon, xb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if epoch%10==0:
                logger.info(f"AE epoch {epoch} loss {np.mean(losses):.4f}")
        ae.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_trans, dtype=torch.float32)
            recon = ae(X_test_t)
            mse = ((recon - X_test_t)**2).mean(dim=1).numpy()
            # mse higher = more anomalous
            models["Autoencoder"] = (ae, mse, X_train_trans, X_test_trans)
            logger.info("Autoencoder trained")
    except Exception as e:
        logger.info(f"Autoencoder not trained (torch missing or error): {e}")

    # Evaluate each
    eval_dir = PROJECT_ROOT / config.get("paths",{}).get("evaluation_dir","evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_scores = {}

    for name, (model, test_scores, _, _) in models.items():
        metrics = evaluate_anomaly(y_test, test_scores, contamination)
        logger.info(f"{name} metrics {metrics}")
        all_scores[name] = test_scores.tolist() if hasattr(test_scores, "tolist") else list(test_scores)
        results.append({"model": name, **metrics})

        # Threshold analysis: find threshold that maximizes F? But unsupervised
        # We'll compute confusion at contamination threshold
        # Sort scores, top contamination% as anomalies
        k = int(len(test_scores)*contamination)
        thr = np.sort(test_scores)[::-1][k] if k < len(test_scores) else 0.5
        y_pred = (test_scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        logger.info(f"{name} @ thr {thr:.4f} TP {tp} FP {fp} FN {fn} TN {tn}")

    # Ensemble: average normalized scores
    try:
        if len(models)>=2:
            # Normalize each score to 0-1 via rank or minmax
            normalized = []
            for name in models:
                s = np.array(all_scores[name])
                # min-max
                if s.max()!=s.min():
                    ns = (s - s.min())/(s.max()-s.min())
                else:
                    ns = s
                normalized.append(ns)
            ensemble_score = np.mean(normalized, axis=0)
            metrics = evaluate_anomaly(y_test, ensemble_score, contamination)
            results.append({"model": "EnsembleAnomalyAvg", **metrics})
            models["EnsembleAnomalyAvg"] = (None, ensemble_score, None, None)
            logger.info(f"Ensemble metrics {metrics}")
    except Exception as e:
        logger.warning(f"Ensemble failed {e}")

    # Save results
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results.to_csv(eval_dir/"anomaly_detection_results.csv", index=False)

    save_json({
        "approach": "03_anomaly_detection",
        "contamination": contamination,
        "train_only_legit": train_only_legit,
        "results": results,
        "note_anomaly_vs_fraud": "Anomaly score measures deviation from normal pattern, NOT verified fraud probability. Fraud probability requires supervised calibration. Anomaly is unsupervised risk indicator.",
        "limitations": [
            "Unsupervised methods have high false positives",
            "Cannot distinguish fraud vs rare but legitimate cases without labels",
            "Requires calibrated threshold via labeled data for operational use",
            "If trained only on legit, assumes clean training data",
            "High-cardinality categorical may degrade distance-based methods"
        ]
    }, eval_dir/"anomaly_detection_metrics.json")

    # Save data for visualizations: anomaly scores
    try:
        import json
        # Save per model scores for later plots
        with open(eval_dir/"anomaly_scores.json","w") as f:
            json.dump({"y_test": y_test.tolist(), "scores": all_scores}, f)
    except Exception as e:
        logger.warning(f"Saving anomaly scores failed {e}")

    # Save artifacts (models)
    artifacts_dir = PROJECT_ROOT / config.get("paths",{}).get("artifacts_dir","data/processed/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, (model,_,_,_) in models.items():
        if model is not None and name!="EnsembleAnomalyAvg":
            try:
                if name=="Autoencoder":
                    import torch
                    torch.save(model.state_dict(), artifacts_dir/f"anomaly_{name}.pt")
                else:
                    save_model(model, artifacts_dir/f"anomaly_{name}.joblib")
            except Exception as e:
                logger.warning(f"Saving {name} failed {e}")
    # Save preprocessor
    save_model(preprocessor, artifacts_dir/"anomaly_preprocessor.joblib")

    # Documentation for anomaly approach
    doc_dir = PROJECT_ROOT / config.get("paths",{}).get("documentation_dir","documentation")
    doc_dir.mkdir(parents=True, exist_ok=True)
    with open(doc_dir/"anomaly_detection.md","w") as f:
        f.write("# Anomaly Detection Approach\n\n")
        f.write("## Methods\nIsolation Forest, LOF, One-Class SVM, EllipticEnvelope, Autoencoder optional, Ensemble avg\n\n")
        f.write("## Training\nOnly non-fraud records used where configured. Contamination 0.06 based on fraud rate.\n\n")
        f.write("## Distinction\n- Anomaly score: deviation measure\n- Fraud probability: calibrated supervised probability\n- Fraud label: ground truth\n\n")
        f.write("Anomaly score must NOT be presented as verified fraud probability unless calibrated.\n\n")
        f.write("## Evaluation\nPrecision@k, Recall@k, PR-AUC, ROC-AUC using fraud labels as reference.\n\n")
        f.write("## Results summary\n")
        f.write(df_results.to_string() if not df_results.empty else "No results")
        f.write("\n\n## Limitations\n- High FP, cannot replace supervised model\n- Needs human review\n")

    logger.info("Anomaly detection completed")
    print(f"ANOMALY_RESULTS {df_results.head().to_dict() if not df_results.empty else {}}")

if __name__ == "__main__":
    main()
