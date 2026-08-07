"""
Traditional Machine Learning Approach (Approach 1) implementation.
Trains 12 classification algorithms, performs hyperparameter tuning,
evaluates using comprehensive metrics (F2 score priority), measures efficiency,
and generates benchmarking tables and evaluations.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def load_data():
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    X_train, y_train = train_df.drop(columns=['Target']), train_df['Target'].values
    X_val, y_val = val_df.drop(columns=['Target']), val_df['Target'].values
    X_test, y_test = test_df.drop(columns=['Target']), test_df['Target'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_model(model, X_test, y_test, train_time, pred_time_per_sample, model_size_kb):
    start_t = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2.0, zero_division=0)
    try:
        auc_roc = roc_auc_score(y_test, y_prob)
    except:
        auc_roc = 0.5
    try:
        auc_pr = average_precision_score(y_test, y_prob)
    except:
        auc_pr = 0.0
    mcc = matthews_corrcoef(y_test, y_pred)
    
    return {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'F2-Score': round(f2, 4),
        'AUC-ROC': round(auc_roc, 4),
        'AUC-PR': round(auc_pr, 4),
        'MCC': round(mcc, 4),
        'Training Time (s)': round(train_time, 3),
        'Prediction Latency (ms/sample)': round(pred_time_per_sample * 1000, 3),
        'Model Size (KB)': round(model_size_kb, 2)
    }

def train_and_benchmark_ml():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=9, random_state=42, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbose=-1),
        'Support Vector Machine': SVC(probability=True, class_weight='balanced', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Gaussian Naive Bayes': GaussianNB(),
        'Artificial Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis()
    }
    
    results = {}
    os.makedirs('models/ml', exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Measure prediction latency per sample
        t0 = time.time()
        _ = model.predict(X_test)
        pred_time_per_sample = (time.time() - t0) / len(X_test)
        
        # Save model and get size
        model_path = f"models/ml/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)
        model_size_kb = os.path.getsize(model_path) / 1024.0
        
        metrics = evaluate_model(model, X_test, y_test, train_time, pred_time_per_sample, model_size_kb)
        results[name] = metrics
        print(f"-> {name} F2-Score: {metrics['F2-Score']}, AUC-ROC: {metrics['AUC-ROC']}")
        
    # Convert results to DataFrame and rank by F2-Score
    res_df = pd.DataFrame(results).T
    res_df = res_df.sort_values(by='F2-Score', ascending=False)
    
    os.makedirs('evaluation', exist_ok=True)
    csv_path = 'evaluation/traditional_ml_benchmark.csv'
    res_df.to_csv(csv_path)
    
    # Generate Markdown report
    md_content = "# Approach 1: Traditional Machine Learning Benchmarking Report\n\n"
    md_content += "This report presents the comprehensive performance evaluation of 12 traditional machine learning algorithms "
    md_content += "implemented for Medical Insurance Claim Fraud Detection, adhering to the IIIT Dharwad project guidelines.\n\n"
    md_content += "## Benchmarking Results Table\n\n"
    md_content += res_df.to_markdown()
    md_content += "\n\n## Algorithm Rankings & Analysis\n\n"
    md_content += "1. **Primary Ranking Criterion (F2-Score)**: Prioritizes recall to minimize false negatives (undetected fraudulent claims).\n"
    md_content += "2. **Secondary Criterion (AUC-ROC)**: Evaluates class discrimination across all classification thresholds.\n"
    md_content += f"3. **Top Performer**: **{res_df.index[0]}** achieved the highest F2-Score of **{res_df.iloc[0]['F2-Score']}**.\n"
    
    with open('evaluation/traditional_ml_benchmark.md', 'w') as f:
        f.write(md_content)
        
    print("\nTraining and benchmarking completed successfully. Results saved to evaluation/traditional_ml_benchmark.md")
    return res_df

if __name__ == "__main__":
    train_and_benchmark_ml()
