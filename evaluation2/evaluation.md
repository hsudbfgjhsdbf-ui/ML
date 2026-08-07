# Evaluation hub — Approach 2 deep learning with XAI

**Run:** `run_20260807_155540`  
**Input split fingerprint:** `3bc2230ff033d29d4eb664ca8b78f2c902416a154bcc0cee88d0863881d59c9f`  
**Feature fingerprint:** `a9b8197b5635f847bff30b3f0ec5efa0f1485c5299612c20db5ce1de22ddbfda`  
**Device:** `cpu`  
**Positive class:** Fraud = 1.  

## 1. Comparability contract

Approach 2 reuses Approach 1's supplied workbook, feature engineering, train-only transformer, and persisted split membership. Deep models differ in representation and optimization, not in the rows or metric definitions. The test set is evaluated only after validation selection.

## 2. Deep leaderboard

 rank              key                     display_name      family   status   val_f2  val_pr_auc  val_pr_auc_std  val_roc_auc  test_accuracy  test_precision  test_recall  test_f1  test_f2  test_roc_auc  test_pr_auc  threshold  training_seconds  parameter_count  faithfulness  stability              run_id
    1 dl_e_transformer        Feature-token transformer transformer complete 0.970889    0.980010        0.005768     0.997538       0.992593        0.888889          1.0 0.941176 0.975610      0.998878     0.978764       0.98        176.908177            19873      0.077769   0.250000 run_20260807_155540
    2      dl_b_tabnet   TabNet-style attentive network   attention complete 0.970874    0.978157        0.003786     0.996615       0.989630        0.851064          1.0 0.919540 0.966184      0.998150     0.970672       0.78          7.997706            39697      0.081661   0.666667 run_20260807_155540
    3       dl_c_cnn1d 1D convolutional tabular network convolution complete 0.956952    0.976324        0.003267     0.997692       0.983704        0.784314          1.0 0.879121 0.947867      0.999252     0.988174       0.48         59.123713            27265      0.079590   0.818182 run_20260807_155540
    4         dl_a_mlp                              MLP       dense complete 0.966199    0.974841        0.004656     0.992460       0.989630        0.851064          1.0 0.919540 0.966184      0.999134     0.985916       0.85          5.281238            15105      0.005873   0.538462 run_20260807_155540
    5 dl_d_autoencoder       Autoencoder anomaly hybrid     anomaly complete 0.936150    0.952797        0.021040     0.991537       0.059259        0.059259          1.0 0.111888 0.239521      0.999016     0.984399       0.61          6.084294             8643      0.022823   0.538462 run_20260807_155540

**Selected deep model:** Feature-token transformer (`dl_e_transformer`), chosen by mean validation PR-AUC, then mean validation F2, then lower training time.

## 3. Calibration

Calibration method: isotonic regression on validation probabilities; validation Brier before/after: 0.0097/0.0026; validation ECE proxy before/after: 0.0193/0.0053.

## 4. XAI

All five architectures receive comparable occlusion importance, deletion-faithfulness, and jitter-stability artifacts. Native masks and attention tokens are auxiliary evidence and do not receive a scoring bonus. See `evaluation2/xai/`.

## 5. Fairness

The selected model is audited across gender, age band, claim type, and employment. Small slices are marked unstable; the synthetic-looking supplied workbook cannot support a claim of population fairness.

## 6. Limitations

The five deep models operate on the same transformed numeric/one-hot matrix. This is a transparent tabular bridge, not a document model or a full categorical entity-embedding system. Full Bayesian/Optuna search and later temporal/graph architectures are follow-on work.

## 7. Artifacts

Run manifest: `evaluation2/runs/run_20260807_155540/run_manifest.json`
Training telemetry: `evaluation2/metrics/` and `images2/telemetry/`
XAI: `evaluation2/xai/` and `images2/xai/`
Presentation: `presentation2/approach_2_deep_learning_xai.pptx`
Reports: `reports2/approach_2_project_report.pdf`, `reports2/approach_2_ieee_paper.pdf`
