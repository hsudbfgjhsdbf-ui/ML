# Data Quality Report

- Rows 4500 cols 19
- Missing {'ClaimID': 0, 'PatientID': 0, 'ProviderID': 0, 'ClaimAmount': 0, 'ClaimDate': 0, 'DiagnosisCode': 0, 'ProcedureCode': 0, 'PatientAge': 0, 'PatientGender': 0, 'ProviderSpecialty': 0, 'ClaimStatus': 0, 'PatientIncome': 0, 'PatientMaritalStatus': 0, 'PatientEmploymentStatus': 0, 'ProviderLocation': 0, 'ClaimType': 0, 'ClaimSubmissionMethod': 0, 'Cluster': 0, 'ClaimLegitimacy': 0}
- Outliers {
  "ClaimAmount": {
    "q1": 2509.0725,
    "q3": 7462.452499999999,
    "iqr": 4953.379999999999,
    "outlier_count": 0,
    "outlier_pct": 0.0
  },
  "PatientAge": {
    "q1": 25.0,
    "q3": 75.0,
    "iqr": 50.0,
    "outlier_count": 0,
    "outlier_pct": 0.0
  },
  "PatientIncome": {
    "q1": 52791.905,
    "q3": 115768.41750000001,
    "iqr": 62976.51250000001,
    "outlier_count": 0,
    "outlier_pct": 0.0
  }
}
- Class imbalance {'counts': {0: 4230, 1: 270}, 'ratio_minority_majority': 0.06382978723404255, 'fraud_rate': 0.06, 'is_imbalanced': np.True_}
- Leakage heuristics ['ClaimStatus', 'PatientMaritalStatus', 'PatientEmploymentStatus']
