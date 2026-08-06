"""Schema checks that halt processing on material data violations."""
from __future__ import annotations
import pandas as pd

def validate(df: pd.DataFrame) -> dict:
    """Validate essential claims constraints. Args: DataFrame. Returns: gate report."""
    checks={'G2_required_columns':{'claim_id','claimant_id','is_fraud'}.issubset(df.columns),'G4_unique_claim_id':df.claim_id.is_unique,'G7_positive_amounts':bool((df.total_claimed_amount_inr>0).all()),'G11_binary_label':set(df.is_fraud.unique())<={0,1},'G12_prevalence':.03<=df.is_fraud.mean()<=.20,'G_privacy_no_names':not any(x in df.columns for x in ['name','phone','email','aadhaar'])}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','rows':len(df),'fraud_rate':float(df.is_fraud.mean()),'checks':checks}
