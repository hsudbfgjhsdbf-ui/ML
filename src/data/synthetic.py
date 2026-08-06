"""Deterministic synthetic Indian medical-insurance claim generator."""
from __future__ import annotations
import numpy as np
import pandas as pd

def generate_claims(rows: int, fraud_rate: float, seed: int) -> pd.DataFrame:
    """Generate plausible, non-identifying claim records. Args: rows/rate/seed. Returns: claim DataFrame."""
    rng=np.random.default_rng(seed); n_people=max(1,rows//5)
    people=rng.integers(0,n_people,rows); age=np.clip(rng.gamma(5,8,rows)+18,18,85).astype(int)
    tier=rng.choice(['tier_1','tier_2','tier_3'],rows,p=[.30,.35,.35])
    plan=rng.choice(['individual','family_floater','senior_citizen'],rows,p=[.48,.42,.10])
    diagnosis=rng.choice(['cardiac','orthopedic','maternity','respiratory','gastro','renal','oncology','infectious'],rows,p=[.12,.18,.10,.15,.12,.08,.07,.18])
    hosp=rng.integers(1,1200,rows); los=np.clip(rng.poisson(3,rows)+1,1,30)
    si=rng.choice([300000,500000,1000000,2000000],rows,p=[.25,.35,.28,.12])
    base={'cardiac':220000,'orthopedic':145000,'maternity':65000,'respiratory':55000,'gastro':85000,'renal':190000,'oncology':380000,'infectious':45000}
    amount=np.array([base[x] for x in diagnosis])*rng.lognormal(0,.45,rows)
    amount=np.clip(amount,5000,900000).round(0)
    submit=np.clip(rng.gamma(2,12,rows),0,180).astype(int); early=rng.random(rows)<.10
    duplicate=rng.random(rows)<.018; docs=np.clip(rng.beta(8,2,rows),0,1); provider_risk=rng.random(1200)<.03
    provider_signal=provider_risk[hosp-1]; ratio=amount/si
    score=-3.2+2.0*(ratio>.55)+1.5*(submit>75)+2.1*duplicate+1.0*early+1.4*provider_signal+1.2*(docs<.55)+rng.normal(0,.8,rows)
    # Set prevalence deterministically by quantile, avoiding direct target leakage features.
    cutoff=np.quantile(score,1-fraud_rate); fraud=(score>=cutoff).astype(int)
    fraud_type=np.where(fraud==0,'legitimate',rng.choice(['fabricated_bill','upcoding','duplicate_claim','phantom_admission','provider_collusion','unbundling'],rows))
    dates=pd.Timestamp('2024-01-01')+pd.to_timedelta(rng.integers(0,730,rows),unit='D')
    return pd.DataFrame({'claim_id':[f'CLM{i:010d}' for i in range(rows)],'claimant_id':[f'CLT{x:07d}' for x in people], 'age_at_claim':age,'gender':rng.choice(['female','male','other'],rows,p=[.48,.51,.01]),'city_tier':tier,'plan_type':plan,'sum_insured_inr':si,'annual_premium_inr':(si*rng.uniform(.008,.03,rows)).round(0),'claim_type':rng.choice(['cashless','reimbursement'],rows,p=[.55,.45]),'admission_type':rng.choice(['emergency','planned','transferred'],rows,p=[.48,.47,.05]),'diagnosis_group':diagnosis,'hospital_id':[f'HSP{x:04d}' for x in hosp],'hospital_tier':rng.choice(['tier_1','tier_2','tier_3'],rows),'length_of_stay_days':los,'total_claimed_amount_inr':amount,'room_charges_inr':(amount*rng.uniform(.08,.2,rows)).round(0),'pharmacy_share':rng.uniform(.1,.35,rows),'diagnostics_share':rng.uniform(.05,.25,rows),'submission_delay_days':submit,'intimation_delay_days':np.maximum(0,submit-rng.integers(0,15,rows)),'documents_complete_first_pass':docs>.7,'document_set_completeness':docs.round(3),'duplicate_invoice_flag':duplicate,'early_claim_flag':early,'claims_last_12m_count':rng.poisson(1.4,rows),'provider_claim_volume_30d':rng.poisson(35,rows),'claim_date':dates.strftime('%d-%m-%Y'),'fraud_type':fraud_type,'is_fraud':fraud})
