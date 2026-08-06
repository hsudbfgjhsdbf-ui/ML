"""Single entry point for Approach 1 synthetic data, EDA, benchmarking, and documents."""
from __future__ import annotations
import argparse, shutil, time
from pathlib import Path
import joblib, matplotlib.pyplot as plt, pandas as pd, seaborn as sns, yaml
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_curve, roc_curve
from src.data.synthetic import generate_claims
from src.data.validation import validate
from src.evaluation.metrics import choose_threshold, measure
from src.features.build import feature_frame, transformer
from src.models.zoo import registry
from src.reporting.documents import build
from src.utils.common import ROOT, logger, seed_everything, stamp, write_json

def split_indices(df, seed):
    """Create claimant-disjoint train/validation/test indexes. Args: claims/seed. Returns: index tuples."""
    g=GroupShuffleSplit(n_splits=1,test_size=.30,random_state=seed); tr,hold=next(g.split(df,groups=df.claimant_id)); g2=GroupShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1); va_rel,te_rel=next(g2.split(df.iloc[hold],groups=df.iloc[hold].claimant_id)); return tr,hold[va_rel],hold[te_rel]
def main():
    """Execute the configured reproducible pipeline. Args: CLI. Returns: None."""
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config/default.yaml'); ap.add_argument('--dry-run',action='store_true'); args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text());
    if args.dry_run: print('S0-S14 plan: generate, validate, EDA, split, preprocess, benchmark, evaluate, publish documents'); return
    seed_everything(cfg['seed']); run_id=stamp(); run=ROOT/'evaluation'/'runs'/run_id; run.mkdir(parents=True); log=logger(run/'pipeline.log'); started=time.time(); log.info('S0 started | run=%s',run_id)
    df=generate_claims(cfg['data']['rows'],cfg['data']['fraud_rate'],cfg['seed']); raw=ROOT/'data/raw/synthetic_claims.csv'; raw.parent.mkdir(parents=True,exist_ok=True); df.to_csv(raw,index=False); report=validate(df); write_json(run/'data_quality_report.json',report)
    if report['status']!='PASS': raise RuntimeError(f'Data gate failed: {report}')
    log.info('S1-S2 complete | rows=%d fraud_rate=%.3f',len(df),df.is_fraud.mean())
    sns.set_theme(style='whitegrid'); (ROOT/'images/eda').mkdir(parents=True,exist_ok=True)
    for col,kind in [('total_claimed_amount_inr','hist'),('age_at_claim','hist'),('diagnosis_group','count'),('plan_type','count')]:
        plt.figure(figsize=(8,4)); (sns.histplot(df[col],bins=40) if kind=='hist' else sns.countplot(data=df,y=col,order=df[col].value_counts().index)); plt.title(f'Synthetic claims: {col}'); plt.tight_layout(); plt.savefig(ROOT/f'images/eda/{col}.png',dpi=cfg['reporting']['dpi']); plt.close()
    x=feature_frame(df); y=df.is_fraud; tr,va,te=split_indices(df,cfg['seed']); pre=transformer(x.iloc[tr]); Xtr=pre.fit_transform(x.iloc[tr]); Xva=pre.transform(x.iloc[va]); Xte=pre.transform(x.iloc[te]);
    rows=[]; probabilities={}; models=registry(cfg['seed'])
    for key in cfg['models']['enabled']:
        model=models[key]; t=time.time(); dense = key in {'gaussian_nb','hist_gradient_boosting'}; fit_x=Xtr.toarray() if dense and hasattr(Xtr,'toarray') else Xtr; val_x=Xva.toarray() if dense and hasattr(Xva,'toarray') else Xva; model.fit(fit_x,y.iloc[tr]); pv=model.predict_proba(val_x)[:,1]; threshold=choose_threshold(y.iloc[va].to_numpy(),pv); m=measure(y.iloc[va].to_numpy(),pv,threshold); m.update({'model_key':key,'family':type(model).__name__,'val_pr_auc':m.pop('pr_auc'),'val_roc_auc':m.pop('roc_auc'),'val_f2':m.pop('f2'),'train_seconds':time.time()-t,'status':'complete'}); rows.append(m); probabilities[key]=(model,pv,threshold); write_json(run/f'{key}_metrics.json',m); log.info('S8 %s PR-AUC %.4f',key,m['val_pr_auc'])
    board=pd.DataFrame(rows).sort_values(['val_pr_auc','val_f2'],ascending=False).reset_index(drop=True); board.insert(0,'rank',board.index+1); board.to_csv(run/'leaderboard.csv',index=False); winner=board.iloc[0].model_key; model,_,threshold=probabilities[winner]; final_dense=winner in {'gaussian_nb','hist_gradient_boosting'}; test_x=Xte.toarray() if final_dense and hasattr(Xte,'toarray') else Xte; pt=model.predict_proba(test_x)[:,1]; test=measure(y.iloc[te].to_numpy(),pt,threshold); write_json(run/'test_results.json',test); joblib.dump({'preprocessor':pre,'model':model,'features':list(x.columns)},run/'best_model.joblib')
    pd.DataFrame({'split':['train','validation','test'],'rows':[len(tr),len(va),len(te)],'fraud_rate':[y.iloc[tr].mean(),y.iloc[va].mean(),y.iloc[te].mean()]}).to_csv(run/'split_summary.csv',index=False)
    if (ROOT/'evaluation/latest').exists(): shutil.rmtree(ROOT/'evaluation/latest')
    shutil.copytree(run,ROOT/'evaluation/latest'); shutil.copy2(run/'leaderboard.csv',ROOT/'evaluation/leaderboard.csv'); shutil.copy2(run/'test_results.json',ROOT/'evaluation/test_results.json')
    build(ROOT,board,test,run_id); manifest={'run_id':run_id,'config':cfg,'winner':winner,'duration_seconds':time.time()-started,'stages':'S0-S14','data_quality':report}; write_json(run/'run_manifest.json',manifest)
    (ROOT/'evaluation/evaluation.md').write_text(f'# Evaluation results\n\nRun: `{run_id}`\n\n## Winner\n\n`{winner}` selected by validation PR-AUC.\n\n## Leaderboard\n\n'+board.to_markdown(index=False)+'\n\n## Test metrics\n\n```json\n'+str(test)+'\n```\n',encoding='utf-8')
    log.info('SUCCESS | winner=%s test PR-AUC=%.4f | documents generated',winner,test['pr_auc'])
if __name__=='__main__': main()
