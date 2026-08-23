#!/usr/bin/env python3
"""Held-out tuning study for the U/G/D composite acquisition weights.

Candidate weights lie on a 0.2 simplex grid and are evaluated only on development
seeds 0-4. The selected weights are then frozen and evaluated on seeds 5-14.
This analysis is separate from the originally pre-specified 0.40/0.25/0.35 policy.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse, json
import numpy as np
import pandas as pd
import benchmark_extended as b

OUT=Path(__file__).resolve().parent; R=OUT/'results'; R.mkdir(exist_ok=True)
DEV=list(range(5)); TEST=list(range(5,15))
GRID=[]
for iu in range(6):
    for ig in range(6-iu):
        idd=5-iu-ig
        GRID.append((iu/5,ig/5,idd/5))


def choose(sel,mu,std,comp,w,batch=b.BATCH):
    wu,wg,wd=w
    rem=np.ones(len(b.XY),bool); rem[sel]=False
    u=b.norm01(std); prox=np.exp(-np.abs(np.abs(mu)-b.TAU)/(0.30*b.TAU)); dist=b.norm01(b.min_dist_to_selected(sel))
    score=wu*u+wg*comp+wd*prox
    score=0.95*score+0.05*dist
    score[~rem]=-np.inf; picked=[]; temp=score.copy()
    for _ in range(batch):
        j=int(np.argmax(temp)); picked.append(j)
        if len(picked)==batch: break
        d=np.sqrt(((b.XY-b.XY[j])**2).sum(axis=1)); temp *= (0.35+0.65*b.norm01(d))
        temp[~rem]=-np.inf; temp[picked]=-np.inf
    return np.array(picked,int)


def run_one(args):
    seed,family,noise,w=args
    true,comp,_,obs=b.scenario_data(seed,family,noise); sel=b.init_indices(seed)
    while len(sel)<b.MAX_BUDGET:
        mu,std=b.fit_gp(sel,obs,noise); add=choose(sel,mu,std,comp,w,min(b.BATCH,b.MAX_BUDGET-len(sel))); sel=np.concatenate([sel,add])
    mu,std=b.fit_gp(sel,obs,noise); rmse,mae,agr,missed,_,_=b.metrics(mu,std,true); c=b.confusion_metrics(mu,true)
    return dict(seed=seed,family=family,noise=noise,w_u=w[0],w_g=w[1],w_d=w[2],rmse=rmse,mae=mae,decision_agreement=agr,missed_exceedance=missed,**c)


def evaluate(seeds,weights,workers):
    tasks=[(s,f,n,w) for w in weights for s in seeds for f in b.FAMILIES for n in b.NOISE_LEVELS]
    if workers==1: rows=[run_one(t) for t in tasks]
    else:
        with Pool(workers) as pool: rows=pool.map(run_one,tasks)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=max(1,min(8,cpu_count()//2 or 1))); args=ap.parse_args()
    dev=evaluate(DEV,GRID,args.workers); dev.to_csv(R/'evidence_weight_grid_dev.csv',index=False)
    s=dev.groupby(['w_u','w_g','w_d'],as_index=False).agg(n=('decision_agreement','size'),missed=('missed_exceedance','mean'),agreement=('decision_agreement','mean'),balanced_accuracy=('balanced_accuracy','mean'),rmse=('rmse','mean'))
    s=s.sort_values(['missed','agreement'],ascending=[True,False]); s.to_csv(R/'evidence_weight_grid_dev_summary.csv',index=False)
    best=(float(s.iloc[0].w_u),float(s.iloc[0].w_g),float(s.iloc[0].w_d))
    test=evaluate(TEST,[best],args.workers); test['method']='evidence_tuned_frozen'; test.to_csv(R/'evidence_tuned_heldout.csv',index=False)
    ts=test.agg({'missed_exceedance':'mean','decision_agreement':'mean','balanced_accuracy':'mean','rmse':'mean','recall':'mean','precision':'mean','false_positive_rate':'mean'}).to_dict()
    (R/'evidence_weight_selection.json').write_text(json.dumps({'grid_step':0.2,'dev_seeds':DEV,'test_seeds':TEST,'selected_weights':{'uncertainty':best[0],'geometry':best[1],'decision_proximity':best[2]},'heldout_summary':ts},indent=2))
    print('TOP 10 DEV')
    print(s.head(10).to_string(index=False)); print('\nSELECTED',best); print('\nHELD-OUT'); print(pd.Series(ts).to_string())

if __name__=='__main__': main()
