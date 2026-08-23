#!/usr/bin/env python3
"""Sensitivity of the main adaptive policies to fixed GP length scale."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import numpy as np
import pandas as pd
import benchmark_extended as b

OUT=Path(__file__).resolve().parent; R=OUT/'results'; R.mkdir(exist_ok=True)
LENGTHS=[0.10,0.16,0.24]
METHODS=['uncertainty_only','two_sided_straddle','evidence_gated']


def run_one(args):
    seed,family,noise,method,ls=args
    true,comp,_,obs=b.scenario_data(seed,family,noise); sel=b.init_indices(seed)
    while len(sel)<b.MAX_BUDGET:
        mu,std=b.fit_gp(sel,obs,noise,length_scale=ls)
        add=b.choose_batch(method,sel,mu,std,comp,min(b.BATCH,b.MAX_BUDGET-len(sel)))
        sel=np.concatenate([sel,add])
    mu,std=b.fit_gp(sel,obs,noise,length_scale=ls)
    rmse,mae,agr,missed,_,_=b.metrics(mu,std,true); c=b.confusion_metrics(mu,true)
    return dict(seed=seed,family=family,noise=noise,method=method,length_scale=ls,rmse=rmse,mae=mae,decision_agreement=agr,missed_exceedance=missed,**c)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=max(1,min(8,cpu_count()//2 or 1))); args=ap.parse_args()
    tasks=[(s,f,n,m,ls) for ls in LENGTHS for s in b.SEEDS for f in b.FAMILIES for n in b.NOISE_LEVELS for m in METHODS]
    if args.workers==1: rows=[run_one(t) for t in tasks]
    else:
        with Pool(args.workers) as pool: rows=pool.map(run_one,tasks)
    df=pd.DataFrame(rows); df.to_csv(R/'kernel_sensitivity_results.csv',index=False)
    sm=df.groupby(['length_scale','method'],as_index=False).agg(n=('decision_agreement','size'),rmse=('rmse','mean'),agreement=('decision_agreement','mean'),missed=('missed_exceedance','mean'),balanced_accuracy=('balanced_accuracy','mean'))
    sm.to_csv(R/'kernel_sensitivity_summary.csv',index=False)
    print(sm.to_string(index=False))
if __name__=='__main__': main()
