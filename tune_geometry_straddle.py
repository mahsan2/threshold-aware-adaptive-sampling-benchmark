#!/usr/bin/env python3
"""Development/test tuning of a geometry-informed two-sided straddle rule.

The geometry weight is selected *only* on development seeds 0-4, then frozen and
assessed on held-out seeds 5-14. The primary development criterion is mean missed
exceedance; mean decision agreement breaks ties.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse, json
import numpy as np
import pandas as pd
import benchmark_extended as b

OUT=Path(__file__).resolve().parent
RESULTS=OUT/'results'; RESULTS.mkdir(exist_ok=True)
WEIGHTS=[0.0,0.10,0.20,0.30,0.40,0.50]
DEV_SEEDS=list(range(5))
TEST_SEEDS=list(range(5,15))


def choose_geometry_straddle(sel,mu,std,complexity,w,batch=b.BATCH):
    remaining=np.ones(len(b.XY),dtype=bool); remaining[sel]=False
    raw=b.norm01(1.96*std-np.abs(np.abs(mu)-b.TAU))
    score=(1-w)*raw+w*complexity
    score[~remaining]=-np.inf
    picked=[]; temp=score.copy()
    for _ in range(batch):
        j=int(np.argmax(temp)); picked.append(j)
        if len(picked)==batch: break
        d=np.sqrt(((b.XY-b.XY[j])**2).sum(axis=1))
        temp *= (0.35+0.65*b.norm01(d))
        temp[~remaining]=-np.inf; temp[picked]=-np.inf
    return np.array(picked,dtype=int)


def run_one(args):
    seed,family,noise,w=args
    true,comp,_,obs=b.scenario_data(seed,family,noise)
    sel=b.init_indices(seed)
    while len(sel)<b.MAX_BUDGET:
        mu,std=b.fit_gp(sel,obs,noise)
        add=choose_geometry_straddle(sel,mu,std,comp,w,min(b.BATCH,b.MAX_BUDGET-len(sel)))
        sel=np.concatenate([sel,add])
    mu,std=b.fit_gp(sel,obs,noise)
    rmse,mae,agr,missed,_,_=b.metrics(mu,std,true)
    c=b.confusion_metrics(mu,true)
    return dict(seed=seed,family=family,noise=noise,weight=w,rmse=rmse,mae=mae,
                decision_agreement=agr,missed_exceedance=missed,**c)


def eval_weights(seeds,weights,workers):
    tasks=[(s,f,n,w) for w in weights for s in seeds for f in b.FAMILIES for n in b.NOISE_LEVELS]
    if workers==1:
        rows=[run_one(t) for t in tasks]
    else:
        with Pool(workers) as pool: rows=pool.map(run_one,tasks)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=max(1,min(8,cpu_count()//2 or 1)))
    args=ap.parse_args()
    dev=eval_weights(DEV_SEEDS,WEIGHTS,args.workers)
    dev.to_csv(RESULTS/'geometry_straddle_dev.csv',index=False)
    dev_summary=dev.groupby('weight',as_index=False).agg(
        n=('decision_agreement','size'), missed=('missed_exceedance','mean'),
        agreement=('decision_agreement','mean'), balanced_accuracy=('balanced_accuracy','mean'), rmse=('rmse','mean'))
    dev_summary=dev_summary.sort_values(['missed','agreement'],ascending=[True,False])
    dev_summary.to_csv(RESULTS/'geometry_straddle_dev_summary.csv',index=False)
    best=float(dev_summary.iloc[0].weight)
    test=eval_weights(TEST_SEEDS,[best],args.workers)
    test['method']='geometry_straddle_frozen'
    test.to_csv(RESULTS/'geometry_straddle_test.csv',index=False)
    test_summary=test.agg({'missed_exceedance':'mean','decision_agreement':'mean','balanced_accuracy':'mean','rmse':'mean','recall':'mean','precision':'mean','false_positive_rate':'mean'}).to_dict()
    (RESULTS/'geometry_straddle_selection.json').write_text(json.dumps({'candidate_weights':WEIGHTS,'dev_seeds':DEV_SEEDS,'test_seeds':TEST_SEEDS,'selected_weight':best,'test_summary':test_summary},indent=2))
    print('DEV SUMMARY')
    print(dev_summary.to_string(index=False))
    print('\nSELECTED',best)
    print('\nHELD-OUT TEST')
    print(pd.Series(test_summary).to_string())

if __name__=='__main__': main()
