#!/usr/bin/env python3
"""Extended geometry-prior robustness ablation including direct threshold baselines."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import benchmark_extended as b

OUT=Path(__file__).resolve().parent; R=OUT/'results'; R.mkdir(exist_ok=True)
PLACEMENTS=['geometry_independent','low_complexity']
FAMILIES=['local_defect','mixed']
SEEDS=list(range(8))
METHODS=['uncertainty_only','two_sided_straddle','posterior_risk','evidence_no_geometry','evidence_gated']


def make_surface_ablation(seed,family,placement):
    rng=np.random.default_rng(seed*1009+b.FAMILIES.index(family)*9176+13)
    cx,cy=rng.uniform(.25,.75,2); sx,sy=rng.uniform(.10,.20,2)
    h=(0.45*np.sin(np.pi*b.Xg)*np.sin(np.pi*b.Yg)
       +0.25*np.exp(-(((b.Xg-cx)/sx)**2+((b.Yg-cy)/sy)**2)/2)
       +0.12*(b.Xg-0.5)**2)
    gy,gx=np.gradient(h,b.y,b.x); gmag=np.sqrt(gx**2+gy**2)
    gyy,_=np.gradient(gy,b.y,b.x); _,gxx=np.gradient(gx,b.y,b.x)
    lap=np.abs(gxx+gyy); comp=b.norm01(gmag+0.12*lap)
    raw=gaussian_filter(rng.normal(size=(b.GRID_N,b.GRID_N)),sigma=2.2,mode='reflect')
    raw=(raw-raw.mean())/(raw.std()+1e-12); phi1,phi2=rng.uniform(0,2*np.pi,2)
    dev=0.010*raw+0.018*(comp-comp.mean()); flatc=comp.ravel()
    if placement=='geometry_independent': idx=rng.integers(len(flatc))
    elif placement=='low_complexity': idx=rng.choice(np.where(flatc<=np.quantile(flatc,.30))[0])
    else: raise ValueError(placement)
    dx,dy=b.XY[idx]
    if family=='local_defect':
        sig=rng.uniform(.035,.075); amp=rng.choice([-1,1])*rng.uniform(.075,.105)
        dev += amp*np.exp(-((b.Xg-dx)**2+(b.Yg-dy)**2)/(2*sig**2)); dev += 0.012*np.sin(2*np.pi*b.Xg+phi1)
    elif family=='mixed':
        sig=rng.uniform(.04,.08); amp=rng.choice([-1,1])*rng.uniform(.065,.095)
        dev += 0.025*np.sin(5*np.pi*b.Xg+phi1)*np.cos(3*np.pi*b.Yg+phi2)
        dev += amp*np.exp(-((b.Xg-dx)**2+(b.Yg-dy)**2)/(2*sig**2))
    else: raise ValueError(family)
    dev-=np.median(dev); return dev.ravel(),comp.ravel(),h.ravel()


def run_one(args):
    placement,family,noise,seed,method=args
    true,comp,_=make_surface_ablation(seed,family,placement)
    rng=np.random.default_rng(seed*31337+int(noise*1e6)*7+b.FAMILIES.index(family)*101)
    obs=true+rng.normal(0,noise,len(true)); sel=b.init_indices(seed)
    while len(sel)<b.MAX_BUDGET:
        mu,std=b.fit_gp(sel,obs,noise); add=b.choose_batch(method,sel,mu,std,comp,min(b.BATCH,b.MAX_BUDGET-len(sel))); sel=np.concatenate([sel,add])
    mu,std=b.fit_gp(sel,obs,noise); rmse,mae,agr,missed,_,_=b.metrics(mu,std,true); c=b.confusion_metrics(mu,true)
    return dict(placement=placement,family=family,noise=noise,seed=seed,method=method,rmse=rmse,mae=mae,decision_agreement=agr,missed_exceedance=missed,**c)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=max(1,min(8,cpu_count()//2 or 1))); args=ap.parse_args()
    tasks=[(p,f,n,s,m) for p in PLACEMENTS for f in FAMILIES for n in b.NOISE_LEVELS for s in SEEDS for m in METHODS]
    if args.workers==1: rows=[run_one(t) for t in tasks]
    else:
        with Pool(args.workers) as pool: rows=pool.map(run_one,tasks)
    df=pd.DataFrame(rows); df.to_csv(R/'robustness_ablation_extended_results.csv',index=False)
    sm=df.groupby(['placement','method'],as_index=False).agg(n=('decision_agreement','size'),rmse=('rmse','mean'),agreement=('decision_agreement','mean'),missed=('missed_exceedance','mean'),balanced_accuracy=('balanced_accuracy','mean'),recall=('recall','mean'),precision=('precision','mean'))
    sm.to_csv(R/'robustness_ablation_extended_summary.csv',index=False)
    fam=df.groupby(['placement','family','method'],as_index=False).agg(n=('decision_agreement','size'),rmse=('rmse','mean'),agreement=('decision_agreement','mean'),missed=('missed_exceedance','mean'),balanced_accuracy=('balanced_accuracy','mean'))
    fam.to_csv(R/'robustness_ablation_extended_by_family.csv',index=False)
    print(sm.to_string(index=False))
if __name__=='__main__': main()
