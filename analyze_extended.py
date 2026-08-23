#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT=Path(__file__).resolve().parent
R=OUT/'results'
TARGET='evidence_gated'
BASELINES=['space_filling','geometry_only','uncertainty_only','two_sided_straddle','randomized_straddle','posterior_risk','evidence_no_geometry']


def holm_adjust(pvals):
    p=np.asarray(pvals,float); n=len(p); order=np.argsort(p); adj=np.empty(n,float); running=0.0
    for rank,idx in enumerate(order):
        val=(n-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj


def rank_biserial(d):
    d=np.asarray(d,float); d=d[d!=0]
    if len(d)==0: return 0.0
    ranks=pd.Series(np.abs(d)).rank(method='average').to_numpy()
    Wp=ranks[d>0].sum(); Wm=ranks[d<0].sum(); denom=Wp+Wm
    return float((Wp-Wm)/denom) if denom else 0.0


def cluster_bootstrap_ci(paired, metric, n_boot=10000, seed=20260822):
    # paired has columns seed, target, baseline; resample seed clusters with replacement.
    rng=np.random.default_rng(seed)
    seeds=np.array(sorted(paired.seed.unique()))
    by_seed={s:paired.loc[paired.seed==s, metric].to_numpy() for s in seeds}
    vals=[]
    for _ in range(n_boot):
        chosen=rng.choice(seeds,size=len(seeds),replace=True)
        pieces=[by_seed[s] for s in chosen]
        vals.append(np.concatenate(pieces).mean())
    return np.quantile(vals,[.025,.975])


def main():
    df=pd.read_csv(R/'extended_final_scenarios.csv')
    keys=['seed','family','noise']
    summary=df.groupby('method',as_index=False).agg(
        n=('decision_agreement','size'), rmse=('rmse','mean'), agreement=('decision_agreement','mean'),
        missed=('missed_exceedance','mean'), recall=('recall','mean'), precision=('precision','mean'),
        fpr=('false_positive_rate','mean'), balanced_accuracy=('balanced_accuracy','mean'),
        prevalence=('prevalence','mean'))
    summary.to_csv(R/'final_summary_all_methods.csv',index=False)

    fam=df.groupby(['family','method'],as_index=False).agg(
        n=('decision_agreement','size'),rmse=('rmse','mean'),agreement=('decision_agreement','mean'),
        missed=('missed_exceedance','mean'),recall=('recall','mean'),precision=('precision','mean'),
        fpr=('false_positive_rate','mean'),balanced_accuracy=('balanced_accuracy','mean'))
    fam.to_csv(R/'final_summary_by_family.csv',index=False)

    noise=df.groupby(['noise','method'],as_index=False).agg(
        n=('decision_agreement','size'),rmse=('rmse','mean'),agreement=('decision_agreement','mean'),
        missed=('missed_exceedance','mean'),recall=('recall','mean'),precision=('precision','mean'),
        fpr=('false_positive_rate','mean'),balanced_accuracy=('balanced_accuracy','mean'))
    noise.to_csv(R/'final_summary_by_noise.csv',index=False)

    metrics=['decision_agreement','missed_exceedance','rmse','balanced_accuracy']
    rows=[]
    t=df[df.method==TARGET].set_index(keys)
    for base in BASELINES:
        b=df[df.method==base].set_index(keys)
        joined=t.join(b,lsuffix='_target',rsuffix='_base',how='inner').reset_index()
        for metric in metrics:
            d=joined[f'{metric}_target']-joined[f'{metric}_base']
            try: p_scenario=float(wilcoxon(d,zero_method='wilcox',alternative='two-sided').pvalue)
            except ValueError: p_scenario=1.0
            seed_d=(joined.assign(diff=d).groupby('seed',as_index=False)['diff'].mean())
            try: p_seed=float(wilcoxon(seed_d['diff'],zero_method='wilcox',alternative='two-sided').pvalue)
            except ValueError: p_seed=1.0
            ci=cluster_bootstrap_ci(seed_d.rename(columns={'diff':metric}),metric)
            rows.append(dict(metric=metric,baseline=base,n_scenarios=len(d),n_seed_clusters=len(seed_d),
                             mean_difference=float(d.mean()),cluster_bootstrap_ci_low=float(ci[0]),
                             cluster_bootstrap_ci_high=float(ci[1]),scenario_wilcoxon_p=p_scenario,
                             seed_cluster_wilcoxon_p=p_seed,seed_level_rank_biserial=rank_biserial(seed_d['diff'])))
    comp=pd.DataFrame(rows)
    # Holm correction within each metric across the six baselines, using seed-cluster p-values.
    comp['holm_seed_p']=np.nan
    for metric,g in comp.groupby('metric'):
        idx=g.index.to_numpy(); comp.loc[idx,'holm_seed_p']=holm_adjust(g['seed_cluster_wilcoxon_p'].to_numpy())
    comp.to_csv(R/'paired_comparisons_clustered.csv',index=False)

    # Held-out comparison for the development-selected geometry-straddle policy.
    gs_path=R/'geometry_straddle_test.csv'
    if gs_path.exists():
        gs=pd.read_csv(gs_path)
        test=df[df.seed>=5].copy()
        held=[]
        for base in [TARGET,'two_sided_straddle','uncertainty_only']:
            a=gs.set_index(keys); bb=test[test.method==base].set_index(keys)
            j=a.join(bb,lsuffix='_geo_straddle',rsuffix='_base').reset_index()
            for metric in metrics:
                d=j[f'{metric}_geo_straddle']-j[f'{metric}_base']
                seed_d=j.assign(diff=d).groupby('seed',as_index=False)['diff'].mean()
                try: p=float(wilcoxon(seed_d['diff']).pvalue)
                except ValueError: p=1.0
                ci=cluster_bootstrap_ci(seed_d.rename(columns={'diff':metric}),metric)
                held.append(dict(metric=metric,baseline=base,mean_difference=float(d.mean()),
                                 ci_low=float(ci[0]),ci_high=float(ci[1]),seed_wilcoxon_p=p))
        pd.DataFrame(held).to_csv(R/'geometry_straddle_heldout_comparisons.csv',index=False)

    print('\nFINAL SUMMARY')
    print(summary.to_string(index=False))
    print('\nCLUSTERED COMPARISONS')
    print(comp.to_string(index=False))

if __name__=='__main__': main()
