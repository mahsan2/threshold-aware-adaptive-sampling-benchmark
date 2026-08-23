#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import benchmark_extended as b

OUT=Path(__file__).resolve().parent; R=OUT/'results'; F=OUT/'figures'; F.mkdir(exist_ok=True)
LABELS={
'space_filling':'Space filling','geometry_only':'Geometry only','uncertainty_only':'GP uncertainty',
'two_sided_straddle':'Two-sided Straddle','randomized_straddle':'Randomized Straddle',
'posterior_risk':'Posterior risk','evidence_no_geometry':'Composite, no geometry','evidence_gated':'Geometry-informed composite'
}
MAIN=['space_filling','uncertainty_only','two_sided_straddle','randomized_straddle','evidence_gated']


def save(fig,name):
    fig.tight_layout(); fig.savefig(F/name,dpi=300,bbox_inches='tight'); plt.close(fig)


def main():
    # Example surface
    z,comp,_=b.make_surface(3,'mixed'); Z=z.reshape(b.GRID_N,b.GRID_N)
    fig,ax=plt.subplots(figsize=(6.2,4.7)); c=ax.contourf(b.Xg,b.Yg,Z,levels=25); ax.contour(b.Xg,b.Yg,np.abs(Z),levels=[b.TAU],linewidths=1.5)
    fig.colorbar(c,ax=ax,label='Dimensional deviation (normalized units)'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Example mixed-family ground-truth field')
    save(fig,'fig1_synthetic_surface_extended.png')

    df=pd.read_csv(R/'extended_trajectories.csv')
    for metric,ylabel,name in [
        ('missed_exceedance','Missed-exceedance rate','fig2_missed_exceedance_extended.png'),
        ('decision_agreement','Tolerance-decision agreement','fig3_decision_agreement_extended.png'),
        ('rmse','Surface reconstruction RMSE','fig4_rmse_extended.png')]:
        fig,ax=plt.subplots(figsize=(7.0,4.8))
        for m in MAIN:
            g=df[df.method==m].groupby('budget',as_index=False)[metric].agg(['mean','sem']).reset_index()
            line=ax.plot(g['budget'],g['mean'],label=LABELS[m],linewidth=1.8)[0]
            ax.fill_between(g['budget'],g['mean']-1.96*g['sem'],g['mean']+1.96*g['sem'],alpha=.12,color=line.get_color())
        ax.set_xlabel('Sampling budget (grid points)'); ax.set_ylabel(ylabel); ax.legend(frameon=False,ncol=2)
        save(fig,name)

    fam=pd.read_csv(R/'final_summary_by_family.csv')
    methods=['uncertainty_only','two_sided_straddle','evidence_gated']
    families=b.FAMILIES; xx=np.arange(len(families)); width=.24
    fig,ax=plt.subplots(figsize=(7.1,4.8))
    for i,m in enumerate(methods):
        vals=[fam[(fam.family==f)&(fam.method==m)].missed.iloc[0] for f in families]
        ax.bar(xx+(i-1)*width,vals,width,label=LABELS[m])
    ax.set_xticks(xx, ['Smooth','Waviness','Local defect','Mixed']); ax.set_ylabel('Missed-exceedance rate at budget 100'); ax.legend(frameon=False)
    save(fig,'fig5_family_comparison.png')

    ks=pd.read_csv(R/'kernel_sensitivity_summary.csv')
    fig,ax=plt.subplots(figsize=(6.6,4.6))
    for m in ['uncertainty_only','two_sided_straddle','evidence_gated']:
        g=ks[ks.method==m].sort_values('length_scale'); ax.plot(g.length_scale,g.missed,marker='o',label=LABELS[m])
    ax.set_xlabel('Fixed GP length scale'); ax.set_ylabel('Missed-exceedance rate at budget 100'); ax.legend(frameon=False)
    save(fig,'fig6_kernel_sensitivity.png')

if __name__=='__main__': main()
