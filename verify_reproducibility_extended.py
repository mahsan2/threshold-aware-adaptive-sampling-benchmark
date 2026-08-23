#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent; R=HERE/'results'
EXPECTED={
'evidence_gated': {'agreement':0.9759466666666666,'missed':0.262863130452806,'rmse':0.006864859461434},
'uncertainty_only': {'agreement':0.9697333333333333,'missed':0.373478339211506,'rmse':0.006036902950580},
'two_sided_straddle': {'agreement':0.9753733333333333,'missed':0.269883837324782,'rmse':0.006978907070017},
}
s=pd.read_csv(R/'final_summary_all_methods.csv').set_index('method')
for method,vals in EXPECTED.items():
    for k,v in vals.items():
        got=float(s.loc[method,k])
        if abs(got-v)>5e-12:
            raise SystemExit(f'FAIL {method} {k}: got {got}, expected {v}')
print('PASS: saved extended benchmark summaries match the packaged reference values.')
