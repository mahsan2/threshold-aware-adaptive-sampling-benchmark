# Extended benchmark: decision-relevant findings

## Principal 120-scenario benchmark at 100 points

| Policy | RMSE | Decision agreement | Missed exceedance | Balanced accuracy |
|---|---:|---:|---:|---:|
| Geometry-informed composite | 0.006865 | 0.975947 | 0.262863 | 0.863943 |
| Two-sided Straddle | 0.006979 | 0.975373 | 0.269884 | 0.860134 |
| Randomized Straddle | 0.007898 | 0.975000 | 0.288490 | 0.851320 |
| Posterior-risk | 0.008414 | 0.975467 | 0.283472 | 0.853987 |
| No-geometry composite | 0.007225 | 0.975387 | 0.291844 | 0.849450 |
| GP uncertainty | 0.006037 | 0.969733 | 0.373478 | 0.807995 |
| Space filling | 0.006036 | 0.970040 | 0.403111 | 0.793300 |
| Geometry only | 0.006995 | 0.966827 | 0.418244 | 0.785406 |

## Seed-clustered paired inference for missed exceedances

Composite minus GP uncertainty: mean difference -0.110615; 95% cluster-bootstrap CI [-0.152139, -0.071106]; Holm-adjusted p=0.001526.

Composite minus deterministic two-sided Straddle: mean difference -0.007021; 95% cluster-bootstrap CI [-0.028015, 0.015981]; Holm-adjusted p=0.504761.

Composite minus randomized Straddle: mean difference -0.025627; 95% cluster-bootstrap CI [-0.045183, -0.008464]; Holm-adjusted p=0.060303.

These results support a threshold-aware versus reconstruction-oriented distinction, but they do **not** support a claim that the composite acquisition is superior to deterministic Straddle overall.

## Geometry robustness

When localized-defect placement is independent of geometry, missed exceedance is 0.241342 for the full composite, 0.254021 for the no-geometry composite, 0.279286 for deterministic Straddle, and 0.355318 for GP uncertainty.

When defects are deliberately restricted to low-complexity regions, missed exceedance is 0.152718 for the full composite, 0.143086 for the no-geometry composite, 0.152010 for deterministic Straddle, and 0.262556 for GP uncertainty. Geometry therefore helps conditionally and can be mildly counterproductive when the process-error mechanism contradicts the prior.

## Held-out weight analysis

A simplex search on development seeds 0-4 selected uncertainty/geometry/decision weights 0.4/0.2/0.4. On held-out seeds 5-14, it achieved missed exceedance 0.266318 and agreement 0.975420. It did not establish a statistically clear improvement over the original pre-specified composite or deterministic Straddle.

## GP length-scale sensitivity

At length scales 0.10, 0.16, and 0.24, threshold-aware policies retain an advantage over GP uncertainty on missed exceedances. The relative ordering of the composite and deterministic Straddle changes with length scale, reinforcing the conclusion that the benchmark supports threshold awareness more strongly than a unique advantage of the composite formula.
