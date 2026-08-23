# Revision changelog

## Scientific changes

- Added deterministic **two-sided Straddle**, **randomized Straddle**, and **posterior-risk** threshold-aware comparators.
- Added a **no-geometry** ablation and an exploratory posterior-risk-plus-geometry comparator.
- Re-ran the full principal benchmark on the original 120 matched scenarios and common 100-point budget.
- Replaced scenario-level-only inference with **seed-clustered paired inference**: differences are averaged within each of the 15 seed identities across family/noise conditions, then tested using seed-level Wilcoxon tests and 10,000 cluster-bootstrap resamples.
- Added **held-out acquisition-weight analysis** using seeds 0-4 for development and 5-14 for testing.
- Added **geometry-plus-Straddle tuning** on the same development/test split.
- Added **GP length-scale sensitivity** at 0.10, 0.16, and 0.24.
- Extended the geometry-placement robustness study to include direct threshold-aware comparators.

## Main conclusion after re-analysis

The original geometry-informed composite reproduces its prior benchmark values and remains better than space-filling, geometry-only, and GP-uncertainty sampling on missed exceedances. It is **not statistically distinguishable from deterministic two-sided Straddle overall** under seed-clustered inference. The manuscript therefore no longer claims that the composite is a superior new level-set method. Its defensible contribution is a controlled metrology benchmark showing that threshold-aware objectives drive most of the false-safe reduction, while nominal geometry has conditional value.

## Manuscript changes

- Removed digital-twin implementation from the title and core novelty claim because no digital-twin system is evaluated.
- Reframed the paper around **dimensional conformity**, **false-safe error**, and **direct level-set comparison**.
- Rewrote the abstract, introduction, methods, results, discussion, limitations, and conclusion around the new analyses.
- Added direct level-set literature and reordered references by first appearance.
- Removed the previous stopping-rule analysis from the central contribution because it distracts from the stronger acquisition-objective result.
- Numbered the acquisition equations and aligned claims with the actual statistical evidence.
