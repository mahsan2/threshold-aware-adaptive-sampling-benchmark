# Data dictionary

## `results/extended_trajectories.csv`
One row per scenario, method, and sampling budget.

- `seed`: synthetic realization seed (0-14).
- `family`: `smooth`, `waviness`, `local_defect`, or `mixed`.
- `noise`: simulated Gaussian measurement-noise standard deviation.
- `method`: sampling policy.
- `budget`: number of acquired grid points.
- `rmse`, `mae`: reconstruction error against known synthetic ground truth.
- `decision_agreement`: fraction of grid cells whose conformity label matches ground truth.
- `missed_exceedance`: false-safe conditional rate, FN/(TP+FN).
- `precision`, `recall`, `false_positive_rate`, `balanced_accuracy`: class-balance-aware decision metrics.
- `prevalence` / `exceedance_fraction`: fraction of ground-truth grid cells beyond tolerance.
- remaining columns: posterior uncertainty and stopping-gate diagnostics.

## `results/extended_final_scenarios.csv`
Budget-100 rows extracted from the trajectory file.

## `results/final_summary_all_methods.csv`
Pooled budget-100 means for every policy.

## `results/paired_comparisons_clustered.csv`
Paired comparisons of the geometry-informed composite against each baseline. The conservative inference treats the 15 seed identities as clusters because the same initial Latin-hypercube design is reused across family/noise combinations sharing a seed.

## `results/final_summary_by_family.csv`
Budget-100 performance stratified by surface family.

## `results/kernel_sensitivity_*.csv`
Sensitivity to fixed Matérn-3/2 GP length scales 0.10, 0.16, and 0.24.

## `results/robustness_ablation_extended_*.csv`
Defect-placement stress tests where localized defects are independent of nominal geometry or deliberately placed in low-complexity regions.

## `results/evidence_weight_grid_dev*.csv` and `evidence_tuned_heldout.csv`
A separate development/test analysis. Candidate U/G/D weights are selected only on seeds 0-4 and then frozen on seeds 5-14. This is not used to overwrite the originally pre-specified 0.40/0.25/0.35 policy.

## `results/geometry_straddle_*.csv`
Development/test study of an additive geometry term on top of two-sided Straddle.
