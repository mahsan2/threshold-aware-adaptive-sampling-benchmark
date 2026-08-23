# Manuscript-to-result crosswalk

This file identifies the packaged result files supporting the principal numerical claims in the revised manuscript.

- Principal policy means at budget 100: `results/final_summary_all_methods.csv`
- Scenario-level final metrics for all policies: `results/extended_final_scenarios.csv`
- Full sampling trajectories: `results/extended_trajectories.csv`
- Seed-clustered paired comparisons and Holm adjustment: `results/paired_comparisons_clustered.csv`
- Family-specific summaries: `results/final_summary_by_family.csv`
- Noise-specific summaries: `results/final_summary_by_noise.csv`
- Geometry-placement robustness: `results/robustness_ablation_extended_results.csv` and `results/robustness_ablation_extended_summary.csv`
- GP length-scale sensitivity: `results/kernel_sensitivity_results.csv` and `results/kernel_sensitivity_summary.csv`
- Held-out composite-weight analysis: `results/evidence_weight_grid_dev_summary.csv`, `results/evidence_weight_selection.json`, and `results/evidence_tuned_heldout.csv`
- Held-out geometry-plus-Straddle analysis: `results/geometry_straddle_dev_summary.csv`, `results/geometry_straddle_selection.json`, and `results/geometry_straddle_test.csv`
- Reproduction check: `verify_reproducibility_extended.py`

All files are generated from synthetic data by scripts included in this archive.
