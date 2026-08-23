# Threshold-aware adaptive sampling for dimensional conformity inspection

This package is a self-contained extended reproducibility archive for the controlled dimensional-inspection benchmark. It reproduces the original four policies and adds the closest threshold-oriented comparators requested during the novelty audit.

## What was added

1. **Two-sided Straddle**: a dimensional-tolerance adaptation of the classic GP Straddle criterion, using the distance to the nearest boundary `||mu|-tau|` and `1.96*sigma`.
2. **Randomized Straddle**: a two-sided adaptation of the randomized confidence parameter described by Inatsu et al. (TMLR, 2024), with a chi-square(2) draw once per adaptive batch.
3. **Posterior-risk sampling**: samples where the posterior probability that the mean-based conformity label is wrong is largest.
4. **No-geometry ablation** of the original composite acquisition.
5. **Seed-clustered inference** to avoid treating repeated seed identities across family/noise conditions as fully independent experimental units.
6. **GP length-scale sensitivity**, **geometry-placement stress tests**, and **development/test weight analyses**.

## Core finding after the stronger benchmark

At a 100-point budget, the original geometry-informed composite reproduces the manuscript values (agreement about 0.97595; missed-exceedance about 0.26286). It remains clearly better than space-filling, geometry-only, and GP-uncertainty sampling on missed exceedances. However, it is **not statistically distinguishable from the direct two-sided Straddle comparator** under seed-clustered inference: mean missed-exceedance difference = -0.0070, 95% seed-cluster bootstrap CI [-0.0280, 0.0160], seed-level Wilcoxon p = 0.359. This result is intentionally preserved rather than hidden; it changes the defensible novelty claim from “a superior new threshold sampler” to “a metrology-specific benchmark showing that threshold-aware objectives, rather than reconstruction objectives, drive most of the false-safe reduction, while nominal geometry provides conditional rather than universal benefit.”

## Reproduce everything

```bash
python -m pip install -r requirements.txt
python run_all.py --workers 8
```

Typical runtime in the packaged environment is under a few minutes on a multicore workstation.

## Individual analyses

```bash
python benchmark_extended.py --workers 8
python tune_evidence_weights.py --workers 8
python tune_geometry_straddle.py --workers 8
python kernel_sensitivity.py --workers 8
python robustness_ablation_extended.py --workers 8
python analyze_extended.py
python make_figures_extended.py
python verify_reproducibility_extended.py
```

## Provenance and scope

All benchmark data are synthetic and dimensionless. The package does not contain physical CMM, optical-scanner, CT, or other metrology measurements. No physical validation is claimed. The same synthetic generator, fixed GP kernel, common initialization, and matched noise convention used in the original benchmark are retained.

## Main methods

- `space_filling`
- `geometry_only`
- `uncertainty_only`
- `evidence_gated` (original geometry-informed composite)
- `evidence_no_geometry`
- `two_sided_straddle`
- `randomized_straddle`
- `posterior_risk`
- `risk_geometry` (exploratory; not required for the main manuscript claims)

## Important interpretation

The direct level-set comparison is the major change. The code and revised manuscript do **not** claim that the geometry-informed composite outperforms Straddle overall because the benchmark does not support that statement. The defensible result is that threshold-aware strategies substantially reduce false-safe errors relative to reconstruction-oriented strategies, while geometry effects depend on the error mechanism.

## License

No reuse license is assigned in this archive. The authors should choose a license before public release.

## Submission-support files

- `SUMMARY_FINDINGS.md`: concise numerical interpretation of the extended benchmark.
- `MANUSCRIPT_RESULT_CROSSWALK.md`: maps manuscript claims to exact result files.
- `HIGHLIGHTS.txt`: four journal highlights, each under 85 characters.
- `COVER_LETTER_PRECISION_ENGINEERING.txt`: conservative cover-letter draft that does not overclaim superiority over Straddle.
- `CHANGELOG.md`: scientific and manuscript revision history.
- `REMAINING_MANUAL_ITEMS.txt`: author/funding/license/physical-validation items that cannot be inferred or fabricated.
