#!/usr/bin/env python3
"""Extended reproducible benchmark for geometry-informed decision-aware adaptive sampling.

Adds two direct threshold-oriented comparators to the original four policies:
  * two_sided_straddle: two-sided extension of the classic GP straddle rule
    score = 1.96 * posterior_std - distance_to_nearest_tolerance_boundary.
  * posterior_risk: posterior probability that the mean-based conformity label is wrong.

Also includes an exploratory risk_geometry policy that combines posterior classification
risk with nominal geometry. The original evidence_gated policy is reproduced exactly.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from multiprocessing import Pool, cpu_count
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
RESULTS.mkdir(exist_ok=True)

GRID_N = 25
MAX_BUDGET = 100
BATCH = 4
INIT_N = 16
TAU = 0.035
FAMILIES = ["smooth", "waviness", "local_defect", "mixed"]
NOISE_LEVELS = [0.003, 0.008]
SEEDS = list(range(15))
ORIGINAL_METHODS = ["space_filling", "geometry_only", "uncertainty_only", "evidence_gated"]
DIRECT_COMPARATORS = ["two_sided_straddle", "randomized_straddle", "posterior_risk", "evidence_no_geometry"]
EXPLORATORY_METHODS = ["risk_geometry"]
METHODS = ORIGINAL_METHODS + DIRECT_COMPARATORS + EXPLORATORY_METHODS

x = np.linspace(0, 1, GRID_N)
y = np.linspace(0, 1, GRID_N)
Xg, Yg = np.meshgrid(x, y, indexing="xy")
XY = np.column_stack([Xg.ravel(), Yg.ravel()])


def norm01(a):
    a = np.asarray(a, float)
    mn, mx = np.nanmin(a), np.nanmax(a)
    if mx - mn < 1e-12:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def make_surface(seed, family):
    """Exact reproduction of the original synthetic benchmark generator."""
    rng = np.random.default_rng(seed * 1009 + FAMILIES.index(family) * 9176 + 13)
    cx, cy = rng.uniform(.25, .75, 2)
    sx, sy = rng.uniform(.10, .20, 2)
    h = (0.45 * np.sin(np.pi * Xg) * np.sin(np.pi * Yg)
         + 0.25 * np.exp(-(((Xg - cx) / sx) ** 2 + ((Yg - cy) / sy) ** 2) / 2)
         + 0.12 * (Xg - 0.5) ** 2)
    gy, gx = np.gradient(h, y, x)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    gyy, _ = np.gradient(gy, y, x)
    _, gxx = np.gradient(gx, y, x)
    lap = np.abs(gxx + gyy)
    complexity = norm01(gmag + 0.12 * lap)

    raw = gaussian_filter(rng.normal(size=(GRID_N, GRID_N)), sigma=2.2, mode="reflect")
    raw = (raw - raw.mean()) / (raw.std() + 1e-12)
    phi1, phi2 = rng.uniform(0, 2 * np.pi, 2)
    dev = 0.010 * raw + 0.018 * (complexity - complexity.mean())

    if family == "smooth":
        dev += 0.028 * np.sin(2 * np.pi * Xg + phi1) * np.cos(2 * np.pi * Yg + phi2)
    elif family == "waviness":
        dev += 0.036 * np.sin(6 * np.pi * Xg + phi1) * np.sin(4 * np.pi * Yg + phi2)
    elif family == "local_defect":
        flatc = complexity.ravel()
        if rng.random() < 0.7:
            candidates = np.where(flatc >= np.quantile(flatc, .7))[0]
            idx = rng.choice(candidates)
        else:
            idx = rng.integers(len(flatc))
        dx, dy = XY[idx]
        sig = rng.uniform(.035, .075)
        amp = rng.choice([-1, 1]) * rng.uniform(.075, .105)
        dev += amp * np.exp(-((Xg - dx) ** 2 + (Yg - dy) ** 2) / (2 * sig ** 2))
        dev += 0.012 * np.sin(2 * np.pi * Xg + phi1)
    elif family == "mixed":
        flatc = complexity.ravel()
        if rng.random() < 0.65:
            candidates = np.where(flatc >= np.quantile(flatc, .65))[0]
            idx = rng.choice(candidates)
        else:
            idx = rng.integers(len(flatc))
        dx, dy = XY[idx]
        sig = rng.uniform(.04, .08)
        amp = rng.choice([-1, 1]) * rng.uniform(.065, .095)
        dev += 0.025 * np.sin(5 * np.pi * Xg + phi1) * np.cos(3 * np.pi * Yg + phi2)
        dev += amp * np.exp(-((Xg - dx) ** 2 + (Yg - dy) ** 2) / (2 * sig ** 2))
    else:
        raise ValueError(family)
    dev -= np.median(dev)
    return dev.ravel(), complexity.ravel(), h.ravel()


def init_indices(seed):
    rng = np.random.default_rng(seed + 123456)
    sampler = qmc.LatinHypercube(d=2, seed=rng)
    pts = sampler.random(n=INIT_N)
    inds = []
    for p in pts:
        j = np.argmin(((XY - p) ** 2).sum(axis=1))
        if j not in inds:
            inds.append(j)
    while len(inds) < INIT_N:
        sel = XY[inds]
        d2 = ((XY[:, None, :] - sel[None, :, :]) ** 2).sum(axis=2)
        score = np.sqrt(d2.min(axis=1))
        score[inds] = -1
        inds.append(int(np.argmax(score)))
    return np.array(inds, dtype=int)


def fit_gp(sel, obs, noise_sigma, length_scale=0.16):
    kernel = ConstantKernel(0.05 ** 2, constant_value_bounds="fixed") * Matern(
        length_scale=length_scale, length_scale_bounds="fixed", nu=1.5
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=noise_sigma ** 2,
        optimizer=None,
        normalize_y=False,
        random_state=0,
    )
    gp.fit(XY[sel], obs[sel])
    mu, std = gp.predict(XY, return_std=True)
    return mu, std


def min_dist_to_selected(sel):
    d2 = ((XY[:, None, :] - XY[sel][None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def posterior_probs(mu, std):
    s = np.maximum(std, 1e-12)
    p_accept = norm.cdf((TAU - mu) / s) - norm.cdf((-TAU - mu) / s)
    p_accept = np.clip(p_accept, 0.0, 1.0)
    p_exceed = 1.0 - p_accept
    return p_accept, p_exceed


def base_score(method, mu, std, complexity, dist, random_beta_sqrt=None):
    if method == "space_filling":
        return dist.copy()
    if method == "geometry_only":
        return 0.65 * complexity + 0.35 * dist
    if method == "uncertainty_only":
        return norm01(std)
    if method == "evidence_gated":
        u = norm01(std)
        proximity = np.exp(-np.abs(np.abs(mu) - TAU) / (0.30 * TAU))
        score = 0.40 * u + 0.25 * complexity + 0.35 * proximity
        return 0.95 * score + 0.05 * dist
    if method == "evidence_no_geometry":
        u = norm01(std)
        proximity = np.exp(-np.abs(np.abs(mu) - TAU) / (0.30 * TAU))
        score = (0.40 / 0.75) * u + (0.35 / 0.75) * proximity
        return 0.95 * score + 0.05 * dist
    if method == "two_sided_straddle":
        # Two-sided extension of the classic straddle criterion. The closest
        # tolerance boundary is at +/- TAU, so boundary distance is ||mu|-TAU|.
        return norm01(1.96 * std - np.abs(np.abs(mu) - TAU))
    if method == "randomized_straddle":
        if random_beta_sqrt is None:
            raise ValueError("random_beta_sqrt is required for randomized_straddle")
        return norm01(random_beta_sqrt * std - np.abs(np.abs(mu) - TAU))
    if method == "posterior_risk":
        p_accept, p_exceed = posterior_probs(mu, std)
        # Probability that the deterministic posterior-mean label is incorrect.
        return np.minimum(p_accept, p_exceed)
    if method == "risk_geometry":
        p_accept, p_exceed = posterior_probs(mu, std)
        risk = norm01(np.minimum(p_accept, p_exceed))
        score = 0.75 * risk + 0.25 * complexity
        return 0.95 * score + 0.05 * dist
    raise ValueError(method)


def choose_batch(method, sel, mu, std, complexity, batch=BATCH, random_beta_sqrt=None):
    remaining = np.ones(len(XY), dtype=bool)
    remaining[sel] = False
    dist = norm01(min_dist_to_selected(sel))
    score = base_score(method, mu, std, complexity, dist, random_beta_sqrt=random_beta_sqrt)
    score = np.asarray(score, float).copy()
    score[~remaining] = -np.inf

    picked = []
    temp = score.copy()
    for _ in range(batch):
        j = int(np.argmax(temp))
        picked.append(j)
        if len(picked) == batch:
            break
        d = np.sqrt(((XY - XY[j]) ** 2).sum(axis=1))
        # Same within-batch separation mechanism for every policy.
        temp *= (0.35 + 0.65 * norm01(d))
        temp[~remaining] = -np.inf
        temp[picked] = -np.inf
    return np.array(picked, dtype=int)


def confusion_metrics(mu, true):
    truth = np.abs(true) > TAU
    pred = np.abs(mu) > TAU
    tp = int(np.logical_and(truth, pred).sum())
    fn = int(np.logical_and(truth, ~pred).sum())
    fp = int(np.logical_and(~truth, pred).sum())
    tn = int(np.logical_and(~truth, ~pred).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    fpr = fp / max(1, fp + tn)
    specificity = tn / max(1, fp + tn)
    bal = 0.5 * (recall + specificity)
    return dict(
        tp=tp, fn=fn, fp=fp, tn=tn,
        prevalence=(tp + fn) / len(true),
        recall=recall,
        precision=precision,
        false_positive_rate=fpr,
        balanced_accuracy=bal,
    )


def metrics(mu, std, true):
    err = mu - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    truth = np.abs(true) > TAU
    pred = np.abs(mu) > TAU
    agreement = float(np.mean(truth == pred))
    denom = max(1, int(truth.sum()))
    missed = float(np.logical_and(truth, ~pred).sum() / denom)
    return rmse, mae, agreement, missed, float(np.mean(std)), float(np.quantile(std, .95))


def internal_gate(mu, std, prev_labels, stable_count):
    labels = np.abs(mu) > TAU
    change = 1.0 if prev_labels is None else float(np.mean(labels != prev_labels))
    p_accept, p_exceed = posterior_probs(mu, std)
    conf = np.maximum(p_accept, p_exceed)
    conf_q05 = float(np.quantile(conf, .05))
    ambiguous_90 = float(np.mean(conf < .90))
    critical = np.abs(np.abs(mu) - TAU) <= (1.96 * std + .006)
    crit_q95 = float(np.quantile(std[critical], .95)) if critical.any() else 0.0
    global_q95 = float(np.quantile(std, .95))
    stable_count = stable_count + 1 if change <= .005 else 0
    stop = (conf_q05 >= .80 and ambiguous_90 <= .40 and stable_count >= 2)
    return stop, labels, stable_count, change, crit_q95, global_q95, conf_q05, ambiguous_90


def scenario_data(seed, family, noise_sigma):
    true, comp, nominal = make_surface(seed, family)
    rng = np.random.default_rng(seed * 31337 + int(noise_sigma * 1e6) * 7 + FAMILIES.index(family) * 101)
    obs = true + rng.normal(0, noise_sigma, size=len(true))
    return true, comp, nominal, obs


def run_policy(seed, family, noise_sigma, method, length_scale=0.16, store_trajectory=True):
    true, comp, _, obs = scenario_data(seed, family, noise_sigma)
    sel = init_indices(seed)
    method_rng = np.random.default_rng(seed * 7919 + FAMILIES.index(family) * 104729 + int(noise_sigma * 1e6) * 17 + 2024)
    rows = []
    prev = None
    stable = 0
    gate_budget = np.nan
    gate_success = np.nan
    while True:
        mu, std = fit_gp(sel, obs, noise_sigma, length_scale=length_scale)
        rmse, mae, agr, missed, meanstd, q95std = metrics(mu, std, true)
        stop, prev2, stable, change, crit_q95, global_q95, conf_q05, amb90 = internal_gate(mu, std, prev, stable)
        if method == "evidence_gated" and stop and np.isnan(gate_budget):
            gate_budget = len(sel)
            gate_success = bool(agr >= .97 and missed <= .10)
        if store_trajectory:
            c = confusion_metrics(mu, true)
            rows.append(dict(
                seed=seed, family=family, noise=noise_sigma, method=method, budget=len(sel),
                rmse=rmse, mae=mae, decision_agreement=agr, missed_exceedance=missed,
                mean_std=meanstd, q95_std=q95std, label_change=change,
                critical_q95_std=crit_q95, confidence_q05=conf_q05,
                ambiguous_90=amb90, gate_fired=bool(method == "evidence_gated" and stop),
                gate_budget=gate_budget, gate_success=gate_success,
                exceedance_fraction=float(np.mean(np.abs(true) > TAU)), **c
            ))
        prev = prev2
        if len(sel) >= MAX_BUDGET:
            break
        beta_sqrt = float(np.sqrt(method_rng.chisquare(df=2))) if method == "randomized_straddle" else None
        add = choose_batch(method, sel, mu, std, comp, min(BATCH, MAX_BUDGET - len(sel)), random_beta_sqrt=beta_sqrt)
        sel = np.concatenate([sel, add])
    if store_trajectory:
        return rows
    c = confusion_metrics(mu, true)
    return dict(seed=seed, family=family, noise=noise_sigma, method=method, budget=len(sel),
                rmse=rmse, mae=mae, decision_agreement=agr, missed_exceedance=missed, **c)


def run_scenario(args):
    seed, family, noise, methods = args
    out = []
    for method in methods:
        out.extend(run_policy(seed, family, noise, method, store_trajectory=True))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, min(8, cpu_count() // 2 or 1)))
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = p.parse_args()
    methods = args.methods
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"Unknown methods: {sorted(unknown)}")

    config = dict(
        grid_n=GRID_N, max_budget=MAX_BUDGET, batch=BATCH, init_n=INIT_N, tau=TAU,
        families=FAMILIES, noise_levels=NOISE_LEVELS, seeds=args.seeds, methods=methods,
        gp_kernel="0.05^2 * Matern(nu=1.5, length_scale=0.16)",
        straddle_beta=1.96,
        posterior_risk_definition="min(P(accept|data), P(exceed|data))",
    )
    (RESULTS / "extended_benchmark_config.json").write_text(json.dumps(config, indent=2))

    tasks = [(s, f, n, methods) for s in args.seeds for f in FAMILIES for n in NOISE_LEVELS]
    t0 = time.time()
    if args.workers == 1:
        chunks = [run_scenario(t) for t in tasks]
    else:
        with Pool(processes=args.workers) as pool:
            chunks = pool.map(run_scenario, tasks)
    rows = [r for chunk in chunks for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "extended_trajectories.csv", index=False)
    final = df[df.budget == MAX_BUDGET].copy()
    final.to_csv(RESULTS / "extended_final_scenarios.csv", index=False)
    summary = final.groupby("method", as_index=False).agg(
        n=("decision_agreement", "size"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        agreement_mean=("decision_agreement", "mean"),
        agreement_sd=("decision_agreement", "std"),
        missed_mean=("missed_exceedance", "mean"),
        missed_sd=("missed_exceedance", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
        fpr_mean=("false_positive_rate", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        prevalence_mean=("prevalence", "mean"),
    )
    summary.to_csv(RESULTS / "extended_summary_budget100.csv", index=False)
    print(summary.to_string(index=False))
    print(f"elapsed_s={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
