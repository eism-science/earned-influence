"""Minimal illustrative simulation: contribution-weighted voting against equal weighting.

This is a separate Bernoulli-voting toy model. It illustrates how the comparison
is computed under one set of stipulated assumptions. It does not test the
empirical premise, and it does not implement the experiment proposed in the
paper (language-model agents, survey-calibrated preferences, collective stocks,
several aggregation rules). The model stipulates that correcting an error
improves an agent's competence and sets an adjustable dependence between
contribution and the propensity to correct, so its results cannot be evidence
for that dependence. Its numbers are outputs of the chosen parameters, not
findings about real electorates.

Model. Each round the electorate decides a binary issue that has a correct
answer. Agent i votes correctly with probability p_i. After a wrong vote the
agent either corrects (probability s_i; p_i moves toward 1) or entrenches
(p_i moves toward a floor, never upward). A correct vote changes nothing. Every SHIFT rounds
the world changes and competence is reset near chance; contribution and
correction propensity stay fixed. Contribution is x_i = AVTR_i * t_i in [0, 1],
with AVTR_i and t_i drawn independently. A rule with cap K gives weight
1 + (K - 1) x_i: K = 1 is equal weighting, K = 2 is the earned-influence weight
V_i = 1 + AVTR_i t_i proposed in the paper, and larger K are counterfactual
comparisons with more unequal weights. All rules are scored on the same votes.
A collective decision is correct if the weight on the correct answer exceeds
half the total; ties count as incorrect.

Dependence. `dep` is a latent parameter of a rank-based Gaussian-copula
construction linking x_i and s_i. It is NOT the correlation between them. The
script reports the mean, across simulated populations, of the within-population
Pearson and Spearman correlations between x and s.

Sign changes. Where the difference from equal weighting changes sign between
two adjacent `dep` grid points, the location is linearly interpolated. These
are exploratory, resolution-dependent estimates without uncertainty intervals,
not established thresholds.

Usage:
  python earned_influence_toy.py main          # main sweep, all caps (about 45 s)
  python earned_influence_toy.py sensitivity   # one-at-a-time sensitivity (about 5 min)
Outputs: results/results_main.json, results/results_sensitivity.json, and a printed summary.
"""
import json
import os
import platform
import sys

import numpy as np
import scipy
from scipy.stats import norm, spearmanr

BASE = dict(N=101, T=400, SHIFT=50, ETA=0.10, P0=0.55, FLOOR=0.20, AVTR=(1.2, 4.0))
CAPS = [1, 2, 3, 5, 10, 20]
DEPS = np.round(np.arange(-0.9, 0.91, 0.1), 2) + 0.0  # + 0.0 avoids printing -0.0


def weights(x, K):
    """Weight of each agent under cap K: 1 + (K - 1) x, in [1, K] for x in [0, 1]."""
    return 1 + (K - 1) * x


def update_competence(p, wrong, fix, ETA, FLOOR):
    """After a wrong vote: correcting agents move toward 1, the rest move toward FLOOR but never upward."""
    dig = wrong & ~fix
    p = np.where(fix, p + ETA * (1 - p), p)
    return np.where(dig, p - ETA * np.maximum(p - FLOOR, 0), p)


def simulate(dep, seed, n_pops, N, T, SHIFT, ETA, P0, FLOOR, AVTR):
    rng = np.random.default_rng(seed)
    avtr = rng.beta(AVTR[0], AVTR[1], (n_pops, N))
    t = rng.uniform(0, 1, (n_pops, N))
    x = avtr * t
    zx = norm.ppf((np.argsort(np.argsort(x, axis=1), axis=1) + 0.5) / N)
    zs = dep * zx + np.sqrt(1 - dep**2) * rng.standard_normal((n_pops, N))
    s = norm.cdf(zs)
    pearson = np.mean([np.corrcoef(x[k], s[k])[0, 1] for k in range(n_pops)])
    spearman = np.mean([spearmanr(x[k], s[k])[0] for k in range(n_pops)])
    w = {K: weights(x, K) for K in CAPS}
    correct = {K: np.zeros(n_pops) for K in CAPS}
    p = np.full((n_pops, N), P0)
    for r in range(T):
        if r % SHIFT == 0:
            p = np.clip(P0 + 0.05 * rng.standard_normal((n_pops, N)), 0.3, 0.8)
        v = rng.random((n_pops, N)) < p
        for K, wk in w.items():
            correct[K] += (wk * v).sum(1) > 0.5 * wk.sum(1)
        wrong = ~v
        fix = wrong & (rng.random((n_pops, N)) < s)
        p = update_competence(p, wrong, fix, ETA, FLOOR)
    acc = {K: correct[K] / T for K in CAPS}
    diff = {K: acc[K] - acc[1] for K in CAPS if K > 1}
    return dict(dep=float(dep), pearson=float(pearson), spearman=float(spearman),
                base_acc=float(acc[1].mean()),
                diff={K: float(d.mean()) for K, d in diff.items()},
                se={K: float(d.std(ddof=1) / np.sqrt(n_pops)) for K, d in diff.items()})


def weight_summary(params, n_pops=400, seed=999):
    """Distribution of realised weights under each cap (it does not depend on dep)."""
    rng = np.random.default_rng(seed)
    x = rng.beta(*params["AVTR"], (n_pops, params["N"])) * rng.uniform(0, 1, (n_pops, params["N"]))
    out = {}
    for K in CAPS:
        w = weights(x, K)
        n_eff = (w.sum(1) ** 2 / (w**2).sum(1)).mean()
        out[K] = dict(mean=float(w.mean()), p90=float(np.quantile(w, 0.9)), max=float(w.max()),
                      n_eff_share=float(n_eff / params["N"]))
    return out


def interpolated_sign_changes(deps, d):
    """Sign changes of d between adjacent grid points, linearly interpolated (exploratory)."""
    out = []
    for j in range(len(d) - 1):
        if np.sign(d[j]) != np.sign(d[j + 1]) and d[j] != d[j + 1]:
            out.append(float(deps[j] - d[j] * (deps[j + 1] - deps[j]) / (d[j + 1] - d[j])))
    return out


def sweep(params, n_pops, seed0):
    rows = [simulate(dep, seed0 + i, n_pops, **params) for i, dep in enumerate(DEPS)]
    changes = {K: interpolated_sign_changes(DEPS, [r["diff"][K] for r in rows]) for K in CAPS if K > 1}
    return rows, changes


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)


def main_sweep():
    n_pops, seed0 = 400, 1000
    rows, changes = sweep(BASE, n_pops=n_pops, seed0=seed0)
    wsum = weight_summary(BASE)
    print(f"Main sweep: {n_pops} simulated populations per setting, base parameters {BASE}.")
    print("Realised weights (cap: mean / 90th percentile / max; effective electorate as share of N):")
    for K in CAPS:
        ws = wsum[K]
        print(f"  cap {K:>2}: {ws['mean']:.3f} / {ws['p90']:.3f} / {ws['max']:.3f}; {ws['n_eff_share']:.3f}")
    print("\nDifference in share of correct decisions vs equal weighting, percentage points (standard error).")
    print(f"{'dep':>5} {'pearson':>8} {'spearman':>9} {'equal acc':>9} "
          + " ".join(f"{'cap ' + str(K):>15}" for K in CAPS[1:]))
    for r in rows:
        cells = " ".join(f"{100 * r['diff'][K]:+7.2f} ({100 * r['se'][K]:.2f})" for K in CAPS[1:])
        print(f"{r['dep']:5.1f} {r['pearson']:8.3f} {r['spearman']:9.3f} {100 * r['base_acc']:8.1f}% {cells}")
    print("\nInterpolated grid sign changes (exploratory, no uncertainty intervals):",
          {K: [round(c, 3) for c in v] for K, v in changes.items()})
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "results_main.json"), "w") as f:
        json.dump(dict(params=BASE, n_pops=n_pops, seed0=seed0, deps=DEPS.tolist(), caps=CAPS,
                       weights=wsum, rows=rows, interpolated_sign_changes=changes,
                       environment=environment()), f, indent=1, default=str)


def sensitivity():
    n_pops, seed0 = 200, 5000
    variants = [("N", 21), ("N", 51), ("N", 201), ("FLOOR", 0.05), ("FLOOR", 0.40),
                ("P0", 0.50), ("P0", 0.65), ("SHIFT", 10), ("SHIFT", 100),
                ("ETA", 0.03), ("ETA", 0.20), ("AVTR", (2.0, 2.0))]
    out = []
    print(f"Sensitivity: {n_pops} populations per setting; one parameter changed at a time from {BASE}.")
    print(f"{'change':>16} {'equal acc':>9} {'cap2 -0.9':>9} {'cap2 +0.9':>9} "
          f"{'cap10 -0.9':>10} {'cap10 +0.9':>10}  sign changes (cap 2)")
    for name, val in variants:
        params = dict(BASE)
        params[name] = val
        rows, changes = sweep(params, n_pops=n_pops, seed0=seed0)
        lo, hi = rows[0], rows[-1]
        out.append(dict(change={name: val}, rows=rows, interpolated_sign_changes=changes))
        print(f"{name + '=' + str(val):>16} {100 * rows[len(DEPS) // 2]['base_acc']:8.1f}% "
              f"{100 * lo['diff'][2]:+9.2f} {100 * hi['diff'][2]:+9.2f} "
              f"{100 * lo['diff'][10]:+10.2f} {100 * hi['diff'][10]:+10.2f}  "
              f"{[round(c, 3) for c in changes[2]]}", flush=True)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "results_sensitivity.json"), "w") as f:
        json.dump(dict(base=BASE, n_pops=n_pops, seed0=seed0, deps=DEPS.tolist(), caps=CAPS,
                       variants=out, environment=environment()), f, indent=1, default=str)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "main"
    if part in ("main", "all"):
        main_sweep()
    if part in ("sensitivity", "all"):
        sensitivity()
