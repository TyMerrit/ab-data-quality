"""Simulation (a): when does SRM catch selective exposure loss before it fools you?

Setup. Assignment is perfect, 50/50. In treatment, a fraction L of exposure
events is never logged, and the analysis runs on logged (exposed) units, as
most dashboards do. Lost units are not random: they come from some part of
the metric distribution.

Two tests race each other as the experiment grows:

- the SRM check on logged units (chi-square, p < 0.001) - catches the loss;
- the A/B test itself (Welch t-test, p < 0.05) - there is no true effect,
  so every rejection is a false positive caused by the loss.

Theory (normal approximation, m logged units per arm before loss):

    z_srm  ~ L * sqrt(m / 2)
    z_test ~ d * L * sqrt(m / 2),    d = (mean of all units - mean of lost units) / sd

Both grow like sqrt(m), so sample size does not decide the race. The ratio
z_test / z_srm = |d| does. The A/B test crosses its 1.96 threshold when
z_srm = 1.96 / |d|, while SRM needs z_srm = 3.29. So once |d| > 1.96 / 3.29
~ 0.6, a fake effect is more likely than not before SRM is more likely than
not to fire. Selective loss on heavy-tailed metrics (revenue, engagement)
easily has |d| well above that.

Run:  python experiments/sim_exposure_loss.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style  # noqa: E402

LOSS = 0.01              # 1% of treatment exposures never logged
SIGMA = 1.0              # revenue per user ~ LogNormal(0, SIGMA)
SRM_ALPHA = 1e-3
TEST_ALPHA = 0.05
M_GRID_MC = [3_000, 10_000, 30_000, 100_000, 300_000]
RUNS = 200
SEED = 20261016

# Mechanisms: probability that a treatment unit's exposure is lost, as a function
# of its revenue quantile. Each loses LOSS of units overall.
MECHANISMS = {
    "random loss": lambda q: np.full_like(q, LOSS),
    "lost among top 20% spenders": lambda q: np.where(q > 0.80, LOSS / 0.20, 0.0),
    "lost among top 2% spenders": lambda q: np.where(q > 0.98, LOSS / 0.02, 0.0),
}


def mechanism_d(fn, n: int = 2_000_000, seed: int = 0) -> float:
    """Standardized distance between all units and lost units, by simulation."""
    rng = np.random.default_rng(seed)
    y = rng.lognormal(0, SIGMA, size=n)
    q = stats.rankdata(y) / n
    w = fn(q)
    mu_lost = float((w * y).sum() / w.sum())
    return abs(float((y.mean() - mu_lost) / y.std()))


def power(z: np.ndarray, crit: float) -> np.ndarray:
    return stats.norm.cdf(np.abs(z) - crit) + stats.norm.cdf(-np.abs(z) - crit)


def monte_carlo(fn, m: int, runs: int, rng: np.random.Generator) -> tuple[float, float]:
    srm_hits = 0
    fp_hits = 0
    for _ in range(runs):
        mt = int(rng.binomial(2 * m, 0.5))      # assignment itself is random
        yc = rng.lognormal(0, SIGMA, size=2 * m - mt)
        yt = rng.lognormal(0, SIGMA, size=mt)
        # quantile of each treatment unit in the population distribution
        q = stats.lognorm.cdf(yt, s=SIGMA)
        keep = rng.random(mt) >= fn(q)
        yt_kept = yt[keep]
        counts = np.array([len(yc), keep.sum()], dtype=float)
        exp = counts.sum() / 2
        p_srm = stats.chi2.sf(((counts - exp) ** 2 / exp).sum(), df=1)
        p_test = stats.ttest_ind(yt_kept, yc, equal_var=False).pvalue
        srm_hits += p_srm < SRM_ALPHA
        fp_hits += p_test < TEST_ALPHA
    return srm_hits / runs, fp_hits / runs


def main() -> None:
    _style.apply()
    rng = np.random.default_rng(SEED)
    d_values = {name: mechanism_d(fn) for name, fn in MECHANISMS.items()}
    print("d (standardized distance of lost units):", {k: round(v, 2) for k, v in d_values.items()})

    m_curve = np.logspace(np.log10(2_000), np.log10(3_000_000), 200)
    z_srm = LOSS * np.sqrt(m_curve / 2)
    srm_detect = power(z_srm, stats.norm.isf(SRM_ALPHA / 2))

    rows = []
    for name, fn in MECHANISMS.items():
        for m in M_GRID_MC:
            srm_mc, fp_mc = monte_carlo(fn, m, RUNS, rng)
            rows.append({"mechanism": name, "d": d_values[name], "units_per_arm": m,
                         "srm_detection_mc": srm_mc, "false_positive_mc": fp_mc})
            print(f"{name:30s} m={m:>9,}  SRM detects {srm_mc:5.1%}   A/B false positive {fp_mc:5.1%}")
    mc = pd.DataFrame(rows)
    mc.to_csv(_style.RESULTS / "sim_exposure_loss_mc.csv", index=False)

    analytic = pd.DataFrame({"units_per_arm": m_curve, "srm_detection": srm_detect})
    for name, d in d_values.items():
        analytic[f"false_positive[{name}]"] = power(d * z_srm, stats.norm.isf(TEST_ALPHA / 2))
    analytic.to_csv(_style.RESULTS / "sim_exposure_loss_analytic.csv", index=False)

    # ---- figure ------------------------------------------------------------
    fig, ax = _style.plt.subplots(figsize=(9, 5))
    ax.plot(m_curve, srm_detect, color=_style.TEXT, linewidth=2.4, label="SRM check catches the loss (p < 0.001)")
    for (name, d), color in zip(d_values.items(), _style.SERIES):
        fp = power(d * z_srm, stats.norm.isf(TEST_ALPHA / 2))
        ax.plot(m_curve, fp, color=color, linestyle="--", label=f"A/B test shows a fake effect: {name} (d = {d:.1f})")
        sub = mc[mc["mechanism"] == name]
        ax.scatter(sub["units_per_arm"], sub["false_positive_mc"], color=color, s=40,
                   edgecolor=_style.SURFACE, linewidth=1.5, zorder=3)
    sub = mc[mc["mechanism"] == "random loss"]
    ax.scatter(sub["units_per_arm"], sub["srm_detection_mc"], color=_style.TEXT, s=40,
               edgecolor=_style.SURFACE, linewidth=1.5, zorder=3, marker="s")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(_style.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("logged units per arm")
    ax.set_ylabel("probability")
    ax.set_title("1% of treatment exposures lost: who notices first?", pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), fontsize=9.5, ncol=1)
    ax.text(0, 1.01, "Lines: normal approximation. Dots: Monte Carlo, 200 runs per point. "
            "Revenue ~ LogNormal(0, 1); no true effect.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=_style.TEXT_2)
    print("saved", _style.save(fig, "exposure_loss_race.png"))


if __name__ == "__main__":
    main()
