"""Simulation (b): your SRM check is peeking too.

Platforms recompute SRM every day (or every hour). That is a repeated
significance test, and the usual peeking arithmetic applies to it. We
compare monitoring rules over a 14-day test:

- chi-square at p < 0.05, checked daily       (the common default)
- chi-square at p < 0.001, checked daily      (strict threshold)
- chi-square at p < 0.001, checked hourly     (strict threshold, many looks)
- anytime-valid mixture LR at alpha = 0.001   (Dirichlet-multinomial, as in
  Lindon & Malek 2022), checked hourly - valid at any number of looks
- the same mixture LR with its threshold calibrated by simulation so that its
  realized false alarm rate matches chi-square p < 0.001 daily
- chi-square at p < 0.001, single look at the end (reference)

Why the calibrated variant: Ville's inequality is conservative. At nominal
alpha = 0.001 the realized false alarm rate of the mixture is around 0.01%,
so comparing it with chi-square at the same nominal alpha understates its
power. The calibrated variant compares the two at equal realized false
alarms. Note what calibration gives up: the guarantee then holds only for this
traffic, horizon and look schedule (it comes from simulation, not from Ville).

Under H0 (true 50/50) we report the false alarm rate; under H1 (true share
49.5% or 49% in treatment) the detection rate and median day of the alarm.

Run:  python experiments/sim_srm_monitoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style  # noqa: E402
from abdq.stats import log_mixture_srm  # noqa: E402

DAYS = 14
HOURS = DAYS * 24
RUNS = 20_000
CONCENTRATION = 10_000.0
SEED = 7


def chi2_p(t: np.ndarray, n: np.ndarray) -> np.ndarray:
    exp = n / 2
    stat = ((t - exp) ** 2 + ((n - t) - exp) ** 2) / exp
    return stats.chi2.sf(stat, df=1)


def first_alarm(alarm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """alarm: (runs, looks) bool. Returns (any_alarm, index of first alarm or -1)."""
    any_ = alarm.any(axis=1)
    idx = np.where(any_, alarm.argmax(axis=1), -1)
    return any_, idx


def simulate_paths(daily_traffic: int, share: float, rng: np.random.Generator):
    per_hour = daily_traffic // 24
    t_hourly = rng.binomial(per_hour, share, size=(RUNS, HOURS)).cumsum(axis=1).astype(float)
    n_hourly = per_hour * np.arange(1, HOURS + 1, dtype=float)
    day_idx = np.arange(23, HOURS, 24)               # end of each day
    p_daily = chi2_p(t_hourly[:, day_idx], n_hourly[day_idx])
    p_hourly = chi2_p(t_hourly, n_hourly)
    counts = np.stack([n_hourly - t_hourly, t_hourly], axis=-1)
    log_m = log_mixture_srm(counts, np.array([0.5, 0.5]), CONCENTRATION)
    return p_daily, p_hourly, log_m


def calibrate_mixture(daily_traffic: int, rng: np.random.Generator) -> tuple[float, float]:
    """Threshold on log M so the mixture's false alarm rate under H0 equals that of
    chi-square p < 0.001 checked daily. Uses its own null sample (not reused below)."""
    p_daily, _, log_m = simulate_paths(daily_traffic, 0.5, rng)
    target = float((p_daily < 1e-3).any(axis=1).mean())
    threshold = float(np.quantile(log_m.max(axis=1), 1 - target))
    return threshold, target


def run_scenario(daily_traffic: int, share: float, threshold: float, rng: np.random.Generator) -> list[dict]:
    p_daily, p_hourly, log_m = simulate_paths(daily_traffic, share, rng)

    rules = {
        "chi-square p<0.05, daily": (p_daily < 0.05, 1),
        "chi-square p<0.001, daily": (p_daily < 1e-3, 1),
        "chi-square p<0.001, hourly": (p_hourly < 1e-3, 1 / 24),
        "mixture LR alpha=0.001, hourly": (log_m >= np.log(1e3), 1 / 24),
        "mixture LR calibrated, hourly": (log_m >= threshold, 1 / 24),
        "chi-square p<0.001, end only": (p_daily[:, -1:] < 1e-3, None),
    }
    rows = []
    for name, (alarm, step_days) in rules.items():
        any_, idx = first_alarm(alarm)
        if step_days is None:
            day = np.where(any_, DAYS, np.nan)
        else:
            day = np.where(any_, (idx + 1) * step_days, np.nan)
        rows.append({
            "daily_traffic": daily_traffic,
            "true_treatment_share": share,
            "rule": name,
            "alarm_rate": float(any_.mean()),
            "median_alarm_day": float(np.nanmedian(day)) if any_.any() else np.nan,
        })
    return rows


def main() -> None:
    _style.apply()
    rng = np.random.default_rng(SEED)
    rows = []
    calib = []
    for traffic in (10_000, 100_000):
        threshold, target = calibrate_mixture(traffic, rng)
        calib.append({"daily_traffic": traffic, "target_false_alarm_rate": target,
                      "log_threshold": threshold, "equivalent_nominal_alpha": float(np.exp(-threshold))})
        print(f"{traffic:,}/day: calibrated mixture threshold log M >= {threshold:.2f} "
              f"(nominal alpha {np.exp(-threshold):.3f}) to match {target:.2%} false alarms")
        for share in (0.5, 0.495, 0.49):
            rows += run_scenario(traffic, share, threshold, rng)
    pd.DataFrame(calib).to_csv(_style.RESULTS / "sim_srm_monitoring_calibration.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(_style.RESULTS / "sim_srm_monitoring.csv", index=False)

    wide = df.pivot_table(index=["daily_traffic", "rule"], columns="true_treatment_share",
                          values=["alarm_rate", "median_alarm_day"], sort=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(wide.round(3))

    # figure: false alarms vs detection at 10k users/day (at 100k/day every rule detects)
    fig, ax = _style.plt.subplots(figsize=(9.5, 4.6))
    order = ["chi-square p<0.05, daily", "chi-square p<0.001, daily", "chi-square p<0.001, hourly",
             "mixture LR alpha=0.001, hourly", "mixture LR calibrated, hourly", "chi-square p<0.001, end only"]
    labels = ["p < 0.05\ndaily", "p < 0.001\ndaily", "p < 0.001\nhourly", "anytime-valid\nα = 0.001",
              "anytime-valid\ncalibrated α", "p < 0.001\nend only"]
    sub = df[df["daily_traffic"] == 10_000].set_index(["rule", "true_treatment_share"])
    x = np.arange(len(order))
    w = 0.38
    fa = [sub.loc[(r, 0.5), "alarm_rate"] for r in order]
    de = [sub.loc[(r, 0.495), "alarm_rate"] for r in order]
    ax.bar(x - w / 2 - 0.01, fa, width=w, color=_style.SERIES[1], label="false alarm (true split 50/50)")
    ax.bar(x + w / 2 + 0.01, de, width=w, color=_style.SERIES[0], label="detection (true split 49.5/50.5)")
    for xi, v in zip(x - w / 2 - 0.01, fa):
        ax.text(xi, v + 0.015, f"{v:.1%}", ha="center", va="bottom", fontsize=9, color=_style.TEXT)
    for xi, v in zip(x + w / 2 + 0.01, de):
        ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=_style.TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(_style.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("simulated tests with an SRM alarm")
    ax.set_title("Monitoring SRM is repeated testing", pad=22)
    ax.text(0, 1.01, "14-day test, 10,000 users/day, 20,000 simulated tests per bar.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=_style.TEXT_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    print("saved", _style.save(fig, "srm_monitoring.png"))


if __name__ == "__main__":
    main()
