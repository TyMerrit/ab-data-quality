"""Step 2: the assignment layer.

Source of truth: the assignment log. Questions: does the realized split match
the design, and does every unit stay in one variant?
"""

from __future__ import annotations

import pandas as pd

from abdq.config import ExperimentConfig
from abdq.results import CheckKind, CheckResult, CheckStatus
from abdq.stats import mixture_srm_alarm, srm_chi2

C = CheckKind.CONFIRMATORY


def unit_table(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """One row per unit: its first assignment. Also returns crossover unit ids."""
    per_unit = assignments.groupby("unit_id")["variant"].nunique()
    crossover = per_unit.index[per_unit > 1]
    units = assignments.sort_values("assigned_at").drop_duplicates("unit_id", keep="first")
    return units.set_index("unit_id"), crossover


def srm_result(
    step: str,
    name: str,
    counts: dict[str, int],
    config: ExperimentConfig,
    population: str,
    kind: CheckKind = C,
) -> CheckResult:
    th = config.thresholds
    stat, p = srm_chi2(counts, config.expected_split)
    alarm_mix, log_m = mixture_srm_alarm(counts, config.expected_split, th.srm_alpha)
    total = sum(counts.values())
    shares = {v: (counts.get(v, 0) / total if total else float("nan")) for v in config.variants}
    if config.srm_method == "anytime":
        failed = alarm_mix
        rule = f"mixture LR >= 1/{th.srm_alpha:g}"
    else:
        failed = p < th.srm_alpha
        rule = f"chi-square p < {th.srm_alpha:g}"
    status = CheckStatus.FAIL if failed else CheckStatus.PASS
    shares_txt = ", ".join(f"{v}={s:.4f}" for v, s in shares.items())
    msg = (
        f"SRM on {population}: observed shares {shares_txt}; p = {p:.2e} "
        f"({'alarm' if failed else 'ok'} under {rule})."
    )
    if failed:
        msg += " SRM is a symptom, not a cause: find out where the units went before reading any metric."
    return CheckResult(step, name, status, kind, msg, {
        "counts": counts,
        "observed_shares": shares,
        "expected_split": dict(config.expected_split),
        "chi2": stat,
        "p_value": p,
        "log_mixture_lr": log_m,
        "method": config.srm_method,
    })


def run(assignments: pd.DataFrame, config: ExperimentConfig) -> tuple[list[CheckResult], pd.DataFrame]:
    th = config.thresholds
    out: list[CheckResult] = []
    units, crossover = unit_table(assignments)
    units = units[units["variant"].isin(config.variants)]

    # 2.1 SRM on assigned units
    counts = units["variant"].value_counts().reindex(config.variants, fill_value=0).astype(int).to_dict()
    out.append(srm_result("2.1", "srm_assignment", counts, config, "assigned units"))

    # 2.2 units in more than one variant
    share = len(crossover) / max(len(units), 1)
    status = CheckStatus.FAIL if share > th.max_crossover_share else (
        CheckStatus.WARN if len(crossover) else CheckStatus.PASS
    )
    out.append(CheckResult(
        "2.2", "crossover_units", status, C,
        f"Units assigned to more than one variant: {len(crossover)} ({share:.4%}). "
        "Usually an unstable unit id (re-hashed after login, new device) or a salt change mid-test.",
        {"crossover_units": int(len(crossover)), "share": share},
    ))

    # 2.3 assignment time inside the experiment window (informational)
    early = int((units["assigned_at"] < config.analysis_start).sum())
    late = int((units["assigned_at"] > config.analysis_end).sum())
    status = CheckStatus.WARN if (early or late) else CheckStatus.PASS
    out.append(CheckResult(
        "2.3", "assignment_window", status, CheckKind.DIAGNOSTIC,
        f"Units assigned before the window: {early}; after the window: {late}.",
        {"before_window": early, "after_window": late},
    ))

    units = units.assign(is_crossover=units.index.isin(crossover))
    return out, units


def daily_srm_path(units: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Cumulative SRM statistics by assignment day, for monitoring dashboards."""
    df = units.reset_index()
    df["day"] = df["assigned_at"].dt.floor("D")
    daily = df.pivot_table(index="day", columns="variant", values="unit_id", aggfunc="count", fill_value=0)
    daily = daily.reindex(columns=config.variants, fill_value=0).cumsum()
    rows = []
    for day, row in daily.iterrows():
        counts = row.astype(int).to_dict()
        _, p = srm_chi2(counts, config.expected_split)
        _, log_m = mixture_srm_alarm(counts, config.expected_split, config.thresholds.srm_alpha)
        rows.append({"day": day, "n": int(sum(counts.values())), "p_value": p, "log_mixture_lr": log_m})
    return pd.DataFrame(rows)

