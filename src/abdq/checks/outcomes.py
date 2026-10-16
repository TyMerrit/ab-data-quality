"""Step 4: the outcome layer.

Two failure modes that look identical on day one and need opposite responses:

immaturity  events arrive late but do arrive. Completeness converges to 1.
            Response: wait. Status: preliminary.
loss        events never arrive. Completeness plateaus below 1.
            Response: block until fixed. Status: untrustworthy.

Without an independent source of truth, permanent loss that hits one variant
harder is indistinguishable from a treatment effect. That is why the loss
check needs `reference_counts` (e.g. orders from the transactional database).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from abdq.config import ExperimentConfig
from abdq.results import CheckKind, CheckResult, CheckStatus

C = CheckKind.CONFIRMATORY
D = CheckKind.DIAGNOSTIC


def run(
    units: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: ExperimentConfig,
    reference_counts: dict[str, int] | None,
) -> list[CheckResult]:
    th = config.thresholds
    out: list[CheckResult] = []

    o = outcomes.merge(units[["variant"]], left_on="unit_id", right_index=True, how="left")

    # 4.1 outcome events from units that were never assigned
    orphans = int(o["variant"].isna().sum())
    share = orphans / max(len(o), 1)
    status = CheckStatus.FAIL if share > th.max_orphan_share else (CheckStatus.WARN if orphans else CheckStatus.PASS)
    out.append(CheckResult(
        "4.1", "orphan_outcomes", status, C,
        f"Outcome events from units with no assignment: {orphans} ({share:.4%}). Usually an id "
        "mismatch between the event pipeline and the assignment service.",
        {"orphan_events": orphans, "share": share},
    ))
    o = o[o["variant"].notna()]
    in_window = o[(o["event_time"] >= config.analysis_start) & (o["event_time"] <= config.analysis_end)]

    # 4.2 maturity
    lag = (o["processing_time"] - o["event_time"])
    buffer = config.as_of - config.analysis_end
    if config.max_expected_lag is not None:
        bound = config.max_expected_lag
        source = "configured"
    else:
        bound = lag.quantile(0.99) if len(lag) else pd.Timedelta(0)
        source = "estimated from data (right-censored, likely an underestimate)"
    mature = buffer >= bound
    over_bound = float((lag > bound).mean()) if len(lag) else 0.0
    if not mature:
        status = CheckStatus.PRELIMINARY
        msg = (f"Data pulled {buffer} after the window closed, but lags reach {bound} ({source}). "
               "Late events are still arriving: treat the result as preliminary.")
    elif config.max_expected_lag is None:
        status = CheckStatus.WARN
        msg = (f"Buffer {buffer} covers the estimated p99 lag {bound}, but the estimate is censored. "
               "Set max_expected_lag from pipeline history to make this check confirmatory.")
    elif over_bound > 0.01:
        status = CheckStatus.WARN
        msg = f"{over_bound:.2%} of events exceed the configured max lag {bound}: the bound may be stale."
    else:
        status = CheckStatus.PASS
        msg = f"Buffer {buffer} covers the max expected lag {bound}."
    out.append(CheckResult("4.2", "maturity", status, C, msg, {
        "buffer": str(buffer), "lag_bound": str(bound), "lag_bound_source": source,
        "share_over_bound": over_bound,
    }))

    # 4.3 lag asymmetry between variants (diagnostic)
    lag_s = lag.dt.total_seconds()
    ctrl = lag_s[o["variant"] == config.control]
    per_variant = {}
    flagged = []
    for v in config.variants:
        if v == config.control:
            continue
        tv = lag_s[o["variant"] == v]
        if len(tv) < 20 or len(ctrl) < 20:
            continue
        ks = stats.ks_2samp(tv, ctrl)
        p95_t, p95_c = np.quantile(tv, 0.95), np.quantile(ctrl, 0.95)
        rel = (p95_t - p95_c) / p95_c if p95_c else 0.0
        per_variant[v] = {"ks_p_value": float(ks.pvalue), "p95_lag_rel_diff": float(rel)}
        if ks.pvalue < th.structural_alpha and abs(rel) > 0.10:
            flagged.append(v)
    status = CheckStatus.WARN if flagged else CheckStatus.PASS
    out.append(CheckResult(
        "4.3", "lag_asymmetry", status, D,
        (f"Lag distribution differs from control for {flagged}. Early reads compare data at "
         "different stages of maturity." if flagged else "Lag distributions are similar across variants."),
        per_variant,
    ))

    # 4.4 completeness against an independent source of truth
    if reference_counts is None:
        out.append(CheckResult(
            "4.4", "completeness_vs_reference", CheckStatus.SKIPPED, C,
            "No reference counts provided. Permanent outcome loss cannot be told apart from a "
            "treatment effect without an independent source of truth.",
        ))
        return out
    observed = in_window["variant"].value_counts().reindex(config.variants, fill_value=0)
    completeness = {v: (observed[v] / reference_counts[v] if reference_counts.get(v) else float("nan"))
                    for v in config.variants}
    worst = min(completeness.values())
    spread = max(completeness.values()) - worst
    if worst >= 1 - th.max_completeness_gap and spread <= th.max_completeness_gap:
        status, msg = CheckStatus.PASS, "Outcome counts match the reference in every variant."
    elif not mature:
        status = CheckStatus.PRELIMINARY
        msg = "Outcome counts are below the reference, but the data is not mature yet."
    else:
        status = CheckStatus.FAIL
        msg = ("Outcome counts stay below the reference after the maturity window: events are "
               "lost, not late." + (" The loss is asymmetric across variants." if spread > th.max_completeness_gap else ""))
    out.append(CheckResult("4.4", "completeness_vs_reference", status, C, msg, {
        "completeness": completeness, "observed": observed.astype(int).to_dict(),
        "reference": dict(reference_counts),
    }))
    return out
