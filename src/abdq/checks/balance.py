"""Step 6: pre-experiment covariate balance.

SRM checks counts. It is blind to a split that is balanced in size but not in
composition, e.g. an assignment service that leaks heavy users into treatment
while sending the same number of light users to control. Pre-period
covariates catch that.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from abdq.config import ExperimentConfig
from abdq.results import CheckKind, CheckResult, CheckStatus
from abdq.stats import standardized_mean_difference

C = CheckKind.CONFIRMATORY


def run(units: pd.DataFrame, covariates: pd.DataFrame | None, config: ExperimentConfig) -> list[CheckResult]:
    if covariates is None:
        return [CheckResult("6.1", "covariate_balance", CheckStatus.SKIPPED, C,
                            "No pre-experiment covariates provided.")]
    th = config.thresholds
    df = covariates.merge(units[["variant"]], left_on="unit_id", right_index=True, how="inner")
    cols = [c for c in covariates.columns if c != "unit_id" and pd.api.types.is_numeric_dtype(covariates[c])]
    others = [v for v in config.variants if v != config.control]
    n_tests = max(len(cols) * len(others), 1)
    alpha = th.structural_alpha / n_tests
    rows = {}
    failed = []
    ctrl = df[df["variant"] == config.control]
    for col in cols:
        for v in others:
            tv = df.loc[df["variant"] == v, col].dropna().to_numpy()
            cv = ctrl[col].dropna().to_numpy()
            p = float(stats.ttest_ind(tv, cv, equal_var=False).pvalue)
            smd = standardized_mean_difference(tv, cv)
            rows[f"{col}|{v}"] = {"smd": smd, "p_value": p}
            if p < alpha:
                failed.append(f"{col} ({v}: SMD {smd:+.3f})")
    status = CheckStatus.FAIL if failed else CheckStatus.PASS
    msg = (f"Pre-period imbalance in {failed} (Bonferroni alpha {alpha:.1e}). Randomization should "
           "make pre-period differences pure noise; this is not." if failed else
           f"Pre-period covariates balanced ({len(cols)} covariates, Bonferroni alpha {alpha:.1e}).")
    return [CheckResult("6.1", "covariate_balance", status, C, msg, {"tests": rows, "alpha": alpha})]
