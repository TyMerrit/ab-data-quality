"""Step 7: segment diagnostics.

Run SRM and exposure-rate checks inside every level of every diagnostic
segment column. These are for *localizing* a problem that a confirmatory
check already found ("the SRM lives in iOS 17.2"). They never block a result
by themselves: with 50 segments, a few will cross any threshold by chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abdq.config import ExperimentConfig
from abdq.data import segment_columns
from abdq.results import CheckKind, CheckResult, CheckStatus
from abdq.stats import rate_by_group_test, srm_chi2

D = CheckKind.DIAGNOSTIC


def run(units_exposed: pd.DataFrame, config: ExperimentConfig) -> list[CheckResult]:
    cols = config.diagnostic_segments
    if cols is None:
        cols = tuple(c for c in segment_columns(units_exposed.reset_index())
                     if c not in ("exposed", "is_crossover"))
    out: list[CheckResult] = []
    alpha = config.thresholds.srm_alpha
    for col in cols:
        if col not in units_exposed.columns:
            continue
        findings = {}
        for lvl, g in units_exposed.groupby(col):
            counts = g["variant"].value_counts().reindex(config.variants, fill_value=0).astype(int).to_dict()
            _, p_srm = srm_chi2(counts, config.expected_split)
            e = g.groupby("variant")["exposed"].agg(["sum", "count"]).reindex(config.variants, fill_value=0)
            p_exp = rate_by_group_test(e["sum"].to_numpy(), e["count"].to_numpy())
            findings[str(lvl)] = {
                "n_units": int(len(g)),
                "srm_p_value": p_srm,
                "exposure_rate_p_value": p_exp,
                "exposure_rate": (e["sum"] / e["count"].replace(0, np.nan)).to_dict(),
            }
        flagged = [k for k, f in findings.items() if min(f["srm_p_value"], f["exposure_rate_p_value"]) < alpha]
        status = CheckStatus.WARN if flagged else CheckStatus.PASS
        out.append(CheckResult(
            "7", f"segment[{col}]", status, D,
            f"Levels of {col} with SRM or exposure-rate imbalance: {flagged or 'none'}.",
            findings,
        ))
    return out
