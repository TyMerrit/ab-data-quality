"""Step 3: the exposure layer.

Assignment can be perfectly balanced while the exposed population is not.
The classic cause is exposure logging that loses events on one client or in
one variant. If the analysis runs on exposed units, that loss changes who is
being compared, and the composition shift shows up as a "treatment effect".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from abdq.checks.assignment import srm_result
from abdq.config import ExperimentConfig
from abdq.results import CheckKind, CheckResult, CheckStatus
from abdq.stats import rate_by_group_test

C = CheckKind.CONFIRMATORY


def run(units: pd.DataFrame, exposures: pd.DataFrame, config: ExperimentConfig) -> tuple[list[CheckResult], pd.DataFrame]:
    th = config.thresholds
    out: list[CheckResult] = []

    exp = exposures.merge(
        units[["variant", "assigned_at"]].rename(columns={"variant": "assigned_variant"}),
        left_on="unit_id", right_index=True, how="left",
    )

    # 3.1 exposures with no assignment
    orphans = int(exp["assigned_variant"].isna().sum())
    share = orphans / max(len(exp), 1)
    status = CheckStatus.FAIL if share > th.max_orphan_share else (CheckStatus.WARN if orphans else CheckStatus.PASS)
    out.append(CheckResult(
        "3.1", "orphan_exposures", status, C,
        f"Exposures with no matching assignment: {orphans} ({share:.4%}). Typically an id that "
        "changes between the assignment service and the client.",
        {"orphan_exposures": orphans, "share": share},
    ))
    exp = exp[exp["assigned_variant"].notna()]

    # 3.2 exposure logged under a different variant than assigned
    mismatch = int((exp["variant"] != exp["assigned_variant"]).sum())
    share = mismatch / max(len(exp), 1)
    status = CheckStatus.FAIL if share > th.max_variant_mismatch_share else (
        CheckStatus.WARN if mismatch else CheckStatus.PASS
    )
    out.append(CheckResult(
        "3.2", "variant_mismatch", status, C,
        f"Exposures logged with a different variant than assigned: {mismatch} ({share:.4%}).",
        {"mismatched_exposures": mismatch, "share": share},
    ))

    # 3.3 exposure before assignment
    early = int((exp["exposed_at"] < exp["assigned_at"]).sum())
    share = early / max(len(exp), 1)
    status = CheckStatus.FAIL if share > th.max_bad_timestamp_share else (
        CheckStatus.WARN if early else CheckStatus.PASS
    )
    out.append(CheckResult(
        "3.3", "exposure_before_assignment", status, C,
        f"Exposures timestamped before the unit was assigned: {early} ({share:.4%}).",
        {"rows": early, "share": share},
    ))

    exposed_ids = pd.Index(exp["unit_id"].unique())
    u = units.assign(exposed=units.index.isin(exposed_ids))

    # 3.4 exposure rate by variant
    g = u.groupby("variant")["exposed"].agg(["sum", "count"]).reindex(config.variants, fill_value=0)
    p = rate_by_group_test(g["sum"].to_numpy(), g["count"].to_numpy())
    rates = (g["sum"] / g["count"].replace(0, np.nan)).to_dict()
    status = CheckStatus.FAIL if p < th.structural_alpha else CheckStatus.PASS
    rates_txt = ", ".join(f"{v}={r:.4f}" for v, r in rates.items())
    out.append(CheckResult(
        "3.4", "exposure_rate_by_variant", status, C,
        f"Exposed / assigned by variant: {rates_txt}; p = {p:.2e}. Unless the treatment itself "
        "changes who reaches the exposure point, these rates should match.",
        {"exposure_rate": rates, "p_value": p},
    ))

    # 3.5 SRM on the exposed population
    counts = u[u["exposed"]]["variant"].value_counts().reindex(config.variants, fill_value=0).astype(int).to_dict()
    out.append(srm_result("3.5", "srm_exposed", counts, config, "exposed units"))

    # 3.6 critical segments: exposure rate by variant within each level (Bonferroni)
    for col in config.critical_segments:
        if col not in u.columns:
            out.append(CheckResult("3.6", f"critical_segment[{col}]", CheckStatus.SKIPPED, C,
                                   f"Segment column {col!r} not on assignments."))
            continue
        levels = list(u[col].dropna().unique())
        alpha = th.structural_alpha / max(len(levels), 1)
        per_level = {}
        failed = []
        for lvl in levels:
            s = u[u[col] == lvl].groupby("variant")["exposed"].agg(["sum", "count"]).reindex(
                config.variants, fill_value=0)
            pl = rate_by_group_test(s["sum"].to_numpy(), s["count"].to_numpy())
            per_level[str(lvl)] = {
                "p_value": pl,
                "exposure_rate": (s["sum"] / s["count"].replace(0, np.nan)).to_dict(),
            }
            if pl < alpha:
                failed.append(str(lvl))
        status = CheckStatus.FAIL if failed else CheckStatus.PASS
        out.append(CheckResult(
            "3.6", f"critical_segment[{col}]", status, C,
            (f"Exposure rate differs between variants within {col} = {failed} "
             f"(Bonferroni alpha {alpha:.1e}). Loss concentrated in one segment is far easier "
             "to see here than in the overall SRM." if failed else
             f"Exposure rate consistent across variants within every level of {col}."),
            {"levels": per_level, "alpha_per_level": alpha},
        ))

    return out, u
