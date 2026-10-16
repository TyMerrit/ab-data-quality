"""Step 5: identity.

Assignment happens on one id, logging often on another (device, cookie,
session). If one variant fragments users into more devices than the other,
per-device counts, joins and even "unique users" drift apart between arms,
and the contrast partly measures the identity layer instead of the treatment.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from abdq.config import ExperimentConfig
from abdq.results import CheckKind, CheckResult, CheckStatus

C = CheckKind.CONFIRMATORY


def run(units: pd.DataFrame, identities: pd.DataFrame | None, config: ExperimentConfig) -> list[CheckResult]:
    if identities is None:
        return [CheckResult("5.1", "ids_per_unit", CheckStatus.SKIPPED, C,
                            "No identity table (unit_id, device_id) provided.")]
    th = config.thresholds
    out: list[CheckResult] = []
    ids = identities.merge(units[["variant"]], left_on="unit_id", right_index=True, how="inner")

    # 5.1 devices per unit, compared across variants
    per_unit = ids.groupby("unit_id").agg(n_ids=("device_id", "nunique"), variant=("variant", "first"))
    per_unit["bucket"] = per_unit["n_ids"].clip(upper=4)
    table = pd.crosstab(per_unit["variant"], per_unit["bucket"]).reindex(config.variants, fill_value=0)
    table = table.loc[:, table.sum(axis=0) > 0]
    if table.shape[1] > 1:
        _, p, _, _ = stats.chi2_contingency(table.to_numpy())
    else:
        p = 1.0
    means = per_unit.groupby("variant")["n_ids"].mean().reindex(config.variants)
    ctrl = means[config.control]
    rel = {v: float(means[v] / ctrl - 1) for v in config.variants if v != config.control}
    worst = max((abs(r) for r in rel.values()), default=0.0)
    failed = p < th.structural_alpha and worst > th.min_identity_effect
    status = CheckStatus.FAIL if failed else CheckStatus.PASS
    out.append(CheckResult(
        "5.1", "ids_per_unit", status, C,
        f"Mean device ids per unit by variant: {means.round(3).to_dict()}; p = {p:.2e}."
        + (" One variant fragments identities more: per-device metrics and joins are not comparable."
           if failed else ""),
        {"mean_ids_per_unit": means.to_dict(), "relative_diff_vs_control": rel, "p_value": float(p)},
    ))

    # 5.2 device ids shared by units in different variants
    v_per_dev = ids.groupby("device_id")["variant"].nunique()
    shared = int((v_per_dev > 1).sum())
    share = shared / max(len(v_per_dev), 1)
    status = CheckStatus.FAIL if share > th.max_crossover_share else (CheckStatus.WARN if shared else CheckStatus.PASS)
    out.append(CheckResult(
        "5.2", "shared_ids_across_variants", status, C,
        f"Device ids linked to units in different variants: {shared} ({share:.4%}). The same "
        "physical device sees both experiences.",
        {"shared_ids": shared, "share": share},
    ))
    return out

