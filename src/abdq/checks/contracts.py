"""Step 1: data contracts.

Syntactic and basic semantic checks. They make the tables readable; they say
nothing yet about whether the data is fit for inference. The step also
returns cleaned tables (null keys dropped, exact duplicates removed, one
first-exposure row per unit) that every later step works on.
"""

from __future__ import annotations

import pandas as pd

from abdq.config import ExperimentConfig
from abdq.data import (
    ASSIGNMENT_COLUMNS,
    EXPOSURE_COLUMNS,
    IDENTITY_COLUMNS,
    OUTCOME_COLUMNS,
    ExperimentData,
    require_columns,
)
from abdq.results import CheckKind, CheckResult, CheckStatus

C = CheckKind.CONFIRMATORY
D = CheckKind.DIAGNOSTIC


def run(data: ExperimentData, config: ExperimentConfig) -> tuple[list[CheckResult], ExperimentData]:
    require_columns(data.assignments, ASSIGNMENT_COLUMNS, "assignments")
    require_columns(data.exposures, EXPOSURE_COLUMNS, "exposures")
    require_columns(data.outcomes, OUTCOME_COLUMNS, "outcomes")
    if data.identities is not None:
        require_columns(data.identities, IDENTITY_COLUMNS, "identities")
    if data.covariates is not None:
        require_columns(data.covariates, ("unit_id",), "covariates")

    th = config.thresholds
    out: list[CheckResult] = []
    clean = data.copy()

    # 1.1 null keys -------------------------------------------------------
    null_counts = {}
    for name, df, keys in (
        ("assignments", clean.assignments, ["unit_id", "variant"]),
        ("exposures", clean.exposures, ["unit_id", "variant"]),
        ("outcomes", clean.outcomes, ["event_id", "unit_id"]),
    ):
        mask = df[keys].isna().any(axis=1)
        null_counts[name] = (int(mask.sum()), len(df))
        setattr(clean, name, df.loc[~mask])
    worst = max((k / n if n else 0.0) for k, n in null_counts.values())
    status = CheckStatus.FAIL if worst > th.max_bad_timestamp_share else (
        CheckStatus.WARN if worst > 0 else CheckStatus.PASS
    )
    out.append(CheckResult(
        "1.1", "null_keys", status, C,
        f"Rows with null keys (worst table): {worst:.4%}. Dropped before further checks.",
        {t: {"null_rows": k, "rows": n} for t, (k, n) in null_counts.items()},
    ))

    # 1.2 exact duplicates ------------------------------------------------
    dup_assign = int(clean.assignments.duplicated(["unit_id", "variant"]).sum())
    dup_events = int(clean.outcomes.duplicated(["event_id"]).sum())
    clean.assignments = clean.assignments.drop_duplicates(["unit_id", "variant"])
    clean.outcomes = clean.outcomes.drop_duplicates(["event_id"])
    status = CheckStatus.WARN if (dup_assign or dup_events) else CheckStatus.PASS
    out.append(CheckResult(
        "1.2", "duplicates", status, D,
        f"Duplicate assignment rows: {dup_assign}; duplicate outcome event_ids: {dup_events}. "
        "Removed. Duplicates inflate sums and variances; if they are concentrated in one "
        "variant, investigate the pipeline.",
        {"duplicate_assignment_rows": dup_assign, "duplicate_outcome_events": dup_events},
    ))

    # repeated exposures are normal (a user visits twice); keep the first one
    clean.exposures = (
        clean.exposures.sort_values("exposed_at").drop_duplicates(["unit_id", "variant"], keep="first")
    )

    # 1.3 unexpected variants ----------------------------------------------
    unknown = sorted(set(clean.assignments["variant"]) - set(config.variants))
    status = CheckStatus.FAIL if unknown else CheckStatus.PASS
    out.append(CheckResult(
        "1.3", "unknown_variants", status, C,
        f"Variants not in the design: {unknown}" if unknown else "All assigned variants are in the design.",
        {"unknown_variants": unknown},
    ))

    # 1.4 timestamp sanity ---------------------------------------------------
    o = clean.outcomes
    bad_lag = int((o["processing_time"] < o["event_time"]).sum())
    share = bad_lag / len(o) if len(o) else 0.0
    status = CheckStatus.FAIL if share > th.max_bad_timestamp_share else (
        CheckStatus.WARN if bad_lag else CheckStatus.PASS
    )
    out.append(CheckResult(
        "1.4", "timestamp_order", status, C,
        f"Outcome events processed before they happened: {share:.4%}.",
        {"bad_rows": bad_lag, "rows": len(o)},
    ))

    if data.identities is not None:
        clean.identities = clean.identities.dropna(subset=list(IDENTITY_COLUMNS)).drop_duplicates()

    return out, clean
