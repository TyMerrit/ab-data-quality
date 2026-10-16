"""Input container and table schemas.

The pipeline works on three event tables, one per layer of an experiment:

assignments   who was put in which variant          (unit_id, variant, assigned_at)
exposures     who actually saw the variant           (unit_id, variant, exposed_at)
outcomes      what they did                           (event_id, unit_id, event_time,
                                                       processing_time, value)

`unit_id` is always the randomization unit. Any other columns on
assignments (e.g. `platform`, `country`) are treated as segment columns.

Optional inputs unlock extra checks:

covariates        unit_id + numeric pre-experiment columns   -> step 6 (balance)
identities        unit_id, device_id                         -> step 5 (identity)
reference_counts  variant -> outcome event count from an
                  independent source of truth (e.g. the orders
                  database)                                  -> step 4 (loss)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ASSIGNMENT_COLUMNS = ("unit_id", "variant", "assigned_at")
EXPOSURE_COLUMNS = ("unit_id", "variant", "exposed_at")
OUTCOME_COLUMNS = ("event_id", "unit_id", "event_time", "processing_time", "value")
IDENTITY_COLUMNS = ("unit_id", "device_id")


class SchemaError(ValueError):
    """Raised when a required column is missing. The pipeline cannot run."""


@dataclass
class ExperimentData:
    assignments: pd.DataFrame
    exposures: pd.DataFrame
    outcomes: pd.DataFrame
    covariates: pd.DataFrame | None = None
    identities: pd.DataFrame | None = None
    reference_counts: dict[str, int] | None = None

    def copy(self) -> "ExperimentData":
        return ExperimentData(
            assignments=self.assignments.copy(),
            exposures=self.exposures.copy(),
            outcomes=self.outcomes.copy(),
            covariates=None if self.covariates is None else self.covariates.copy(),
            identities=None if self.identities is None else self.identities.copy(),
            reference_counts=None if self.reference_counts is None else dict(self.reference_counts),
        )


def require_columns(df: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{table}: missing required columns {missing}.")


def segment_columns(assignments: pd.DataFrame) -> list[str]:
    return [c for c in assignments.columns if c not in ASSIGNMENT_COLUMNS and c != "user_id"]
