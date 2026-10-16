"""Experiment configuration and pipeline thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds. Defaults are deliberately strict.

    Every confirmatory statistical check runs on every experiment, so the
    per-check alpha has to be small to keep the pipeline-wide false block
    rate low (see `experiments/validate_the_validator.py`).
    """

    srm_alpha: float = 1e-3
    """p-value threshold for SRM (fixed-horizon chi-square)."""
    structural_alpha: float = 1e-3
    """p-value threshold for other confirmatory tests (exposure rate, balance, ...)."""
    max_crossover_share: float = 1e-3
    """Max share of units that appear in more than one variant."""
    max_orphan_share: float = 1e-3
    """Max share of exposures / outcomes that cannot be linked to an assignment."""
    max_variant_mismatch_share: float = 1e-3
    """Max share of exposures logged with a different variant than assigned."""
    max_bad_timestamp_share: float = 1e-3
    """Max share of rows with impossible timestamps (exposure before assignment, ...)."""
    max_completeness_gap: float = 0.01
    """Max allowed gap between observed and reference outcome counts (1 = 100%)."""
    min_identity_effect: float = 0.02
    """Min relative difference in IDs-per-unit between variants to flag (on top of the test)."""


@dataclass(frozen=True)
class ExperimentConfig:
    """What the experiment was supposed to look like.

    expected_split
        Variant name -> designed traffic share. Shares must sum to 1.
    control
        Name of the control variant.
    analysis_start, analysis_end
        Event-time window the result will be computed on.
    as_of
        When the data was pulled (processing-time cutoff).
    max_expected_lag
        Upper bound on event_time -> processing_time lag for this pipeline,
        known from history. If None, it is estimated from the data (and the
        estimate is right-censored, so the maturity check becomes a warning).
    critical_segments
        Segment columns whose levels are checked as confirmatory (e.g. the
        platform column, if platform-specific logging bugs are a known risk).
    diagnostic_segments
        Segment columns used only to localize a failure.
    """

    expected_split: dict[str, float]
    control: str
    analysis_start: pd.Timestamp
    analysis_end: pd.Timestamp
    as_of: pd.Timestamp
    max_expected_lag: pd.Timedelta | None = None
    critical_segments: tuple[str, ...] = ()
    diagnostic_segments: tuple[str, ...] | None = None
    """None means: use every segment column found on the assignments table."""
    srm_method: str = "fixed"
    """'fixed': chi-square at srm_alpha (fine for a handful of daily looks with a
    strict alpha). 'anytime': Dirichlet-multinomial mixture likelihood ratio,
    valid under continuous monitoring."""
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self) -> None:
        total = sum(self.expected_split.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"expected_split must sum to 1, got {total}.")
        if self.control not in self.expected_split:
            raise ValueError(f"control {self.control!r} is not in expected_split.")
        if self.analysis_end < self.analysis_start:
            raise ValueError("analysis_end must not be earlier than analysis_start.")
        if self.srm_method not in ("fixed", "anytime"):
            raise ValueError("srm_method must be 'fixed' or 'anytime'.")

    @property
    def variants(self) -> list[str]:
        return list(self.expected_split)
