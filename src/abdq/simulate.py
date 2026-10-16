"""Synthetic experiments with known defects.

`generate_experiment` produces a healthy experiment (assignment ->
exposure -> outcome, plus identities, pre-period covariates and reference
counts). Defect injectors then break exactly one layer, so every check can
be tested against a case where the right answer is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from abdq.config import ExperimentConfig
from abdq.data import ExperimentData


@dataclass(frozen=True)
class SimSpec:
    n_units: int = 40_000
    expected_split: dict[str, float] = field(default_factory=lambda: {"control": 0.5, "treatment": 0.5})
    realized_split: dict[str, float] | None = None
    """If set, units are actually assigned with these shares (an SRM)."""
    assignment_bias: float = 0.0
    """>0: heavier pre-period users are more likely to land in treatment, while the
    expected share stays at the design (balanced counts, unbalanced composition)."""
    platforms: dict[str, float] = field(default_factory=lambda: {"ios": 0.4, "android": 0.4, "web": 0.2})
    conversion: dict[str, float] = field(default_factory=lambda: {"ios": 0.08, "android": 0.05, "web": 0.04})
    exposure_rate: float = 0.7
    """Share of assigned units that actually reach the changed surface."""
    effect: float = 0.0
    """Relative lift in conversion for truly exposed treatment units."""
    days: int = 14
    start: pd.Timestamp = pd.Timestamp("2026-09-01")
    lag_mean_hours: float = 1.0
    max_expected_lag: pd.Timedelta = pd.Timedelta(days=2)
    as_of_delay: pd.Timedelta = pd.Timedelta(days=3)
    seed: int = 0


def _config(spec: SimSpec) -> ExperimentConfig:
    end = spec.start + pd.Timedelta(days=spec.days)
    return ExperimentConfig(
        expected_split=dict(spec.expected_split),
        control="control",
        analysis_start=spec.start,
        analysis_end=end,
        as_of=end + spec.as_of_delay,
        max_expected_lag=spec.max_expected_lag,
        critical_segments=("platform",),
    )


def generate_experiment(spec: SimSpec = SimSpec()) -> tuple[ExperimentData, ExperimentConfig]:
    rng = np.random.default_rng(spec.seed)
    config = _config(spec)
    n = spec.n_units
    unit_id = np.arange(1, n + 1)

    platforms = np.array(list(spec.platforms))
    platform = rng.choice(platforms, size=n, p=list(spec.platforms.values()))

    # pre-period covariates, correlated with platform conversion
    base_conv = np.array([spec.conversion[p] for p in platform])
    pre_orders = rng.poisson(base_conv * 20 * rng.gamma(2.0, 0.5, size=n))
    pre_sessions = rng.poisson(5 + 30 * base_conv * rng.gamma(2.0, 0.5, size=n))

    # assignment
    variants = list(spec.expected_split)
    shares = np.array(list((spec.realized_split or spec.expected_split).values()))
    if spec.assignment_bias and len(variants) == 2:
        rank = pd.Series(pre_orders + rng.random(n) * 0.01).rank(pct=True).to_numpy() - 0.5
        p_treat = np.clip(shares[1] + spec.assignment_bias * rank, 0, 1)
        variant = np.where(rng.random(n) < p_treat, variants[1], variants[0])
    else:
        variant = rng.choice(variants, size=n, p=shares)
    window = pd.Timedelta(days=spec.days).total_seconds()
    assigned_at = spec.start + pd.to_timedelta(rng.random(n) * window * 0.9, unit="s")

    assignments = pd.DataFrame({
        "unit_id": unit_id, "variant": variant, "assigned_at": assigned_at, "platform": platform,
    })

    # true exposure (whether the unit reached the changed surface)
    truly_exposed = rng.random(n) < spec.exposure_rate
    exposed_at = assigned_at + pd.to_timedelta(rng.exponential(1800, size=n), unit="s")
    exposures = pd.DataFrame({
        "unit_id": unit_id[truly_exposed], "variant": variant[truly_exposed],
        "exposed_at": exposed_at[truly_exposed], "platform": platform[truly_exposed],
    })

    # outcomes: conversions happen for everyone; the effect only for exposed treatment units
    p_conv = base_conv * (1 + spec.effect * ((variant != variants[0]) & truly_exposed))
    converts = rng.random(n) < p_conv
    n_orders = np.where(converts, 1 + rng.poisson(0.3, size=n), 0)
    idx = np.repeat(np.arange(n), n_orders)
    start_t = np.where(truly_exposed, exposed_at, assigned_at)[idx]
    event_time = pd.DatetimeIndex(start_t) + pd.to_timedelta(rng.random(len(idx)) * 4 * 86400, unit="s")
    lag = pd.to_timedelta(rng.exponential(spec.lag_mean_hours * 3600, size=len(idx)), unit="s")
    outcomes = pd.DataFrame({
        "event_id": np.arange(1, len(idx) + 1),
        "unit_id": unit_id[idx],
        "event_time": event_time,
        "processing_time": event_time + lag,
        "value": np.round(rng.lognormal(3.5, 0.8, size=len(idx)), 2),
    })
    in_window = (outcomes["event_time"] >= config.analysis_start) & (outcomes["event_time"] <= config.analysis_end)
    reference = (
        pd.Series(variant[idx][in_window.to_numpy()]).value_counts().reindex(variants, fill_value=0).astype(int).to_dict()
    )
    outcomes = outcomes[outcomes["processing_time"] <= config.as_of].reset_index(drop=True)

    # identities: most units have one device id, some have more
    k = rng.choice([1, 2, 3], size=n, p=[0.82, 0.15, 0.03])
    identities = pd.DataFrame({
        "unit_id": np.repeat(unit_id, k),
        "device_id": np.arange(1, k.sum() + 1),
    })

    covariates = pd.DataFrame({"unit_id": unit_id, "pre_orders": pre_orders, "pre_sessions": pre_sessions})

    data = ExperimentData(
        assignments=assignments, exposures=exposures, outcomes=outcomes,
        covariates=covariates, identities=identities, reference_counts=reference,
    )
    return data, config


# --- defect injectors -------------------------------------------------------

def inject_crossover(data: ExperimentData, rate: float, seed: int = 1) -> ExperimentData:
    """A share of units gets a second assignment to another variant (unstable id, salt change)."""
    rng = np.random.default_rng(seed)
    d = data.copy()
    a = d.assignments
    pick = a.sample(frac=rate, random_state=rng.integers(1 << 31))
    variants = sorted(a["variant"].unique())
    flipped = pick.assign(
        variant=[variants[(variants.index(v) + 1) % len(variants)] for v in pick["variant"]],
        assigned_at=pick["assigned_at"] + pd.Timedelta(hours=6),
    )
    d.assignments = pd.concat([a, flipped], ignore_index=True)
    return d


def inject_exposure_loss(
    data: ExperimentData, rate: float, variant: str = "treatment",
    segment: tuple[str, str] | None = ("platform", "ios"), seed: int = 2,
) -> ExperimentData:
    """Drop a share of exposure rows in one variant (optionally one segment). Assignment untouched."""
    rng = np.random.default_rng(seed)
    d = data.copy()
    e = d.exposures
    eligible = e["variant"] == variant
    if segment is not None:
        eligible &= e[segment[0]] == segment[1]
    drop = eligible & (rng.random(len(e)) < rate)
    d.exposures = e.loc[~drop].reset_index(drop=True)
    return d


def inject_outcome_loss(data: ExperimentData, rate: float, variant: str = "treatment", seed: int = 3) -> ExperimentData:
    """Permanently lose a share of outcome events in one variant. Reference counts untouched."""
    rng = np.random.default_rng(seed)
    d = data.copy()
    o = d.outcomes.merge(d.assignments.drop_duplicates("unit_id")[["unit_id", "variant"]], on="unit_id")
    drop = (o["variant"] == variant) & (rng.random(len(o)) < rate)
    d.outcomes = o.loc[~drop].drop(columns="variant").reset_index(drop=True)
    return d


def inject_identity_split(data: ExperimentData, rate: float, variant: str = "treatment", seed: int = 4) -> ExperimentData:
    """A share of units in one variant gets an extra device id (e.g. a new SDK resets ids)."""
    rng = np.random.default_rng(seed)
    d = data.copy()
    units = d.assignments.drop_duplicates("unit_id")
    pick = units.loc[(units["variant"] == variant) & (rng.random(len(units)) < rate), "unit_id"]
    start = int(d.identities["device_id"].max()) + 1
    extra = pd.DataFrame({"unit_id": pick.to_numpy(), "device_id": np.arange(start, start + len(pick))})
    d.identities = pd.concat([d.identities, extra], ignore_index=True)
    return d


def late_data_spec(spec: SimSpec = SimSpec()) -> SimSpec:
    """Same experiment, but pulled 6 hours after the window closed with slow processing."""
    return replace(spec, lag_mean_hours=10.0, as_of_delay=pd.Timedelta(hours=6))
