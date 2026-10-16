from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from abdq import CheckStatus, ExperimentStatus, SchemaError, validate_experiment  # noqa: E402
from abdq import simulate as sim  # noqa: E402
from abdq.platform import (  # noqa: E402
    aa_replay,
    assign_variants,
    check_hash_determinism,
    check_hash_independence,
    check_hash_uniformity,
    event_level_ttest,
    unit_level_ttest,
)
from abdq.stats import log_mixture_srm, srm_chi2  # noqa: E402
from run_scenarios import scenarios  # noqa: E402

SPLIT = {"control": 0.5, "treatment": 0.5}


@pytest.mark.parametrize("seed", [0, 1])
def test_scenarios_get_expected_status(seed):
    for name, _defect, expected, data, config in scenarios(seed=seed):
        report = validate_experiment(data, config)
        assert report.status.value == expected, (name, report.reasons)


def test_report_serializes():
    data, config = sim.generate_experiment(sim.SimSpec(n_units=5_000))
    report = validate_experiment(data, config)
    assert '"status"' in report.to_json()
    assert "Experiment status" in report.to_markdown()


def test_missing_column_raises():
    data, config = sim.generate_experiment(sim.SimSpec(n_units=2_000))
    data.exposures = data.exposures.drop(columns="exposed_at")
    with pytest.raises(SchemaError):
        validate_experiment(data, config)


def test_optional_inputs_are_skipped_not_failed():
    data, config = sim.generate_experiment(sim.SimSpec(n_units=5_000))
    data.covariates = None
    data.identities = None
    data.reference_counts = None
    report = validate_experiment(data, config)
    assert report.status is ExperimentStatus.TRUSTWORTHY
    assert report.get("covariate_balance").status is CheckStatus.SKIPPED
    assert report.get("completeness_vs_reference").status is CheckStatus.SKIPPED


def test_segment_diagnostics_never_block():
    data, config = sim.generate_experiment(sim.SimSpec(n_units=20_000))
    report = validate_experiment(data, config)
    for c in report.checks:
        if c.step == "7":
            assert c.kind.value == "diagnostic"


def test_anytime_srm_method():
    spec = sim.SimSpec(seed=3, realized_split={"control": 0.52, "treatment": 0.48})
    data, config = sim.generate_experiment(spec)
    config = replace(config, srm_method="anytime")
    report = validate_experiment(data, config)
    assert report.get("srm_assignment").status is CheckStatus.FAIL


def test_srm_chi2_matches_scipy():
    from scipy import stats
    stat, p = srm_chi2({"control": 5100, "treatment": 4900}, SPLIT)
    ref = stats.chisquare([5100, 4900])
    assert np.isclose(stat, ref.statistic) and np.isclose(p, ref.pvalue)


def test_mixture_srm_is_anytime_valid_under_null():
    """Checked after every 100 units for 20k units, the alarm rate stays below alpha."""
    rng = np.random.default_rng(0)
    runs, steps, per = 4_000, 200, 100
    t = rng.binomial(per, 0.5, size=(runs, steps)).cumsum(axis=1).astype(float)
    n = per * np.arange(1, steps + 1, dtype=float)
    log_m = log_mixture_srm(np.stack([n - t, t], axis=-1), np.array([0.5, 0.5]), 10_000.0)
    alarm_rate = (log_m >= np.log(1 / 0.05)).any(axis=1).mean()
    assert alarm_rate <= 0.05


def test_hash_checks_catch_known_bugs():
    good = lambda ids, salt: assign_variants(ids, salt, SPLIT)  # noqa: E731
    shared = lambda ids, salt: assign_variants(ids, "global", SPLIT)  # noqa: E731
    assert check_hash_uniformity(good, SPLIT, n_ids=50_000).status is CheckStatus.PASS
    assert check_hash_determinism(good, n_ids=5_000).status is CheckStatus.PASS
    assert check_hash_independence(good, SPLIT, n_ids=50_000).status is CheckStatus.PASS
    assert check_hash_independence(shared, SPLIT, n_ids=50_000).status is CheckStatus.FAIL


def test_aa_replay_flags_pseudoreplication():
    rng = np.random.default_rng(1)
    users = rng.lognormal(1, 1, size=1_000)
    k = 1 + rng.poisson(6, size=1_000)
    uid = np.repeat(np.arange(1_000), k)
    events = pd.DataFrame({"unit_id": uid, "value": users[uid] * rng.lognormal(0, 0.5, size=len(uid))})
    good = aa_replay(events, unit_level_ttest, n_replays=200, seed=2)
    bad = aa_replay(events, event_level_ttest, n_replays=200, seed=2)
    assert good.check.status is CheckStatus.PASS
    assert bad.check.status is CheckStatus.FAIL
    assert bad.false_positive_rate > 0.2
