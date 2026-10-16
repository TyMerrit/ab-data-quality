"""Step 0: validating the platform itself.

Per-experiment checks assume the machinery underneath is sound. These checks
validate that machinery, once and after every change to it:

0.1 assignment hashing   uniform buckets, deterministic, independent across
                         experiments (different salts)
0.2 A/A replay           push hundreds of random splits of historical data
                         through the real analysis code; the false positive
                         rate must match alpha and p-values must be uniform

A single live A/A test is one draw from the null. It can expose a gross bug,
but it cannot tell you whether the analysis pipeline is calibrated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from abdq.results import CheckKind, CheckResult, CheckStatus
from abdq.stats import binomial_band

C = CheckKind.CONFIRMATORY


# --- 0.1 assignment hashing ------------------------------------------------

def hash_bucket(unit_ids: Sequence, salt: str, n_buckets: int = 10_000) -> np.ndarray:
    """Deterministic bucket in [0, n_buckets) from sha256(salt:unit_id)."""
    out = np.empty(len(unit_ids), dtype=np.int64)
    for i, uid in enumerate(unit_ids):
        h = hashlib.sha256(f"{salt}:{uid}".encode()).digest()
        out[i] = int.from_bytes(h[:8], "big") % n_buckets
    return out


def assign_variants(unit_ids: Sequence, salt: str, split: dict[str, float], n_buckets: int = 10_000) -> np.ndarray:
    """Reference assignment function: bucket -> variant by cumulative split."""
    buckets = hash_bucket(unit_ids, salt, n_buckets)
    edges = np.cumsum([round(s * n_buckets) for s in split.values()])
    return np.array(list(split))[np.searchsorted(edges, buckets, side="right")]


AssignFn = Callable[[Sequence, str], np.ndarray]


def check_hash_uniformity(assign: AssignFn, split: dict[str, float], n_ids: int = 1_000_000,
                          salt: str = "exp-a", alpha: float = 1e-3) -> CheckResult:
    ids = [f"u{i}" for i in range(n_ids)]
    v = assign(ids, salt)
    counts = pd.Series(v).value_counts().reindex(list(split), fill_value=0)
    expected = n_ids * np.array(list(split.values()))
    stat = float(((counts.to_numpy() - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(stat, df=len(split) - 1))
    status = CheckStatus.FAIL if p < alpha else CheckStatus.PASS
    return CheckResult("0.1a", "hash_uniformity", status, C,
                       f"{n_ids:,} synthetic ids -> shares {(counts / n_ids).round(5).to_dict()}; p = {p:.2e}.",
                       {"counts": counts.to_dict(), "p_value": p})


def check_hash_determinism(assign: AssignFn, n_ids: int = 100_000, salt: str = "exp-a", seed: int = 0) -> CheckResult:
    ids = np.array([f"u{i}" for i in range(n_ids)])
    first = assign(list(ids), salt)
    perm = np.random.default_rng(seed).permutation(n_ids)
    second = np.empty_like(first)
    second[perm] = assign(list(ids[perm]), salt)
    changed = int((first != second).sum())
    status = CheckStatus.FAIL if changed else CheckStatus.PASS
    return CheckResult("0.1b", "hash_determinism", status, C,
                       f"Ids assigned differently on a second, shuffled pass: {changed}.",
                       {"changed": changed})


def check_hash_independence(assign: AssignFn, split: dict[str, float], n_ids: int = 500_000,
                            salts: tuple[str, str] = ("exp-a", "exp-b"), alpha: float = 1e-3) -> CheckResult:
    """Variants in two concurrent experiments must be independent of each other."""
    ids = [f"u{i}" for i in range(n_ids)]
    a = assign(ids, salts[0])
    b = assign(ids, salts[1])
    table = pd.crosstab(a, b)
    _, p, _, _ = stats.chi2_contingency(table.to_numpy(), correction=False)
    status = CheckStatus.FAIL if p < alpha else CheckStatus.PASS
    msg = (f"Cross-tab of variants in two experiments: p = {p:.2e}." +
           (" Assignments are correlated: the experiments are not independent (shared or missing salt?)."
            if status is CheckStatus.FAIL else ""))
    return CheckResult("0.1c", "hash_independence", status, C, msg,
                       {"crosstab": table.to_dict(), "p_value": float(p)})


# --- 0.2 A/A replay -----------------------------------------------------------

AnalysisFn = Callable[[pd.DataFrame, pd.Series], float]
"""(events, unit -> is_treatment) -> p-value of the treatment comparison."""


@dataclass
class AAReplayResult:
    p_values: np.ndarray
    alpha: float
    false_positive_rate: float
    band: tuple[float, float]
    ks_p_value: float
    check: CheckResult = field(repr=False)


def aa_replay(
    events: pd.DataFrame,
    analysis: AnalysisFn,
    unit_col: str = "unit_id",
    n_replays: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    name: str = "aa_replay",
) -> AAReplayResult:
    """Randomly split historical units 50/50 many times and run the real analysis.

    There is no true effect, so p-values should be Uniform(0, 1): the share
    below alpha should sit inside the binomial band around alpha, and a KS
    test against the uniform should not reject.
    """
    rng = np.random.default_rng(seed)
    units = pd.Index(events[unit_col].unique())
    pvals = np.empty(n_replays)
    for i in range(n_replays):
        is_treat = pd.Series(rng.random(len(units)) < 0.5, index=units)
        pvals[i] = analysis(events, is_treat)
    fpr = float((pvals < alpha).mean())
    band = binomial_band(n_replays, alpha)
    ks_p = float(stats.kstest(pvals, "uniform").pvalue)
    ok = band[0] <= fpr <= band[1] and ks_p >= 1e-3
    status = CheckStatus.PASS if ok else CheckStatus.FAIL
    msg = (f"{n_replays} A/A replays: false positive rate {fpr:.1%} (expected {alpha:.0%}, 99% band "
           f"{band[0]:.1%}-{band[1]:.1%}); KS uniformity p = {ks_p:.2e}.")
    if not ok:
        msg += " The analysis pipeline is miscalibrated."
    check = CheckResult("0.2", name, status, C, msg, {"fpr": fpr, "band": band, "ks_p_value": ks_p})
    return AAReplayResult(pvals, alpha, fpr, band, ks_p, check)


# --- reference analyses (one right, one wrong) -------------------------------

def unit_level_ttest(events: pd.DataFrame, is_treat: pd.Series, value: str = "value") -> float:
    """Aggregate to the randomization unit first, then compare units. Correct."""
    per_unit = events.groupby("unit_id")[value].sum().reindex(is_treat.index, fill_value=0.0)
    t = per_unit[is_treat.to_numpy()]
    c = per_unit[~is_treat.to_numpy()]
    return float(stats.ttest_ind(t, c, equal_var=False).pvalue)


def event_level_ttest(events: pd.DataFrame, is_treat: pd.Series, value: str = "value") -> float:
    """Treat every event as an independent observation. Wrong when units have
    several correlated events (pseudoreplication): the standard error is too small."""
    flag = is_treat.reindex(events["unit_id"]).to_numpy()
    return float(stats.ttest_ind(events[value][flag], events[value][~flag], equal_var=False).pvalue)
