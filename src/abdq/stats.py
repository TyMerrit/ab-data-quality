"""Statistical primitives used by the checks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats
from scipy.special import gammaln


def srm_chi2(counts: Mapping[str, int], expected_split: Mapping[str, float]) -> tuple[float, float]:
    """Pearson chi-square goodness-of-fit test of observed counts against the design.

    Returns (statistic, p-value). Counts must be counts of unique randomization
    units, not events.
    """
    variants = list(expected_split)
    observed = np.array([counts.get(v, 0) for v in variants], dtype=float)
    n = observed.sum()
    if n == 0:
        return 0.0, 1.0
    expected = n * np.array([expected_split[v] for v in variants], dtype=float)
    stat = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(stat, df=len(variants) - 1))
    return stat, p


def log_mixture_srm(
    counts: np.ndarray,
    expected_split: np.ndarray,
    concentration: float,
) -> np.ndarray:
    """Log of the Dirichlet-multinomial mixture likelihood ratio for SRM.

    H0: units fall into variants with probabilities `expected_split`.
    The alternative is a Dirichlet(concentration * expected_split) mixture
    over the simplex. Under H0 the ratio is a nonnegative martingale with
    mean 1, so by Ville's inequality

        P(sup_n M_n >= 1/alpha) <= alpha,

    i.e. the test can be evaluated after every new unit, every day, or at
    any data-dependent time, and the false alarm rate stays below alpha.
    This is the construction used by Lindon & Malek (2022).

    counts
        Array of shape (..., k): cumulative unit counts per variant. Leading
        dimensions are broadcast (useful for Monte Carlo).
    concentration
        Total prior mass. Larger values concentrate the prior around the
        design and give more power against small deviations, at the cost of
        power against large ones.
    """
    counts = np.asarray(counts, dtype=float)
    p0 = np.asarray(expected_split, dtype=float)
    a = concentration * p0
    n = counts.sum(axis=-1)
    log_marginal_alt = (
        gammaln(a.sum())
        - gammaln(a.sum() + n)
        + (gammaln(a + counts) - gammaln(a)).sum(axis=-1)
    )
    log_null = (counts * np.log(p0)).sum(axis=-1)
    return log_marginal_alt - log_null


def mixture_srm_alarm(
    counts: Mapping[str, int],
    expected_split: Mapping[str, float],
    alpha: float,
    concentration: float = 10_000.0,
) -> tuple[bool, float]:
    """Anytime-valid SRM check at a single point in time.

    Returns (alarm, log M_n). Because M_n is a test martingale, checking it
    at every look and alarming the first time it crosses 1/alpha controls
    the false alarm rate at alpha regardless of how often you look.
    """
    variants = list(expected_split)
    c = np.array([counts.get(v, 0) for v in variants], dtype=float)
    p0 = np.array([expected_split[v] for v in variants], dtype=float)
    log_m = float(log_mixture_srm(c, p0, concentration))
    return log_m >= np.log(1.0 / alpha), log_m


def rate_by_group_test(successes: np.ndarray, trials: np.ndarray) -> float:
    """Chi-square test of equal success rates across groups. Returns p-value."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    if (trials == 0).any() or successes.sum() == 0 or successes.sum() == trials.sum():
        return 1.0
    table = np.vstack([successes, trials - successes])
    _, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(p)


def standardized_mean_difference(x_treat: np.ndarray, x_ctrl: np.ndarray) -> float:
    """SMD = (mean_t - mean_c) / pooled sd. Scale-free covariate imbalance."""
    pooled = np.sqrt((np.var(x_treat, ddof=1) + np.var(x_ctrl, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(x_treat) - np.mean(x_ctrl)) / pooled)


def binomial_band(n: int, alpha: float, level: float = 0.99) -> tuple[float, float]:
    """Range of the observed false positive rate expected from n null replays."""
    lo = stats.binom.ppf((1 - level) / 2, n, alpha) / n
    hi = stats.binom.ppf(1 - (1 - level) / 2, n, alpha) / n
    return float(lo), float(hi)
