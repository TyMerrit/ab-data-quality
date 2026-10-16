"""The per-experiment validation pipeline.

    validate_experiment(data, config) -> ValidationReport

Steps (each a module under abdq.checks):

    1  contracts    readable data: keys, duplicates, variants, timestamps
    2  assignment   SRM, units in more than one variant
    3  exposure     orphans, variant mismatch, exposure rate, SRM on exposed,
                    critical segments
    4  outcomes     orphans, maturity, lag asymmetry, completeness vs reference
    5  identity     ids per unit by variant, ids shared across variants
    6  balance      pre-experiment covariates
    7  diagnostics  segment-level SRM / exposure rate (never blocking)
    8  decision     trustworthy / preliminary / untrustworthy

Platform-level checks (step 0: hashing, A/A replay) live in abdq.platform,
because they validate the platform once rather than every experiment.
"""

from __future__ import annotations

from abdq.checks import assignment, balance, contracts, diagnostics, exposure, identity, outcomes
from abdq.config import ExperimentConfig
from abdq.data import ExperimentData
from abdq.decision import decide
from abdq.report import ValidationReport


def validate_experiment(data: ExperimentData, config: ExperimentConfig) -> ValidationReport:
    checks = []

    step1, clean = contracts.run(data, config)
    checks += step1

    step2, units = assignment.run(clean.assignments, config)
    checks += step2

    step3, units_exposed = exposure.run(units, clean.exposures, config)
    checks += step3

    checks += outcomes.run(units, clean.outcomes, config, clean.reference_counts)
    checks += identity.run(units, clean.identities, config)
    checks += balance.run(units, clean.covariates, config)
    checks += diagnostics.run(units_exposed, config)

    status, reasons = decide(checks)
    return ValidationReport(status=status, reasons=reasons, checks=checks, config=config)
