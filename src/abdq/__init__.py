"""ab-data-quality: is your experiment data fit for inference?

Quick start:

    from abdq import ExperimentConfig, ExperimentData, validate_experiment
    report = validate_experiment(data, config)
    print(report.status, report.action)
    print(report.to_markdown())
"""

from abdq.config import ExperimentConfig, Thresholds
from abdq.data import ExperimentData, SchemaError
from abdq.pipeline import validate_experiment
from abdq.report import ValidationReport
from abdq.results import CheckKind, CheckResult, CheckStatus, ExperimentStatus

__all__ = [
    "CheckKind",
    "CheckResult",
    "CheckStatus",
    "ExperimentConfig",
    "ExperimentData",
    "ExperimentStatus",
    "SchemaError",
    "Thresholds",
    "ValidationReport",
    "validate_experiment",
]

__version__ = "0.1.0"
