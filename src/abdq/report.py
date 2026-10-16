"""Validation report: markdown for humans, JSON for machines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from abdq.config import ExperimentConfig
from abdq.decision import ACTIONS
from abdq.results import CheckResult, CheckStatus, ExperimentStatus

ICONS = {
    CheckStatus.PASS: "PASS",
    CheckStatus.WARN: "WARN",
    CheckStatus.PRELIMINARY: "PRELIM",
    CheckStatus.FAIL: "FAIL",
    CheckStatus.SKIPPED: "skip",
}


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return str(x)
    return x


@dataclass
class ValidationReport:
    status: ExperimentStatus
    reasons: list[str]
    checks: list[CheckResult]
    config: ExperimentConfig

    @property
    def action(self) -> str:
        return ACTIONS[self.status]

    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    def get(self, name: str) -> CheckResult:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "status": self.status.value,
            "action": self.action,
            "reasons": self.reasons,
            "checks": [c.to_dict() for c in self.checks],
        })

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), indent=2, **kwargs)

    def to_markdown(self) -> str:
        lines = [
            f"## Experiment status: **{self.status.value.upper()}**",
            "",
            f"Action: {self.action}.",
        ]
        if self.reasons:
            lines += ["", "Blocking checks: " + ", ".join(self.reasons) + "."]
        lines += ["", "| Step | Check | Kind | Result | Details |", "|---|---|---|---|---|"]
        for c in self.checks:
            msg = c.message.replace("|", "\\|")
            lines.append(f"| {c.step} | {c.name} | {c.kind.value} | {ICONS[c.status]} | {msg} |")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()
