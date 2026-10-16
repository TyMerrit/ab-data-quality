"""Result types shared by every check in the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CheckStatus(str, Enum):
    """Outcome of a single check.

    PASS         nothing wrong.
    WARN         worth a look, never changes the experiment status.
    PRELIMINARY  data is not final yet (e.g. outcomes still maturing).
    FAIL         the data is not fit for inference.
    SKIPPED      the inputs this check needs were not provided.
    """

    PASS = "pass"
    WARN = "warn"
    PRELIMINARY = "preliminary"
    FAIL = "fail"
    SKIPPED = "skipped"


class CheckKind(str, Enum):
    """Confirmatory checks decide the experiment status.

    Diagnostic checks only help to localize a problem. They are reported but
    never block a result on their own: with enough segments, something always
    looks significant.
    """

    CONFIRMATORY = "confirmatory"
    DIAGNOSTIC = "diagnostic"


class ExperimentStatus(str, Enum):
    TRUSTWORTHY = "trustworthy"
    PRELIMINARY = "preliminary"
    UNTRUSTWORTHY = "untrustworthy"


@dataclass
class CheckResult:
    step: str
    """Pipeline step id, e.g. '2.1'."""
    name: str
    status: CheckStatus
    kind: CheckKind
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "status": self.status.value,
            "kind": self.kind.value,
            "message": self.message,
            "details": self.details,
        }
