"""Step 8: from check results to an experiment status.

    any confirmatory FAIL          -> untrustworthy  (block the result)
    else any confirmatory PRELIM.  -> preliminary    (wait, do not ship yet)
    else                           -> trustworthy    (publish)

WARN and diagnostic results are reported but never change the status.
"""

from __future__ import annotations

from abdq.results import CheckKind, CheckResult, CheckStatus, ExperimentStatus


def decide(checks: list[CheckResult]) -> tuple[ExperimentStatus, list[str]]:
    confirmatory = [c for c in checks if c.kind is CheckKind.CONFIRMATORY]
    failed = [c for c in confirmatory if c.status is CheckStatus.FAIL]
    if failed:
        return ExperimentStatus.UNTRUSTWORTHY, [f"{c.step} {c.name}" for c in failed]
    prelim = [c for c in confirmatory if c.status is CheckStatus.PRELIMINARY]
    if prelim:
        return ExperimentStatus.PRELIMINARY, [f"{c.step} {c.name}" for c in prelim]
    return ExperimentStatus.TRUSTWORTHY, []


ACTIONS = {
    ExperimentStatus.TRUSTWORTHY: "publish",
    ExperimentStatus.PRELIMINARY: "wait for the data to mature; do not ship on this read",
    ExperimentStatus.UNTRUSTWORTHY: "block the result until the root cause is found",
}
