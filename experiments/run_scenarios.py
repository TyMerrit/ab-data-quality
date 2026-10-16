"""Run the full pipeline on one healthy experiment and one experiment per defect.

Writes results/scenarios.csv (one row per scenario) and
results/reports/<scenario>.md (the full validation report).

Run:  python experiments/run_scenarios.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style  # noqa: E402
from abdq import validate_experiment  # noqa: E402
from abdq import simulate as sim  # noqa: E402


def scenarios(seed: int = 0):
    spec = sim.SimSpec(seed=seed)
    data, config = sim.generate_experiment(spec)
    yield "healthy", "none", "trustworthy", data, config
    d, c = sim.generate_experiment(replace(spec, realized_split={"control": 0.51, "treatment": 0.49}))
    yield "srm", "assignment split 51/49 instead of 50/50", "untrustworthy", d, c
    yield "crossover", "1% of units also assigned to the other variant", "untrustworthy", \
        sim.inject_crossover(data, 0.01), config
    yield "exposure_loss_ios", "5% of treatment exposures on iOS not logged", "untrustworthy", \
        sim.inject_exposure_loss(data, 0.05), config
    d, c = sim.generate_experiment(sim.late_data_spec(spec))
    yield "late_outcomes", "slow pipeline, data pulled 6h after the window", "preliminary", d, c
    yield "outcome_loss", "5% of treatment outcome events lost for good", "untrustworthy", \
        sim.inject_outcome_loss(data, 0.05), config
    yield "identity_split", "10% of treatment users get an extra device id", "untrustworthy", \
        sim.inject_identity_split(data, 0.10), config
    d, c = sim.generate_experiment(replace(spec, assignment_bias=0.10))
    yield "covariate_imbalance", "heavy users leak into treatment, counts stay 50/50", "untrustworthy", d, c


def main() -> None:
    out_dir = _style.RESULTS / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, defect, expected, data, config in scenarios():
        report = validate_experiment(data, config)
        (out_dir / f"{name}.md").write_text(f"# Scenario: {name}\n\nInjected defect: {defect}.\n\n"
                                            + report.to_markdown() + "\n")
        srm = report.get("srm_assignment").details["p_value"]
        rows.append({
            "scenario": name, "injected_defect": defect, "expected_status": expected,
            "status": report.status.value, "blocking_checks": "; ".join(report.reasons),
            "srm_assignment_p": srm, "correct": report.status.value == expected,
        })
        print(f"{name:22s} expected {expected:14s} got {report.status.value:14s} {report.reasons}")
    pd.DataFrame(rows).to_csv(_style.RESULTS / "scenarios.csv", index=False)


if __name__ == "__main__":
    main()
