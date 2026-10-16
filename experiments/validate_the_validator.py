"""Validate the validator.

A data-quality pipeline is itself a decision procedure, so it has error rates:

- false block rate: healthy experiments marked untrustworthy;
- detection rate: broken experiments that get caught.

We run the full pipeline on many healthy synthetic experiments and on many
copies of each defect with fresh seeds.

Run:  python experiments/validate_the_validator.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style  # noqa: E402
from abdq import validate_experiment  # noqa: E402
from run_scenarios import scenarios  # noqa: E402

N_SEEDS = 200


def main() -> None:
    rows = []
    for seed in range(N_SEEDS):
        for name, _defect, expected, data, config in scenarios(seed=seed):
            report = validate_experiment(data, config)
            rows.append({"seed": seed, "scenario": name, "expected": expected,
                         "status": report.status.value, "reasons": "; ".join(report.reasons)})
        if (seed + 1) % 25 == 0:
            print(f"{seed + 1}/{N_SEEDS} seeds done")
    df = pd.DataFrame(rows)
    df.to_csv(_style.RESULTS / "validate_the_validator_runs.csv", index=False)
    summary = (df.assign(correct=df["status"] == df["expected"])
                 .groupby(["scenario", "expected"], sort=False)["correct"].mean()
                 .rename("share_with_expected_status").reset_index())
    summary.to_csv(_style.RESULTS / "validate_the_validator.csv", index=False)
    print(summary.to_string(index=False))
    healthy = df[df["scenario"] == "healthy"]
    blocked = healthy[healthy["status"] != "trustworthy"]
    print(f"\nHealthy experiments not marked trustworthy: {len(blocked)}/{len(healthy)}")
    print("Checks responsible:", Counter(r for rs in blocked["reasons"] for r in rs.split("; ") if r))


if __name__ == "__main__":
    main()
