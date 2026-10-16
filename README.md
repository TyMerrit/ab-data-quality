# ab-data-quality

**Is your experiment data fit for inference?** A step-by-step validation pipeline for A/B test data, plus the simulations behind it.

Data can pass every schema check and every business rule and still be useless for comparing variants. Picture purchase events that are perfectly well-formed but get dropped slightly more often in one variant. Every row is valid, and the comparison is broken. This repo checks the property that actually matters: whether the assignment → exposure → outcome chain still supports a causal comparison.

It is the companion code to the post *How to validate your randomization (and your A/B platform)* on [tymerrit.github.io/blog](https://tymerrit.github.io/blog/), and level 1 of the [experimentation hierarchy of needs](https://tymerrit.github.io/blog/experimentation-hierarchy-of-needs/).

## Quick start

```bash
git clone https://github.com/tymerrit/ab-data-quality.git
cd ab-data-quality
pip install -r requirements.txt   # numpy, pandas, scipy, matplotlib, jupyter, pytest
pip install -e .                  # makes `import abdq` work from anywhere
```

A full walkthrough, from one experiment to the simulations, is in [`walkthrough.ipynb`](walkthrough.ipynb).

```python
from abdq import validate_experiment
from abdq.simulate import SimSpec, generate_experiment, inject_exposure_loss

data, config = generate_experiment(SimSpec(n_units=40_000))
data = inject_exposure_loss(data, rate=0.05, variant="treatment", segment=("platform", "ios"))

report = validate_experiment(data, config)
print(report.status)          # ExperimentStatus.UNTRUSTWORTHY
print(report.reasons)         # ['3.4 exposure_rate_by_variant', '3.6 critical_segment[platform]']
print(report.to_markdown())   # full step-by-step report
```

In that example the overall SRM on exposed units passes (p ≈ 0.007, above the 0.001 threshold). The exposure-rate checks catch the loss.

## Inputs

Three event tables (pandas DataFrames), one per layer of the experiment:

| Table | Required columns | Layer |
|---|---|---|
| `assignments` | `unit_id`, `variant`, `assigned_at` (+ any segment columns, e.g. `platform`) | who was put in which variant |
| `exposures` | `unit_id`, `variant`, `exposed_at` | who actually saw the variant |
| `outcomes` | `event_id`, `unit_id`, `event_time`, `processing_time`, `value` | what they did |

`unit_id` is always the randomization unit. Optional inputs unlock more checks:

| Optional input | Unlocks |
|---|---|
| `covariates`: `unit_id` + numeric pre-period columns | step 6, covariate balance |
| `identities`: `unit_id`, `device_id` | step 5, identity checks |
| `reference_counts`: `{variant: count}` of outcome events from an independent source of truth (e.g. the orders DB) | step 4.4, telling *lost* from *late* |

The `ExperimentConfig` holds the design: expected split, control, analysis window, `as_of`, the maximum expected pipeline lag, and which segment columns are critical.

## The pipeline

| Step | Checks | Kind |
|---|---|---|
| **0. Platform** (`abdq.platform`, run once and after changes) | hash uniformity, determinism, independence across experiments; A/A replay with FPR band + KS uniformity | confirmatory |
| **1. Contracts** | null keys, duplicates, unknown variants, timestamp order | mostly confirmatory |
| **2. Assignment** | SRM (fixed or anytime-valid), units in more than one variant, assignment window | confirmatory |
| **3. Exposure** | orphan exposures, variant mismatch, exposure before assignment, exposure rate by variant, SRM on exposed, exposure rate within critical segments | confirmatory |
| **4. Outcomes** | orphan outcomes, maturity, lag asymmetry, completeness vs reference | confirmatory (lag asymmetry: diagnostic) |
| **5. Identity** | ids per unit by variant, ids shared across variants | confirmatory |
| **6. Balance** | pre-period covariates, Welch t-test + SMD, Bonferroni | confirmatory |
| **7. Diagnostics** | SRM and exposure rate within every segment level | diagnostic only |
| **8. Decision** | `trustworthy` / `preliminary` / `untrustworthy` + reasons | |

Two design rules:

- **Confirmatory vs diagnostic.** Only confirmatory checks change the status. Segment-level SRM is diagnostic: it localizes a problem you already know about. With 50 segments, a few will cross any threshold by chance, and letting each of them block a result would be automated p-hacking in reverse.
- **Preliminary vs untrustworthy.** "The data isn't complete yet" and "the data will never be complete" call for opposite responses: wait, or block. Without an independent source of truth, permanent loss in one variant is indistinguishable from a treatment effect, which is why the loss check needs `reference_counts`.

## Using it on your own experiments

The order matters: each step assumes the ones before it passed. Run them like this.

### Once, and after every change to assignment or analysis code: validate the platform

Per-experiment checks can't catch a broken hashing function or a miscalibrated analysis pipeline, so validate those first.

```python
from abdq.platform import (
    aa_replay, check_hash_determinism, check_hash_independence, check_hash_uniformity,
)

split = {"control": 0.5, "treatment": 0.5}

def my_assign(unit_ids, salt):
    """Wrap your production assignment function: ids + experiment salt -> variant names."""
    ...

for check in (check_hash_uniformity(my_assign, split),          # 0.1a buckets match the split
              check_hash_determinism(my_assign),                # 0.1b same id -> same variant, every time
              check_hash_independence(my_assign, split)):       # 0.1c two experiments don't correlate
    print(check.status.value, check.message)

def my_analysis(events, is_treatment):
    """Your real readout code. events: historical rows with a unit_id column;
    is_treatment: boolean Series indexed by unit_id. Returns the p-value."""
    ...

result = aa_replay(historical_events, my_analysis, n_replays=500)    # 0.2 calibration
print(result.check.message)   # false positive rate must sit in the band around alpha
```

If any of these fail, stop: no per-experiment check downstream can be trusted until they pass.

### For every experiment

**1. Pull the three tables.** One row per assignment, one per exposure, one per outcome event. `unit_id` must be the randomization unit in all three. Keep timestamps in one timezone. Add segment columns (`platform`, `country`, `app_version`) to `assignments`: they power the critical-segment and diagnostic checks.

**2. Pull the optional inputs if you have them.** Each one unlocks a check that nothing else can replace:

- `covariates`: pre-experiment metrics per unit. Catches a split that is balanced in counts but not in composition, which SRM can't see.
- `identities`: `unit_id` to `device_id`. Catches one variant fragmenting users into more ids.
- `reference_counts`: outcome counts per variant from an independent system (orders DB, billing). Without it, permanent loss of outcome events is indistinguishable from a treatment effect.

**3. Describe the design.**

```python
import pandas as pd
from abdq import ExperimentConfig, ExperimentData, Thresholds, validate_experiment

config = ExperimentConfig(
    expected_split={"control": 0.5, "treatment": 0.5},
    control="control",
    analysis_start=pd.Timestamp("2026-09-01"),
    analysis_end=pd.Timestamp("2026-09-15"),
    as_of=pd.Timestamp("2026-09-18"),               # when the data was pulled
    max_expected_lag=pd.Timedelta(days=2),          # from your pipeline's history
    critical_segments=("platform",),                # known logging risks -> confirmatory
    srm_method="fixed",                             # "anytime" if you monitor continuously
    thresholds=Thresholds(),                        # defaults are strict on purpose
)

data = ExperimentData(assignments, exposures, outcomes,
                      covariates=covariates, identities=identities,
                      reference_counts={"control": 51_204, "treatment": 50_987})
```

**4. Run and act on the status.**

```python
report = validate_experiment(data, config)

report.status        # trustworthy / preliminary / untrustworthy
report.action        # publish / wait for the data to mature / block until the root cause is found
report.reasons       # the confirmatory checks behind the status, e.g. ['3.4 exposure_rate_by_variant']
report.to_markdown() # full report for the experiment page
report.to_json()     # for dashboards and alerting
```

**5. If the status is `untrustworthy`, investigate in pipeline order.** The lowest failing step is usually the cause, and later failures are its symptoms: an assignment bug (step 2) also skews exposures (step 3) and outcomes (step 4). Then read the step 7 diagnostics to localize it: which platform, country or app version carries the imbalance. Fix the root cause. Don't trim the bigger arm or reweight your way out of an SRM: that hides the evidence and keeps the bias.

**6. When to run it.** During the test, daily, as monitoring: SRM, crossover and exposure checks are meaningful from day one, and `preliminary` from step 4 is expected while data is still arriving. Before the readout, run once more with `as_of` at least `max_expected_lag` after `analysis_end`. Only a `trustworthy` result at that point should be read.

### Running a single step

Each step is a plain function, so you can wire them into your own orchestration (Airflow, dbt post-hooks, a notebook):

```python
from abdq.checks import assignment, balance, contracts, diagnostics, exposure, identity, outcomes
from abdq.decision import decide

checks, clean = contracts.run(data, config)                                   # step 1, returns cleaned tables
step2, units = assignment.run(clean.assignments, config)                      # step 2, returns one row per unit
step3, units_exposed = exposure.run(units, clean.exposures, config)           # step 3
step4 = outcomes.run(units, clean.outcomes, config, clean.reference_counts)  # step 4
step5 = identity.run(units, clean.identities, config)                         # step 5
step6 = balance.run(units, clean.covariates, config)                          # step 6
step7 = diagnostics.run(units_exposed, config)                                # step 7, never blocking
status, reasons = decide(checks + step2 + step3 + step4 + step5 + step6 + step7)  # step 8
```

Steps 3-7 need the `units` table from step 2, and every step expects the cleaned tables from step 1.

### Tuning thresholds

The defaults (`srm_alpha = structural_alpha = 0.001`, share thresholds of 0.1%) give about a 1% false block rate on healthy synthetic experiments (notebook section 6). Before changing them, rerun that section on your own historical experiments: lowering a threshold to catch more defects also blocks more healthy tests, and with ~15 confirmatory checks per experiment the false blocks add up.

## Reproduce

The simulations behind the post (selective exposure loss, SRM monitoring, A/A replay, the validator's own error rates) live in the notebook. Open [`walkthrough.ipynb`](walkthrough.ipynb) and run all cells (about 8 minutes, most of it in section 6). It writes figures to `figures/` and tables and per-scenario reports to `results/`. Tests:

```bash
pytest
```

## Limits

Everything here runs on synthetic data with known defects. The simulations show classes of failure and the trade-offs between checks, not production-calibrated error rates. Thresholds should be calibrated on your own pipeline's history and on past incidents. Interference between units (SUTVA violations) is a design problem, not a data-quality one, and is out of scope.

## References

- Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments*. Cambridge University Press.
- Fabijan, A., et al. (2019). Diagnosing Sample Ratio Mismatch in Online Controlled Experiments: A Taxonomy and Rules of Thumb for Practitioners. *KDD*.
- Lindon, M., & Malek, A. (2022). Anytime-Valid Inference for Multinomial Count Data. *NeurIPS*.
- Johari, R., Pekelis, L., & Walsh, D. (2017). Always Valid Inference: Continuous Monitoring of A/B Tests. *Operations Research*.
- Armitage, P., McPherson, C. K., & Rowe, B. C. (1969). Repeated significance tests on accumulating data. *JRSS A*.

Builds on my master's thesis (HSE University, 2026) on statistical data quality in A/B testing platforms.

## License

MIT
