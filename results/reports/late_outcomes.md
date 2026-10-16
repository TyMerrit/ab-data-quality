# Scenario: late_outcomes

Injected defect: slow pipeline, data pulled 6h after the window.

## Experiment status: **PRELIMINARY**

Action: wait for the data to mature; do not ship on this read.

Blocking checks: 4.2 maturity, 4.4 completeness_vs_reference.

| Step | Check | Kind | Result | Details |
|---|---|---|---|---|
| 1.1 | null_keys | confirmatory | PASS | Rows with null keys (worst table): 0.0000%. Dropped before further checks. |
| 1.2 | duplicates | diagnostic | PASS | Duplicate assignment rows: 0; duplicate outcome event_ids: 0. Removed. Duplicates inflate sums and variances; if they are concentrated in one variant, investigate the pipeline. |
| 1.3 | unknown_variants | confirmatory | PASS | All assigned variants are in the design. |
| 1.4 | timestamp_order | confirmatory | PASS | Outcome events processed before they happened: 0.0000%. |
| 2.1 | srm_assignment | confirmatory | PASS | SRM on assigned units: observed shares control=0.5013, treatment=0.4987; p = 6.03e-01 (ok under chi-square p < 0.001). |
| 2.2 | crossover_units | confirmatory | PASS | Units assigned to more than one variant: 0 (0.0000%). Usually an unstable unit id (re-hashed after login, new device) or a salt change mid-test. |
| 2.3 | assignment_window | diagnostic | PASS | Units assigned before the window: 0; after the window: 0. |
| 3.1 | orphan_exposures | confirmatory | PASS | Exposures with no matching assignment: 0 (0.0000%). Typically an id that changes between the assignment service and the client. |
| 3.2 | variant_mismatch | confirmatory | PASS | Exposures logged with a different variant than assigned: 0 (0.0000%). |
| 3.3 | exposure_before_assignment | confirmatory | PASS | Exposures timestamped before the unit was assigned: 0 (0.0000%). |
| 3.4 | exposure_rate_by_variant | confirmatory | PASS | Exposed / assigned by variant: control=0.7028, treatment=0.6980; p = 2.89e-01. Unless the treatment itself changes who reaches the exposure point, these rates should match. |
| 3.5 | srm_exposed | confirmatory | PASS | SRM on exposed units: observed shares control=0.5030, treatment=0.4970; p = 3.10e-01 (ok under chi-square p < 0.001). |
| 3.6 | critical_segment[platform] | confirmatory | PASS | Exposure rate consistent across variants within every level of platform. |
| 4.1 | orphan_outcomes | confirmatory | PASS | Outcome events from units with no assignment: 0 (0.0000%). Usually an id mismatch between the event pipeline and the assignment service. |
| 4.2 | maturity | confirmatory | PRELIM | Data pulled 0 days 06:00:00 after the window closed, but lags reach 2 days 00:00:00 (configured). Late events are still arriving: treat the result as preliminary. |
| 4.3 | lag_asymmetry | diagnostic | PASS | Lag distributions are similar across variants. |
| 4.4 | completeness_vs_reference | confirmatory | PRELIM | Outcome counts are below the reference, but the data is not mature yet. |
| 5.1 | ids_per_unit | confirmatory | PASS | Mean device ids per unit by variant: {'control': 1.205, 'treatment': 1.212}; p = 9.24e-02. |
| 5.2 | shared_ids_across_variants | confirmatory | PASS | Device ids linked to units in different variants: 0 (0.0000%). The same physical device sees both experiences. |
| 6.1 | covariate_balance | confirmatory | PASS | Pre-period covariates balanced (2 covariates, Bonferroni alpha 5.0e-04). |
| 7 | segment[platform] | diagnostic | PASS | Levels of platform with SRM or exposure-rate imbalance: none. |
