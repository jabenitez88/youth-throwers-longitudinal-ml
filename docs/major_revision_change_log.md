# Major-revision change log

## Temporal predictor correction

The original future-regression feature sets included `delta_days`, the elapsed
time to the subsequently observed next or best record. Because this quantity is
not known at the index assessment, it has been removed from all corrected models.

For gradient boosting with current score, context, implements, and physical tests:

| Outcome | Original R2 | Corrected R2 | Original MAE | Corrected MAE | Original RMSE | Corrected RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Next observation | 0.616 | 0.608 | 43.48 | 44.90 | 70.49 | 71.27 |
| Maximum within 730 days | 0.681 | 0.616 | 44.94 | 51.31 | 65.23 | 71.88 |

The complete same-fold audit for every affected feature set and algorithm is in
`results/delta_days_removal_impact.csv`.

## Incremental value of physical tests

All incremental comparisons now hold the algorithm, observations, folds, and
preprocessing constant. Uncertainty comes from 2,000 paired bootstrap replicates
of out-of-fold predictions, clustered by athlete for repeated-record regression.

- Next-observation regression: physical-test increments were close to zero for
  all algorithms; every interval included zero.
- Horizon-maximum regression: gradient boosting showed Delta R2 = 0.040
  (95% interval 0.012 to 0.068) and an RMSE reduction of 3.79 points
  (1.16 to 6.45). Ridge and random-forest intervals included zero.
- Classification: ROC-AUC increments were small and all intervals included zero.
  Some PR-AUC or Brier improvements appeared for individual algorithm/outcome
  combinations, so the evidence is characterized as task- and algorithm-dependent.

## Added reviewer analyses

- PR-AUC, Brier score, calibration intercept, calibration slope, and calibration curves.
- Ten-times repeated five-fold cross-validation for the 41-case `future_ge950` outcome.
- Strict chronological validation with a 730-day gap between training index dates and holdout baseline dates.
- Follow-up observation-count and follow-up-span summaries and restricted-cohort sensitivities.
- Improvement thresholds of 25, 50, and 75 points plus continuous-change regression.
- Calendar-period sensitivity summaries.
- Participant characteristics, medicine-ball loads, and aggregate data-quality audit.

## Future-maximum follow-up sensitivity

The continuous `future_max_points` analysis was repeated after requiring at least two
or at least three observed future records within each index instance's 730-day window.
The restricted cohorts contained 737 instances from 185 athletes and 524 instances
from 128 athletes, respectively. Gradient-boosting R2 remained 0.500 and 0.526, but
the paired physical-test increment was no longer present: Delta R2 = -0.015 (95%
interval -0.062 to 0.024) and -0.007 (-0.046 to 0.027). The manuscript now describes
the full-cohort increment as not robust to follow-up-exposure restrictions.

## Unresolved source-data confirmation

The spreadsheet does not contain the exact competition date for `PUNTOS`. The
source publication describes it as a season personal best. The authors must confirm
that each score was known on or before the corresponding index assessment before
describing score-anchored models as fully prospective. The repository retains
no-current-score feature sets as a sensitivity analysis.
