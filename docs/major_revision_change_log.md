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
- Exact `future_ge950` validation-fold event counts (9, 8, 8, 8, and 8 positives).
- Applied conversion of a 40-45-point absolute error into representative throwing distances.
- Participant characteristics, medicine-ball loads, and aggregate data-quality audit.

## Future-maximum follow-up sensitivity

The continuous `future_max_points` analysis was repeated after requiring at least two
or at least three observed future records within each index instance's 730-day window.
The restricted cohorts contained 737 instances from 185 athletes and 524 instances
from 128 athletes, respectively. Pooled OOF Gradient-Boosting R2 with physical tests
was 0.560 and 0.561, respectively, but
the paired physical-test increment was no longer present: Delta R2 = -0.015 (95%
interval -0.062 to 0.024) and -0.007 (-0.046 to 0.027). The manuscript now describes
the full-cohort increment as not robust to follow-up-exposure restrictions.

## Calendar-period sensitivity

Full-cohort OOF predictions were summarized by index period rather than refitting
three period-specific models. For `future_max_points`, paired Gradient-Boosting
Delta R2 values after adding physical tests were 0.074 (95% interval 0.012 to
0.151) in 1997-2005, 0.100 (0.041 to 0.172) in 2006-2012, and 0.019 (-0.021 to
0.064) in 2013-2020. For `future_ge900`, paired Random-Forest Delta ROC-AUC values
were 0.001 (-0.031 to 0.031), 0.000 (-0.037 to 0.040), and -0.012 (-0.044 to
0.015), respectively. The classification conclusion was stable, whereas the
regression increment was heterogeneous across historical periods.

## Applied interpretation of prediction error

Using exact entries from the 2017 revised IAAF scoring tables at a representative
800-point level, a 40-45-point difference corresponds to approximately 0.64-0.73 m
in women's shot put and 2.19-2.47 m in men's discus. The manuscript states that
these are local examples rather than fixed conversions because the scoring tables
are progressive and event- and sex-specific.

## Unresolved source-data confirmation

The spreadsheet does not contain the exact competition date for `PUNTOS`. The
source publication describes it as a season personal best. The authors must confirm
that each score was known on or before the corresponding index assessment before
describing score-anchored models as fully prospective. The repository retains
no-current-score feature sets as a sensitivity analysis.

## Second-review technical clarifications

Aggregate baseline cross-tabs were added for the 900- and 950-point outcomes. Table
3 now identifies the best algorithm for every predictor set and outcome. The strict
chronological table now states the exact common `g1_current` predictor set.

The follow-up sensitivity table previously combined mean fold R2 values with a
paired Delta R2 calculated from pooled out-of-fold predictions. The revised table
uses pooled OOF R2, MAE, and RMSE throughout, preserving the paired Delta R2 and its
athlete-cluster bootstrap interval. No model was refitted for this correction.
