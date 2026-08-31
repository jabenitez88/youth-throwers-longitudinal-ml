# Draft response to reviewers

This document is a working draft. Page and line numbers must be inserted after the
revised Word manuscript is finalized. Text marked **AUTHOR CONFIRMATION REQUIRED**
must not be submitted without confirmation from the data owners.

## General comment 1: incremental value of physical tests

**Response.** We agree that the original comparison changed both the predictor set
and, in some rows, the selected algorithm. We have replaced this with paired
within-algorithm comparisons. For each outcome, the same observations, five folds,
preprocessing pipeline, and fixed hyperparameters were used for the model without
physical tests and the model with physical tests. Differences in R2, MAE, RMSE,
ROC-AUC, PR-AUC, and Brier score were calculated from paired out-of-fold predictions.
Uncertainty was estimated using 2,000 percentile bootstrap replicates, clustered by
athlete for repeated-record regression.

For next-observation regression, Delta R2 ranged from -0.007 to 0.001 and all
intervals included zero. For maximum score within 730 days, gradient boosting showed
Delta R2 = 0.040 (95% interval 0.012 to 0.068) and an RMSE reduction of 3.79 points
(1.16 to 6.45), whereas ridge and random-forest intervals included zero. Paired
classification ROC-AUC increments were small and all intervals included zero. We
therefore revised the conclusion to describe the incremental contribution as small,
task-dependent, and algorithm-dependent rather than uniformly absent.

## General comment 2: unequal follow-up exposure

**Response.** We agree. The revised Results report the number of observed future
records and observed follow-up span. The baseline cohort had a median of 2 future
observations (IQR 1-3; range 1-13) and a median observed follow-up span of 371 days
(IQR 210-546; range 21-730). Analyses were repeated among athletes with at least two
(n=180) and at least three (n=118) future observations. We did not enter future
observation count as a predictor because it is not known at the index assessment.
The manuscript now describes threshold non-attainment as an observed non-case and
acknowledges possible informative censoring rather than implying complete continuous
surveillance.

We additionally repeated the continuous `future_max_points` analysis after restricting
each index instance to windows containing at least two or at least three subsequent
observations. Five-fold athlete-grouped cross-validation and the original fixed
hyperparameters were retained. With at least two future observations, the cohort
contained 737 instances from 185 athletes; gradient boosting with physical tests
achieved R2 = 0.500, MAE = 54.94, and RMSE = 74.58. With at least three future
observations, 524 instances from 128 athletes remained; corresponding values were
R2 = 0.526, MAE = 55.29, and RMSE = 70.71.

The paired gradient-boosting increment from physical tests did not persist in these
restricted cohorts. Delta R2 was -0.015 (95% athlete-cluster bootstrap interval -0.062
to 0.024) with at least two future observations and -0.007 (-0.046 to 0.027) with at
least three. RMSE reduction was -1.28 points (-4.85 to 2.11) and -0.59 points (-3.54
to 2.27), respectively; negative values indicate worse performance after adding the
tests. We have therefore revised the interpretation: `future_max_points` prediction
remained moderate under greater observed follow-up exposure, but the small physical-test
increment in the full cohort was not robust to these restrictions.

## General comment 3: prediction horizon leakage

**Response.** We agree. `delta_days` was the subsequently observed interval from the
index assessment to the next or best future record and was therefore unavailable at
the index date. It has been removed from every corrected predictor set and retained
only as a follow-up descriptor. All affected regression models, tables, figures,
abstract values, Results, and Discussion were regenerated.

For the gradient-boosting model with current score, context, implements, and physical
tests, next-observation R2 changed from 0.616 to 0.608, MAE from 43.48 to 44.90, and
RMSE from 70.49 to 71.27. For maximum score within 730 days, R2 changed from 0.681
to 0.616, MAE from 44.94 to 51.31, and RMSE from 65.23 to 71.88. The complete
before/after audit is publicly available in `delta_days_removal_impact.csv`.

## General comment 4: comparability of 900- and 950-point thresholds

**Response.** We clarified that these were operational upper-performance cutoffs,
not formal selection criteria. In this cohort they corresponded approximately to
the 73rd and 86th percentiles of future maximum score. Scores were expressed on the
standard points scale to improve comparability across events, but we agree that
transportability across sex, age category, event, and implement cannot be assumed.
Subgroup distributions are now reported, and this limitation is explicit.

## Specific comment: temporal alignment of current competitive score

**Provisional response; AUTHOR CONFIRMATION REQUIRED.** The spreadsheet contains a
season personal-best score but not the exact date of the competition in which it was
achieved. We are verifying against the original records whether the score assigned
to each index row was achieved on or before that assessment. We will either document
that rule and its quality control or treat score-anchored models as a sensitivity
analysis and emphasize models excluding current score. We have removed the previous
unsupported statement that availability at the index date was established.

## Specific comments: validation, metrics, and small event count

**Response.** Five-fold `GroupKFold` was used for repeated-record regression and
five-fold `StratifiedKFold` for the athlete-level binary outcomes. Stratification
was generated separately for each outcome with `shuffle=True` and `random_state=42`;
the same folds were reused for every algorithm and predictor set within that outcome.
The revised tables include the positive count in each fold. For `future_ge950`
(41 positives), we added ten-times repeated five-fold stratified cross-validation.
Across the ten repeats, mean ROC-AUC was 0.844 for random forest with physical tests
(repeat range summarized by the 2.5th-97.5th percentiles: 0.827-0.866). We added
PR-AUC, Brier score, calibration intercept, calibration slope, and calibration plots.

## Specific comments: chronological validation

**Response.** The original chronological split allowed training outcome windows to
extend toward the holdout baseline period. We replaced it with a strict split. The
holdout cutoff remained 27 October 2012, the 70th percentile of index dates, but
training index dates were restricted to 28 October 2010 or earlier, providing a full
730-day gap. The strict split included 168 training and 82 test athletes with no
athlete overlap. It was applied to `future_max_points`, `future_ge900`, and
`future_ge950`. For `future_ge900`, logistic regression achieved ROC-AUC 0.892,
PR-AUC 0.873, and Brier score 0.150. For `future_max_points`, holdout R2 ranged from
0.128 to 0.320, and the weaker transport performance is now discussed.

## Specific comment: participant characteristics

**Response.** We added a participant-characteristics table for the 289-athlete
baseline cohort, including sex, age category, event, age, current score, physical
tests, medicine-ball load, implement weight, and missingness.

## Specific comment: calendar period

**Response.** We added out-of-fold sensitivity summaries for 1997-2005, 2006-2012,
and 2013-2020 and discuss possible temporal changes in cohort composition,
measurement practice, and calibration. These summaries are not described as
independent external validation.

## Specific comment: medicine-ball protocol

**Response.** Recorded medicine-ball loads are now reported by sex and age category.
They ranged from 3 to 4 kg in most female categories and from 4 to 6 kg in male
categories, with small historical within-category variations. **AUTHOR CONFIRMATION
REQUIRED:** The Methods protocol wording and number of attempts must be checked
against the original testing manual/Fernandez et al. (2023); the analysis code cannot
establish how attempts were administered.

## Specific comment: 730-day rationale

**Response.** We clarified that 730 days was selected a priori as an operational
medium-term horizon covering approximately two annual training and competition
cycles while retaining sufficient repeated observations for grouped validation.
It does not imply complete observation for every athlete.

## Specific comment: improvement threshold sensitivity

**Response.** We retained the pre-specified 50-point outcome and added 25- and
75-point thresholds plus continuous-change regression. Conclusions are now based on
the pattern across these analyses rather than on a single dichotomy.

## Specific comment: permutation importance

**Response.** We clarified that permutation importance was computed only in held-out
validation folds, using three stratified folds, eight permutations per original
feature per fold, and ROC-AUC as the scoring metric. Reported means are averaged
across folds and the standard deviation is the SD of fold means. Interpretation is
now explicitly model-dependent, correlation-dependent, and non-causal.

## Terminology and editorial changes

**Response.** We defined `g1_current` in terms of its actual predictors, replaced
"future potential" with observed future performance/development, changed Figure 1
from "leakage-aware" to "athlete-aware", distinguished descriptive fold intervals
from bootstrap uncertainty intervals, and corrected the Code Availability wording.
Software versions and complete fixed hyperparameters are now reported. No grid
search, randomized search, nested cross-validation, or post-comparison tuning was
performed.
