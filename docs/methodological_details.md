# Methodological details

## Software environment

- Python 3.12.13
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.13.1
- scikit-learn 1.9.0
- Matplotlib 3.9.2
- openpyxl 3.1.5

scikit-learn 1.9.0 is pinned because it reproduces the deposited `GroupKFold`
allocation exactly. The 1.9 release uses stable sorting in `GroupKFold`.

## Data quality and cohort eligibility

Athlete identifiers were constructed from normalized athlete name and date of
birth. Leading/trailing whitespace and repeated internal whitespace were removed,
and names were converted to uppercase. No fuzzy name matching or manual spelling
correction was applied.

The data audit found no exact duplicate rows. Seven athlete-date combinations
were represented by more than one non-identical source row (14 rows in total),
reflecting multiple events or different test values recorded on the same date.
Same-day records were not treated as longitudinal transitions because transitions
were required to have a strictly positive elapsed time.

There was no complete-case deletion across every candidate predictor. Eligibility
required a valid athlete identifier, index date, current score, and at least one
later score within 730 days. The four primary physical tests had no missing values
in the primary spreadsheet. `BOLA` was missing in 14 of 1,564 rows; model pipelines
imputed missing numeric values using the training-fold median. Broader secondary
variables had substantially more missingness and were used only in sensitivity
feature sets. Outcome rows with a missing target would be excluded, although the
primary spreadsheet had no missing `PUNTOS` values.

## Cohorts and outcomes

- Primary dataset: 1,564 records from 502 athletes.
- Next-observation cohort: 1,034 transitions from 289 athletes within 730 days.
- Horizon-maximum cohort: 1,041 index rows from 289 athletes.
- Athlete-level baseline cohort: 289 athletes, one first eligible index record each.

`next_points` was the score at the next strictly later observation within 730
days. `future_max_points` was the maximum score among all observed later records
within 730 days. `future_delta_points` was `future_max_points - PUNTOS`.
Classification outcomes were improvement of at least 50 points and a future
maximum of at least 900 or 950 points. Improvement thresholds of 25 and 75 points
were added as sensitivity analyses.

The 900- and 950-point outcomes represent future performance level, not incident
first attainment. In the 289-athlete baseline cohort, 41 athletes were already at
or above 900 points and 19 were already at or above 950 points. Among the 79
`future_ge900` cases, 35 were already at or above 900 at baseline and 44 attained
the threshold from below. Among the 41 `future_ge950` cases, the corresponding
counts were 18 and 23.

The 730-day window was selected a priori as a medium-term development horizon
covering approximately two annual training and competition cycles while retaining
enough repeated observations for athlete-grouped validation. It is an operational
window, not a claim that every athlete was continuously observed for exactly 730
days.

## Predictor timing correction

`delta_days` is the observed time from the index record to the next or best future
record. It is only known after follow-up and is therefore excluded from every
corrected predictor set. It remains in derived private cohorts solely as an outcome
timing descriptor. `n_future_observations` and `observed_followup_days` are also
follow-up descriptors and are not used as prospective predictors.

The raw spreadsheet does not contain the exact competition date on which `PUNTOS`
was achieved. The source study describes it as a season personal-best score.
Consequently, models containing `PUNTOS` are prospectively interpretable only if
the data owners confirm that the score used for each row was available on or before
that row's index assessment date. Feature sets without `PUNTOS` are retained as a
sensitivity analysis. This timing point requires domain confirmation before the
response to reviewers states that all index predictors were prospectively known.

## Models and fixed hyperparameters

Regression models:

- Ridge: `alpha=10.0`.
- Random forest: `n_estimators=200`, `min_samples_leaf=5`,
  `random_state=42`, `n_jobs=-1`.
- Gradient boosting: `random_state=42`; all remaining parameters used
  scikit-learn 1.9.0 defaults.

Classification models:

- Logistic regression: `max_iter=3000`, `class_weight="balanced"`,
  `random_state=42`.
- Random forest: `n_estimators=200`, `min_samples_leaf=5`,
  `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=-1`.
- Gradient boosting: `random_state=42`; all remaining parameters used
  scikit-learn 1.9.0 defaults.

No `GridSearchCV`, `RandomizedSearchCV`, hyperparameter selection, or nested
cross-validation was performed. Algorithms and hyperparameters were pre-specified.
The reported best-performing algorithm is a descriptive comparison among these
three fixed candidates, not the result of a tuning procedure.

## Preprocessing

All preprocessing was fitted within each training fold using a scikit-learn
`Pipeline` and `ColumnTransformer`.

- Numeric variables: median imputation followed by `StandardScaler`.
- Categorical variables: most-frequent imputation followed by
  `OneHotEncoder(handle_unknown="ignore")`.

The same precomputed folds were reused across every algorithm and feature set for
a given outcome, enabling paired comparisons.

## Predictor-set comparisons

Corrected future-regression sets were:

- `g1_tests_only`: the four primary physical tests.
- `context_g1_no_current`: context plus the four physical tests, without `PUNTOS`.
- `current_only`: `PUNTOS` only.
- `context_current`: `PUNTOS`, age, sex, category, and event.
- `context_current_implements`: `context_current` plus medicine-ball and competition-implement weights.
- `g1_current`: `context_current_implements` plus the four physical tests.
- `broader_g1_current`: `g1_current` plus broader anthropometric and test variables.

The primary incremental comparison used the same algorithm, rows, folds, and
preprocessing and contrasted `context_current_implements` with `g1_current`.
For classification, the paired comparison contrasted `context_current` with
`g1_current` for each fixed algorithm.

## Validation and uncertainty

Repeated-observation regression used five-fold `GroupKFold`, grouping by athlete.
The one-row-per-athlete classification cohort used five-fold `StratifiedKFold`
with `shuffle=True` and `random_state=42`. Stratification was generated separately
for each binary outcome. Fold membership was then held constant across algorithms
and predictor sets for that outcome.

Fold-level metric intervals are descriptive 95% t intervals calculated as
`mean +/- t(0.975, df=4) * SD / sqrt(5)`. They describe fold variability and are
not population-level confidence intervals.

Incremental-value uncertainty was calculated from paired out-of-fold predictions
using 2,000 percentile bootstrap replicates. Regression resampling was clustered
by athlete because athletes contributed repeated rows. Positive delta R2, delta
ROC-AUC, delta PR-AUC, delta MAE, delta RMSE, and delta Brier values were defined
to indicate improvement after adding physical tests.

For the 41-case `future_ge950` outcome, a 10-times repeated five-fold stratified
cross-validation sensitivity analysis was added. Classification reporting includes
ROC-AUC, precision-recall AUC, balanced accuracy, F1, Brier score, calibration
intercept, and calibration slope.

## Chronological validation

The cutoff date was the pre-specified 70th percentile of baseline index dates
(2012-10-27), preserving approximately 30% of athletes as a later holdout. To
prevent training outcomes from extending into the holdout baseline period, training
index dates were required to be on or before 2010-10-28, exactly 730 days before
the cutoff. This produced 168 training athletes and 82 later test athletes, with no
athlete overlap. The same strict split was used for `future_max_points`,
`future_ge900`, and `future_ge950`.

All three strict chronological analyses used the same `g1_current` predictors:
`PUNTOS`, `Edad`, `Sexo`, `Categoria`, `Prueba`, `BOLA`, `ARTEFACTO`, `Dorsal`,
`Salto Vertical`, `Longitud PJ`, and `Triple PJ`.

## Follow-up exposure and calendar sensitivity

The baseline cohort had a median of 2 future observations (IQR 1-3; range 1-13)
and a median observed follow-up span of 371 days (IQR 210-546; range 21-730).
Sensitivity analyses were repeated among athletes with at least two and at least
three future observations. Follow-up count was not entered as a predictor because
it is not known at the index date.

Full-cohort out-of-fold predictions were also summarized according to the calendar
year of the index observation for 1997-2005, 2006-2012, and 2013-2020. These are
sensitivity summaries rather than period-specific model fits or independent temporal
validations. For continuous `future_max_points`, Gradient Boosting was compared with
the `context_current_implements` and `g1_current` predictor sets. For `future_ge900`,
Random Forest was compared with `context_current` and `g1_current`. Within each period,
the incremental value of the four physical tests was calculated from paired OOF
predictions using 2,000 percentile bootstrap replicates; regression resampling was
clustered by athlete. Athletes could contribute index instances to more than one
calendar period in the repeated-instance regression cohort.

For the repeated-instance `future_max_points` cohort, follow-up sensitivity was
defined at the index-instance level. An instance was retained only when its 730-day
window contained at least two or at least three strictly later records. Each restricted
cohort was re-evaluated using five-fold `GroupKFold` by athlete, the same preprocessing,
fixed hyperparameters, and predictor sets as the primary analysis. The paired comparison
between `context_current_implements` and `g1_current` used 2,000 athlete-cluster
bootstrap replicates. Follow-up count remained an eligibility restriction only and was
never entered as a predictor.

The at-least-two restriction yielded 737 prediction instances from 185 athletes; the
at-least-three restriction yielded 524 instances from 128 athletes. Detailed fold,
aggregate-model, and paired-increment results are available in
`future_max_followup_sensitivity_fold_metrics.csv`,
`future_max_followup_sensitivity_models.csv`, and
`future_max_followup_sensitivity_incremental.csv`.

The original sensitivity table displayed mean fold R2 values while its paired Delta
R2 was calculated from pooled OOF predictions. This explains why the at-least-three
values 0.545 and 0.526 did not subtract to the reported -0.007: their direct
fold-mean difference is -0.019, whereas pooled OOF R2 was 0.568 without tests and
0.561 with tests, giving -0.007. For consistency, the revised table reports pooled
OOF R2, MAE, and RMSE throughout. The analogous at-least-two OOF values are 0.574
without tests and 0.560 with tests, giving Delta R2 = -0.015.

For Table 3, the best algorithm within each predictor set was selected by highest
mean five-fold R2. For `next_points`, the seven algorithms in table order were
Ridge, Ridge, Ridge, Random Forest, Random Forest, Ridge, and Random Forest. For
`future_max_points`, they were Ridge, Ridge, Ridge, Ridge, Ridge, Gradient Boosting,
and Ridge.

## Permutation importance

Permutation importance for `future_ge900` was calculated only on held-out
validation folds. A three-fold stratified split was used, with eight permutations
per original feature in each fold and ROC-AUC as the scoring metric. Feature-level
means were averaged across folds; the reported standard deviation is the standard
deviation of the three fold means. Importance is model- and correlation-dependent
and is not interpreted as a causal effect.
