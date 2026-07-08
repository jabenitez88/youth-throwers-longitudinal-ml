# Methodological details

## Cohorts

- Primary dataset: 1,564 records from 502 athletes.
- Next-observation cohort: 1,034 transitions from 289 athletes within 730 days.
- Future maximum performance cohort (`future_max_points`): 1,041 rows from 289 athletes.
- Athlete-level baseline classification cohort: 289 athletes.

## Models and hyperparameters

Regression models:

- Ridge regression: `alpha=10.0`.
- Random forest regression: `n_estimators=200`, `min_samples_leaf=5`, `random_state=42`, `n_jobs=-1`.
- Gradient boosting regression: `random_state=42`; other parameters were scikit-learn defaults.

Classification models:

- Logistic regression: `max_iter=3000`, `class_weight="balanced"`, `random_state=42`.
- Random forest classification: `n_estimators=200`, `min_samples_leaf=5`, `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=-1`.
- Gradient boosting classification: `random_state=42`; other parameters were scikit-learn defaults.

No GridSearchCV, RandomizedSearchCV, or nested cross-validation was performed. Models were compared using pre-specified settings.

## Preprocessing

All preprocessing was fitted inside cross-validation folds using scikit-learn pipelines.

- Numeric predictors: median imputation and standardisation.
- Categorical predictors: most-frequent-category imputation and one-hot encoding with `handle_unknown="ignore"`.

## Validation

Repeated-observation regression used grouped cross-validation by athlete (`GroupKFold`, up to 5 folds). Classification used stratified cross-validation (`StratifiedKFold`, up to 5 folds, shuffle enabled, `random_state=42`) in the one-row-per-athlete baseline cohort.

A chronological holdout was also run for `future_ge900`. Baseline observations up to 2012-10-27 were used for training; later observations were held out for testing.

Chronological holdout counts:

- Training: 207 athletes, 43 positives for `future_ge900` (positive rate = 0.208).
- Test: 82 athletes, 36 positives for `future_ge900` (positive rate = 0.439).
- Full baseline cohort: 289 athletes, 79 positives (positive rate = 0.273).

## Feature-set comparisons

Future regression feature sets:

- `g1_tests_only`: `Dorsal`, `Salto Vertical`, `Longitud PJ`, `Triple PJ`.
- `context_g1_no_current`: `Edad`, `Sexo`, `Categoria`, `Prueba`, `BOLA`, `ARTEFACTO`, plus the four physical tests.
- `current_only`: `PUNTOS`, `delta_days`.
- `context_current`: `PUNTOS`, `delta_days`, `Edad`, `Sexo`, `Categoria`, `Prueba`.
- `context_current_implements`: `PUNTOS`, `delta_days`, `Edad`, `Sexo`, `Categoria`, `Prueba`, `BOLA`, `ARTEFACTO`.
- `g1_current`: `PUNTOS`, `delta_days`, context variables, and the four physical tests.
- `broader_g1_current`: `g1_current` plus `30ml`, `Flexibilidad`, `Talla`, `Peso`, `Envergadura`.

Classification feature sets:

- `context_no_current`: `Edad`, `Sexo`, `Categoria`, `Prueba`, `BOLA`, `ARTEFACTO`.
- `g1_no_current`: context variables plus the four physical tests.
- `context_current`: context variables plus `PUNTOS`.
- `g1_current`: context variables, four physical tests, and `PUNTOS`.

## Specific requested values

For `future_max_points`, the best no-physical-test model was `context_current_implements` with random forest:

- n = 1,041 rows; athletes = 289.
- R2 = 0.6560415384.
- 95% CI for R2 = 0.5445932665 to 0.7674898104.
- MAE = 45.2948236395.
- 95% CI for MAE = 35.7508093528 to 54.8388379261.
- RMSE = 67.6700468200.
- 95% CI for RMSE = 55.6121365445 to 79.7279570956.

For comparison, the best `future_max_points` model including physical tests was `g1_current` with gradient boosting:

- R2 = 0.6813216951.
- MAE = 44.9409179296.
- RMSE = 65.2349537783.

The incremental gain from adding physical tests to the best no-physical-test model was therefore approximately:

- Delta R2 = +0.0253.
- Delta MAE = -0.354 points.
- Delta RMSE = -2.435 points.

The best `improve_ge50` model was logistic regression with the `g1_current` feature set:

- Predictors: `Edad`, `Sexo`, `Categoria`, `Prueba`, `BOLA`, `ARTEFACTO`, `Dorsal`, `Salto Vertical`, `Longitud PJ`, `Triple PJ`, `PUNTOS`.
- AUC = 0.7752230378.
- Balanced accuracy = 0.7021488174.
- F1 = 0.6508685641.
