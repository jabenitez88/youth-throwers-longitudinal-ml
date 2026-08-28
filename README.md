# Athlete-aware longitudinal ML in youth athletics throwers

This repository contains the reproducible analysis code and aggregate outputs for a longitudinal machine-learning study of youth athletics throwers.

## Study aim

The workflow evaluates whether future throwing performance within a 730-day horizon can be forecast from repeated athlete monitoring records while avoiding leakage from repeated observations of the same athlete.

The computer-science contribution is the reproducible, athlete-aware validation workflow:

- automatic construction of future-prediction cohorts from irregular repeated observations;
- athlete-grouped cross-validation for repeated-measure regression;
- chronological holdout validation for future deployment;
- paired incremental feature-set benchmarking with athlete-cluster bootstrap intervals;
- follow-up exposure, threshold, calendar-period, repeated-CV, and calibration sensitivity analyses;
- model-agnostic permutation importance for coach-facing interpretation.

## Major-revision correction

The original release included `delta_days`, the subsequently observed interval to
the future record, in future-regression predictor sets. This variable is not known
at the index assessment and has been removed from all corrected models. It is now
retained only as a private cohort descriptor. The before/after audit is available
in `results/delta_days_removal_impact.csv`.

## Data availability

The raw Excel files are not included because they contain longitudinal athlete records. To rerun the analysis, place the three raw spreadsheets in `data/raw/`:

- `DatosGrupo1.xlsx`
- `DatosGrupo2.xlsx`
- `DatosGrupo3.xlsx`

Alternatively, set the environment variable `ML_THROWERS_RAW_DIR` to the folder containing those files.

Row-level processed cohorts are also excluded from version control by default. Aggregate metrics, figures, and manuscript-facing summaries are included in `results/` and `figures/`.

## Reproduce the analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python src/run_longitudinal_ml.py
python src/run_major_revision_analyses.py
```

The script writes:

- `data/processed/`: row-level derived cohorts, not intended for public sharing;
- `results/`: aggregate metrics and summaries;
- `figures/`: manuscript-ready figures.

## Main outputs

- `results/future_grouped_regression_metrics.csv`
- `results/baseline_classification_metrics.csv`
- `results/temporal_holdout_future_ge900.csv`
- `results/permutation_importance_future_ge900.csv`
- `results/delta_days_removal_impact.csv`
- `results/reviewer_regression_incremental_value.csv`
- `results/reviewer_classification_incremental_value.csv`
- `results/strict_temporal_validation.csv`
- `results/followup_exposure_counts.csv`
- `results/future_ge950_repeated_cv_summary.csv`
- `figures/figure_1_cohort_flow.png`
- `figures/figure_2_classification_auc.png`
- `figures/figure_3_permutation_importance.png`
- `figures/figure_4_future_regression_r2.png`
- `figures/figure_5_calibration.png`

## Important modelling note

Models were compared using pre-specified hyperparameters. No GridSearchCV, RandomizedSearchCV, or nested cross-validation was performed. The goal was to evaluate a transparent leakage-aware longitudinal workflow and the incremental value of feature sets, not to maximise performance through extensive hyperparameter tuning.

The source spreadsheet does not include the exact date on which the seasonal
personal-best score (`PUNTOS`) was achieved. Models containing that predictor are
prospectively interpretable only after the data owners confirm that the score was
available on or before each index assessment. No-current-score feature sets are
included as sensitivity analyses.

## Licence

No public licence has been assigned yet. Add an explicit licence before making the repository public.
