# Athlete-aware longitudinal ML in youth athletics throwers

This repository contains the reproducible analysis code and aggregate outputs for a longitudinal machine-learning study of youth athletics throwers.

## Study aim

The workflow evaluates whether future throwing performance within a 730-day horizon can be forecast from repeated athlete monitoring records while avoiding leakage from repeated observations of the same athlete.

The computer-science contribution is the reproducible, athlete-aware validation workflow:

- automatic construction of future-prediction cohorts from irregular repeated observations;
- athlete-grouped cross-validation for repeated-measure regression;
- chronological holdout validation for future deployment;
- incremental feature-set benchmarking;
- model-agnostic permutation importance for coach-facing interpretation.

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
- `figures/figure_1_cohort_flow.png`
- `figures/figure_2_classification_auc.png`
- `figures/figure_3_permutation_importance.png`
- `figures/figure_4_future_regression_r2.png`

## Important modelling note

Models were compared using pre-specified hyperparameters. No GridSearchCV, RandomizedSearchCV, or nested cross-validation was performed. The goal was to evaluate a transparent leakage-aware longitudinal workflow and the incremental value of feature sets, not to maximise performance through extensive hyperparameter tuning.

## Licence

No public licence has been assigned yet. Add an explicit licence before making the repository public.
