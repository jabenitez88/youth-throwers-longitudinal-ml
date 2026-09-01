from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_longitudinal_ml as ml


BOOTSTRAP_REPLICATES = 2000
REPEATED_CV_REPEATS = 10


def percentile_interval(values: Iterable[float]) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, pred)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
    }


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": ml.safe_auc(y_true, proba),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def calibration_statistics(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=3000)
    try:
        model.fit(logit, y_true)
    except ValueError:
        return float("nan"), float("nan")
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_groups = np.unique(groups)
    sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def paired_regression_bootstrap(
    y: np.ndarray,
    pred_without: np.ndarray,
    pred_with: np.ndarray,
    groups: np.ndarray,
) -> dict[str, tuple[float, float, float]]:
    observed_without = regression_metrics(y, pred_without)
    observed_with = regression_metrics(y, pred_with)
    observed = {
        "delta_r2": observed_with["r2"] - observed_without["r2"],
        "delta_mae": observed_without["mae"] - observed_with["mae"],
        "delta_rmse": observed_without["rmse"] - observed_with["rmse"],
    }
    samples = {metric: [] for metric in observed}
    rng = np.random.default_rng(ml.RANDOM_STATE)
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = bootstrap_indices(groups, rng)
        without = regression_metrics(y[idx], pred_without[idx])
        with_tests = regression_metrics(y[idx], pred_with[idx])
        samples["delta_r2"].append(with_tests["r2"] - without["r2"])
        samples["delta_mae"].append(without["mae"] - with_tests["mae"])
        samples["delta_rmse"].append(without["rmse"] - with_tests["rmse"])
    return {
        metric: (value, *percentile_interval(samples[metric]))
        for metric, value in observed.items()
    }


def paired_classification_bootstrap(
    y: np.ndarray,
    proba_without: np.ndarray,
    proba_with: np.ndarray,
    groups: np.ndarray,
) -> dict[str, tuple[float, float, float]]:
    observed_without = classification_metrics(y, proba_without)
    observed_with = classification_metrics(y, proba_with)
    observed = {
        "delta_roc_auc": observed_with["roc_auc"] - observed_without["roc_auc"],
        "delta_pr_auc": observed_with["pr_auc"] - observed_without["pr_auc"],
        "delta_brier": observed_without["brier"] - observed_with["brier"],
    }
    samples = {metric: [] for metric in observed}
    rng = np.random.default_rng(ml.RANDOM_STATE)
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = bootstrap_indices(groups, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        without = classification_metrics(y[idx], proba_without[idx])
        with_tests = classification_metrics(y[idx], proba_with[idx])
        samples["delta_roc_auc"].append(with_tests["roc_auc"] - without["roc_auc"])
        samples["delta_pr_auc"].append(with_tests["pr_auc"] - without["pr_auc"])
        samples["delta_brier"].append(without["brier"] - with_tests["brier"])
    return {
        metric: (value, *percentile_interval(samples[metric]))
        for metric, value in observed.items()
    }


def evaluate_paired_regression(
    df: pd.DataFrame,
    target: str,
    analysis: str,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    data = df.dropna(subset=[target]).reset_index(drop=True)
    groups = data["athlete_key"].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(data, data[target], groups))
    fold_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}

    for model_name, model in ml.regression_models().items():
        for feature_set, features in feature_sets.items():
            oof = np.full(len(data), np.nan)
            metric_values = {"r2": [], "mae": [], "rmse": []}
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                pipe = ml.make_pipeline(data, features, clone(model))
                pipe.fit(data.loc[train_idx, features], data.loc[train_idx, target])
                pred = pipe.predict(data.loc[test_idx, features])
                oof[test_idx] = pred
                metrics = regression_metrics(data.loc[test_idx, target].to_numpy(), pred)
                for metric, value in metrics.items():
                    metric_values[metric].append(value)
                fold_rows.append(
                    {
                        "analysis": analysis,
                        "target": target,
                        "model": model_name,
                        "feature_set": feature_set,
                        "fold": fold,
                        "n_rows": len(test_idx),
                        "n_athletes": int(data.loc[test_idx, "athlete_key"].nunique()),
                        **metrics,
                    }
                )
            predictions[(model_name, feature_set)] = oof
            aggregate = regression_metrics(data[target].to_numpy(), oof)
            row = {
                "analysis": analysis,
                "target": target,
                "model": model_name,
                "feature_set": feature_set,
                "n_rows": len(data),
                "n_athletes": int(data["athlete_key"].nunique()),
            }
            for metric, values in metric_values.items():
                mean, low, high = ml.mean_ci(values)
                row.update(
                    {
                        f"{metric}_fold_mean": mean,
                        f"{metric}_fold_ci_low": low,
                        f"{metric}_fold_ci_high": high,
                        f"{metric}_oof": aggregate[metric],
                    }
                )
            aggregate_rows.append(row)

    paired_rows = []
    without_name, with_name = list(feature_sets)
    for model_name in ml.regression_models():
        deltas = paired_regression_bootstrap(
            data[target].to_numpy(),
            predictions[(model_name, without_name)],
            predictions[(model_name, with_name)],
            groups,
        )
        row = {
            "analysis": analysis,
            "target": target,
            "model": model_name,
            "without_physical_tests": without_name,
            "with_physical_tests": with_name,
            "n_rows": len(data),
            "n_athletes": int(data["athlete_key"].nunique()),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
        for metric, (estimate, low, high) in deltas.items():
            row.update({metric: estimate, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        paired_rows.append(row)
    return pd.DataFrame(fold_rows), pd.DataFrame(aggregate_rows), pd.DataFrame(paired_rows), predictions


def evaluate_delta_days_removal(
    df: pd.DataFrame,
    target: str,
    analysis: str,
    corrected_feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    data = df.dropna(subset=[target]).reset_index(drop=True)
    groups = data["athlete_key"].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(data, data[target], groups))
    rows = []
    for model_name, model in ml.regression_models().items():
        for feature_set, corrected_features in corrected_feature_sets.items():
            version_results = {}
            original_features = [corrected_features[0], "delta_days", *corrected_features[1:]]
            for version, model_features in {
                "original_with_delta_days": original_features,
                "corrected_without_delta_days": corrected_features,
            }.items():
                fold_values = {"r2": [], "mae": [], "rmse": []}
                oof = np.full(len(data), np.nan)
                for train_idx, test_idx in splits:
                    pipe = ml.make_pipeline(data, model_features, clone(model))
                    pipe.fit(data.loc[train_idx, model_features], data.loc[train_idx, target])
                    pred = pipe.predict(data.loc[test_idx, model_features])
                    oof[test_idx] = pred
                    metrics = regression_metrics(data.loc[test_idx, target].to_numpy(), pred)
                    for metric, value in metrics.items():
                        fold_values[metric].append(value)
                version_results[version] = {
                    **{f"{metric}_fold_mean": float(np.mean(values)) for metric, values in fold_values.items()},
                    **{f"{metric}_oof": value for metric, value in regression_metrics(data[target].to_numpy(), oof).items()},
                }
            original = version_results["original_with_delta_days"]
            corrected = version_results["corrected_without_delta_days"]
            row = {
                "analysis": analysis,
                "target": target,
                "feature_set": feature_set,
                "model": model_name,
                "n_rows": len(data),
                "n_athletes": int(data["athlete_key"].nunique()),
            }
            for metric in ["r2", "mae", "rmse"]:
                for aggregation in ["fold_mean", "oof"]:
                    key = f"{metric}_{aggregation}"
                    row[f"original_{key}"] = original[key]
                    row[f"corrected_{key}"] = corrected[key]
                    row[f"change_{key}"] = corrected[key] - original[key]
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_classification(
    baseline: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    outcomes: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    fold_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    paired_rows: list[dict] = []
    predictions: dict[tuple[str, str, str], np.ndarray] = {}

    for outcome_name, outcome in outcomes.items():
        y = pd.Series(outcome, index=baseline.index).astype(int).to_numpy()
        n_splits = min(5, int(np.bincount(y).min()))
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ml.RANDOM_STATE).split(baseline, y))
        for model_name, model in ml.classification_models().items():
            for feature_set, features in feature_sets.items():
                oof = np.full(len(baseline), np.nan)
                metric_values = {name: [] for name in ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"]}
                for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                    pipe = ml.make_pipeline(baseline, features, clone(model))
                    pipe.fit(baseline.iloc[train_idx][features], y[train_idx])
                    proba = ml.predict_probability(pipe, baseline.iloc[test_idx][features])
                    oof[test_idx] = proba
                    metrics = classification_metrics(y[test_idx], proba)
                    for metric, value in metrics.items():
                        metric_values[metric].append(value)
                    fold_rows.append(
                        {
                            "outcome": outcome_name,
                            "model": model_name,
                            "feature_set": feature_set,
                            "fold": fold,
                            "n": len(test_idx),
                            "n_positive": int(y[test_idx].sum()),
                            "n_negative": int(len(test_idx) - y[test_idx].sum()),
                            **metrics,
                        }
                    )
                predictions[(outcome_name, model_name, feature_set)] = oof
                calibration_intercept, calibration_slope = calibration_statistics(y, oof)
                row = {
                    "outcome": outcome_name,
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_rows": len(y),
                    "n_positive": int(y.sum()),
                    "positive_rate": float(y.mean()),
                    "calibration_intercept_oof": calibration_intercept,
                    "calibration_slope_oof": calibration_slope,
                }
                oof_metrics = classification_metrics(y, oof)
                for metric, values in metric_values.items():
                    mean, low, high = ml.mean_ci(values)
                    row.update(
                        {
                            f"{metric}_mean": mean,
                            f"{metric}_ci_low": low,
                            f"{metric}_ci_high": high,
                            f"{metric}_oof": oof_metrics[metric],
                        }
                    )
                aggregate_rows.append(row)

            without_name, with_name = list(feature_sets)
            deltas = paired_classification_bootstrap(
                y,
                predictions[(outcome_name, model_name, without_name)],
                predictions[(outcome_name, model_name, with_name)],
                baseline["athlete_key"].to_numpy(),
            )
            row = {
                "outcome": outcome_name,
                "model": model_name,
                "without_physical_tests": without_name,
                "with_physical_tests": with_name,
                "n_rows": len(y),
                "n_positive": int(y.sum()),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
            for metric, (estimate, low, high) in deltas.items():
                row.update({metric: estimate, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
            paired_rows.append(row)

    return pd.DataFrame(fold_rows), pd.DataFrame(aggregate_rows), pd.DataFrame(paired_rows), predictions


def repeated_cv_future_ge950(
    baseline: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = (baseline["future_max_points"] >= 950).astype(int).to_numpy()
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=REPEATED_CV_REPEATS,
        random_state=ml.RANDOM_STATE,
    )
    split_list = list(cv.split(baseline, y))
    repeat_rows = []
    for model_name, model in ml.classification_models().items():
        for feature_set, features in feature_sets.items():
            repeat_predictions = np.full((REPEATED_CV_REPEATS, len(baseline)), np.nan)
            for split_number, (train_idx, test_idx) in enumerate(split_list):
                repeat = split_number // 5
                pipe = ml.make_pipeline(baseline, features, clone(model))
                pipe.fit(baseline.iloc[train_idx][features], y[train_idx])
                repeat_predictions[repeat, test_idx] = ml.predict_probability(pipe, baseline.iloc[test_idx][features])
            for repeat in range(REPEATED_CV_REPEATS):
                repeat_rows.append(
                    {
                        "outcome": "future_ge950",
                        "model": model_name,
                        "feature_set": feature_set,
                        "repeat": repeat + 1,
                        **classification_metrics(y, repeat_predictions[repeat]),
                    }
                )
    repeated = pd.DataFrame(repeat_rows)
    summary_rows = []
    for keys, group in repeated.groupby(["outcome", "model", "feature_set"]):
        row = dict(zip(["outcome", "model", "feature_set"], keys))
        row.update({"n": len(baseline), "n_positive": int(y.sum()), "repeats": REPEATED_CV_REPEATS})
        for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "f1", "brier"]:
            low, high = percentile_interval(group[metric])
            row.update({f"{metric}_mean": group[metric].mean(), f"{metric}_p2_5": low, f"{metric}_p97_5": high})
        summary_rows.append(row)
    return repeated, pd.DataFrame(summary_rows)


def strict_temporal_validation(
    baseline: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    ordered_dates = baseline["Comienzo"].sort_values().reset_index(drop=True)
    cutoff = ordered_dates.iloc[int(np.floor(0.70 * (len(ordered_dates) - 1)))]
    training_index_limit = cutoff - pd.Timedelta(days=ml.HORIZON_DAYS)
    train_mask = baseline["Comienzo"] <= training_index_limit
    test_mask = baseline["Comienzo"] > cutoff
    train = baseline.loc[train_mask].copy()
    test = baseline.loc[test_mask].copy()
    rows = []

    for model_name, model in ml.regression_models().items():
        pipe = ml.make_pipeline(baseline, features, clone(model))
        pipe.fit(train[features], train["future_max_points"])
        pred = pipe.predict(test[features])
        rows.append(
            {
                "task": "regression",
                "outcome": "future_max_points",
                "model": model_name,
                **regression_metrics(test["future_max_points"].to_numpy(), pred),
            }
        )
    for outcome_name, threshold in [("future_ge900", 900), ("future_ge950", 950)]:
        y_train = (train["future_max_points"] >= threshold).astype(int)
        y_test = (test["future_max_points"] >= threshold).astype(int).to_numpy()
        for model_name, model in ml.classification_models().items():
            pipe = ml.make_pipeline(baseline, features, clone(model))
            pipe.fit(train[features], y_train)
            proba = ml.predict_probability(pipe, test[features])
            rows.append(
                {
                    "task": "classification",
                    "outcome": outcome_name,
                    "model": model_name,
                    **classification_metrics(y_test, proba),
                }
            )
    out = pd.DataFrame(rows)
    out.insert(3, "feature_set", "g1_current")
    out["cutoff_date"] = str(pd.Timestamp(cutoff).date())
    out["training_index_limit"] = str(pd.Timestamp(training_index_limit).date())
    out["n_train"] = len(train)
    out["n_test"] = len(test)
    out["train_athletes"] = train["athlete_key"].nunique()
    out["test_athletes"] = test["athlete_key"].nunique()
    out["overlapping_athletes"] = len(set(train["athlete_key"]) & set(test["athlete_key"]))
    out["train_positive"] = out["outcome"].map(
        {
            "future_ge900": int((train["future_max_points"] >= 900).sum()),
            "future_ge950": int((train["future_max_points"] >= 950).sum()),
        }
    )
    out["test_positive"] = out["outcome"].map(
        {
            "future_ge900": int((test["future_max_points"] >= 900).sum()),
            "future_ge950": int((test["future_max_points"] >= 950).sum()),
        }
    )
    return out


def followup_exposure_tables(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = ["n_future_observations", "observed_followup_days"]
    rows = []
    strata = {"all": pd.Series(True, index=baseline.index)}
    for threshold in [900, 950]:
        strata[f"future_ge{threshold}_0"] = baseline["future_max_points"] < threshold
        strata[f"future_ge{threshold}_1"] = baseline["future_max_points"] >= threshold
    for stratum, mask in strata.items():
        for variable in variables:
            values = baseline.loc[mask, variable].dropna()
            rows.append(
                {
                    "stratum": stratum,
                    "variable": variable,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    completeness = pd.DataFrame(
        [
            {
                "minimum_future_observations": minimum,
                "n_athletes": int((baseline["n_future_observations"] >= minimum).sum()),
                "percent": float((baseline["n_future_observations"] >= minimum).mean() * 100),
                "future_ge900_positive": int(
                    ((baseline["n_future_observations"] >= minimum) & (baseline["future_max_points"] >= 900)).sum()
                ),
                "future_ge950_positive": int(
                    ((baseline["n_future_observations"] >= minimum) & (baseline["future_max_points"] >= 950)).sum()
                ),
            }
            for minimum in [1, 2, 3, 4]
        ]
    )
    return pd.DataFrame(rows), completeness


def followup_sensitivity(
    baseline: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for minimum in [1, 2, 3]:
        data = baseline[baseline["n_future_observations"] >= minimum].reset_index(drop=True)
        for outcome_name, threshold in [("future_ge900", 900), ("future_ge950", 950)]:
            y = (data["future_max_points"] >= threshold).astype(int).to_numpy()
            n_splits = min(5, int(np.bincount(y).min()))
            if n_splits < 2:
                continue
            splits = list(
                StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=ml.RANDOM_STATE,
                ).split(data, y)
            )
            for model_name, model in ml.classification_models().items():
                oof = np.full(len(data), np.nan)
                for train_idx, test_idx in splits:
                    pipe = ml.make_pipeline(data, features, clone(model))
                    pipe.fit(data.iloc[train_idx][features], y[train_idx])
                    oof[test_idx] = ml.predict_probability(pipe, data.iloc[test_idx][features])
                rows.append(
                    {
                        "minimum_future_observations": minimum,
                        "outcome": outcome_name,
                        "model": model_name,
                        "n": len(data),
                        "n_positive": int(y.sum()),
                        **classification_metrics(y, oof),
                    }
                )
    return pd.DataFrame(rows)


def future_max_followup_sensitivity(
    horizon_cohort: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat future-maximum regression after restricting observed follow-up exposure."""
    fold_tables = []
    aggregate_tables = []
    paired_tables = []
    for minimum in [1, 2, 3]:
        data = horizon_cohort[
            horizon_cohort["n_future_observations"] >= minimum
        ].reset_index(drop=True)
        folds, aggregate, paired, _ = evaluate_paired_regression(
            data,
            "future_max_points",
            f"future_horizon_max_min{minimum}",
            feature_sets,
        )
        for table in [folds, aggregate, paired]:
            table.insert(0, "minimum_future_observations", minimum)
            table.insert(1, "n_instances", len(data))
            table.insert(2, "n_unique_athletes", int(data["athlete_key"].nunique()))
        fold_tables.append(folds)
        aggregate_tables.append(aggregate)
        paired_tables.append(paired)
    return (
        pd.concat(fold_tables, ignore_index=True),
        pd.concat(aggregate_tables, ignore_index=True),
        pd.concat(paired_tables, ignore_index=True),
    )


def participant_characteristics(baseline: pd.DataFrame, features: ml.FeatureSets) -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous_rows = []
    continuous = ["Edad", "PUNTOS", *features.g1_tests, "BOLA", "ARTEFACTO"]
    for variable in continuous:
        values = pd.to_numeric(baseline[variable], errors="coerce").dropna()
        continuous_rows.append(
            {
                "variable": variable,
                "n": len(values),
                "missing": int(baseline[variable].isna().sum()),
                "mean": values.mean(),
                "sd": values.std(),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "min": values.min(),
                "max": values.max(),
            }
        )
    categorical_rows = []
    for variable in ["Sexo", "Categoria", "Prueba"]:
        counts = baseline[variable].fillna("Missing").value_counts(dropna=False)
        for level, count in counts.items():
            categorical_rows.append(
                {
                    "variable": variable,
                    "level": level,
                    "n": int(count),
                    "percent": float(count / len(baseline) * 100),
                }
            )
    return pd.DataFrame(continuous_rows), pd.DataFrame(categorical_rows)


def medicine_ball_loads(baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sex, category), group in baseline.groupby(["Sexo", "Categoria"], dropna=False):
        values = pd.to_numeric(group["BOLA"], errors="coerce").dropna()
        mode = values.mode()
        rows.append(
            {
                "sex": sex,
                "category": category,
                "n": len(group),
                "n_with_recorded_load": len(values),
                "load_kg_min": values.min(),
                "load_kg_max": values.max(),
                "load_kg_mode": mode.iloc[0] if len(mode) else float("nan"),
                "observed_loads_kg": ";".join(str(value) for value in sorted(values.unique())),
            }
        )
    return pd.DataFrame(rows)


def data_quality_audit(g1: pd.DataFrame, features: ml.FeatureSets) -> dict:
    audited_columns = [
        "Atleta",
        "Nacimiento",
        "Comienzo",
        "PUNTOS",
        *features.g1_tests,
        *features.context,
        *features.broader,
    ]
    repeated_dates = g1[g1.duplicated(["athlete_key", "Comienzo"], keep=False)]
    return {
        "rows": int(len(g1)),
        "athletes": int(g1["athlete_key"].nunique()),
        "exact_duplicate_rows": int(g1.duplicated().sum()),
        "athlete_date_groups_with_multiple_rows": int(
            repeated_dates.groupby(["athlete_key", "Comienzo"]).ngroups
        ),
        "rows_in_repeated_athlete_dates": int(len(repeated_dates)),
        "missing_by_column": {
            column: int(g1[column].isna().sum()) for column in dict.fromkeys(audited_columns)
        },
        "identifier_normalization": "trim whitespace, collapse internal whitespace, uppercase name, append date of birth",
        "fuzzy_name_matching": False,
    }


def calendar_period_performance(
    horizon_cohort: pd.DataFrame,
    baseline: pd.DataFrame,
    regression_predictions: dict[tuple[str, str], np.ndarray],
    classification_predictions: dict[tuple[str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_bins = [-np.inf, 2005, 2012, np.inf]
    period_labels = ["1997-2005", "2006-2012", "2013-2020"]
    regression_periods = pd.cut(
        horizon_cohort["year"], bins=period_bins, labels=period_labels
    )
    classification_periods = pd.cut(
        baseline["year"], bins=period_bins, labels=period_labels
    )
    model_rows = []
    paired_rows = []

    regression_target = horizon_cohort["future_max_points"].to_numpy()
    regression_groups = horizon_cohort["athlete_key"].to_numpy()
    regression_model = "GradientBoosting"
    regression_feature_sets = ["context_current_implements", "g1_current"]

    classification_target = (baseline["future_max_points"] >= 900).astype(int).to_numpy()
    classification_groups = baseline["athlete_key"].to_numpy()
    classification_model = "RandomForest"
    classification_feature_sets = ["context_current", "g1_current"]

    for period in period_labels:
        regression_mask = (regression_periods == period).to_numpy()
        regression_y = regression_target[regression_mask]
        regression_group_subset = regression_groups[regression_mask]
        for feature_set in regression_feature_sets:
            prediction = regression_predictions[(regression_model, feature_set)][regression_mask]
            model_rows.append(
                {
                    "period": period,
                    "task": "regression",
                    "outcome": "future_max_points",
                    "model": regression_model,
                    "feature_set": feature_set,
                    "n_instances": int(regression_mask.sum()),
                    "n_athletes": int(np.unique(regression_group_subset).size),
                    "n_positive": float("nan"),
                    **regression_metrics(regression_y, prediction),
                }
            )
        regression_without = regression_predictions[
            (regression_model, regression_feature_sets[0])
        ][regression_mask]
        regression_with = regression_predictions[
            (regression_model, regression_feature_sets[1])
        ][regression_mask]
        regression_delta = paired_regression_bootstrap(
            regression_y,
            regression_without,
            regression_with,
            regression_group_subset,
        )
        regression_row = {
            "period": period,
            "task": "regression",
            "outcome": "future_max_points",
            "model": regression_model,
            "without_physical_tests": regression_feature_sets[0],
            "with_physical_tests": regression_feature_sets[1],
            "n_instances": int(regression_mask.sum()),
            "n_athletes": int(np.unique(regression_group_subset).size),
            "n_positive": float("nan"),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
        for metric, (estimate, low, high) in regression_delta.items():
            regression_row.update(
                {metric: estimate, f"{metric}_ci_low": low, f"{metric}_ci_high": high}
            )
        paired_rows.append(regression_row)

        classification_mask = (classification_periods == period).to_numpy()
        classification_y = classification_target[classification_mask]
        classification_group_subset = classification_groups[classification_mask]
        for feature_set in classification_feature_sets:
            probability = classification_predictions[
                ("future_ge900", classification_model, feature_set)
            ][classification_mask]
            model_rows.append(
                {
                    "period": period,
                    "task": "classification",
                    "outcome": "future_ge900",
                    "model": classification_model,
                    "feature_set": feature_set,
                    "n_instances": int(classification_mask.sum()),
                    "n_athletes": int(np.unique(classification_group_subset).size),
                    "n_positive": int(classification_y.sum()),
                    **classification_metrics(classification_y, probability),
                }
            )
        classification_without = classification_predictions[
            ("future_ge900", classification_model, classification_feature_sets[0])
        ][classification_mask]
        classification_with = classification_predictions[
            ("future_ge900", classification_model, classification_feature_sets[1])
        ][classification_mask]
        classification_delta = paired_classification_bootstrap(
            classification_y,
            classification_without,
            classification_with,
            classification_group_subset,
        )
        classification_row = {
            "period": period,
            "task": "classification",
            "outcome": "future_ge900",
            "model": classification_model,
            "without_physical_tests": classification_feature_sets[0],
            "with_physical_tests": classification_feature_sets[1],
            "n_instances": int(classification_mask.sum()),
            "n_athletes": int(np.unique(classification_group_subset).size),
            "n_positive": int(classification_y.sum()),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
        for metric, (estimate, low, high) in classification_delta.items():
            classification_row.update(
                {metric: estimate, f"{metric}_ci_low": low, f"{metric}_ci_high": high}
            )
        paired_rows.append(classification_row)

    return pd.DataFrame(model_rows), pd.DataFrame(paired_rows)


def plot_calibration_curves(
    baseline: pd.DataFrame,
    predictions: dict[tuple[str, str, str], np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True, sharey=True)
    for ax, (outcome, threshold) in zip(axes, [("future_ge900", 900), ("future_ge950", 950)]):
        y = (baseline["future_max_points"] >= threshold).astype(int).to_numpy()
        for feature_set, color in [("context_current", "#64748B"), ("g1_current", "#0F766E")]:
            proba = predictions[(outcome, "RandomForest", feature_set)]
            observed, predicted = calibration_curve(y, proba, n_bins=5, strategy="quantile")
            ax.plot(predicted, observed, marker="o", color=color, label=ml.pretty_label(feature_set))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#222222", linewidth=1)
        ax.set_title(ml.pretty_label(outcome))
        ax.set_xlabel("Mean predicted probability")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Observed proportion")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ml.FIGURES_DIR / "figure_5_calibration.png", dpi=300)
    plt.close(fig)


def plot_paired_regression_increment(paired: pd.DataFrame) -> None:
    data = paired[
        paired["analysis"].isin(["future_next_observation", "future_horizon_max"])
    ].copy()
    algorithms = ["Ridge", "RandomForest", "GradientBoosting"]
    labels = {
        "future_next_observation": "Next observation",
        "future_horizon_max": "Maximum within 730 days",
    }
    colors = {
        "future_next_observation": "#64748B",
        "future_horizon_max": "#0F766E",
    }
    x = np.arange(len(algorithms))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for offset, analysis in [(-width / 2, "future_next_observation"), (width / 2, "future_horizon_max")]:
        subset = data[data["analysis"] == analysis].set_index("model").loc[algorithms]
        estimates = subset["delta_r2"].to_numpy()
        lower = estimates - subset["delta_r2_ci_low"].to_numpy()
        upper = subset["delta_r2_ci_high"].to_numpy() - estimates
        ax.errorbar(
            x + offset,
            estimates,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            markersize=7,
            capsize=4,
            linewidth=1.6,
            color=colors[analysis],
            label=labels[analysis],
        )
    ax.axhline(0, color="#222222", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Ridge", "Random forest", "Gradient boosting"])
    ax.set_ylabel("Delta R2 after adding physical tests")
    ax.set_title("Paired incremental value of physical tests within algorithm")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ml.FIGURES_DIR / "figure_4_future_regression_r2.png", dpi=300)
    plt.close(fig)


def write_revision_summary(
    paired_regression: pd.DataFrame,
    paired_classification: pd.DataFrame,
    temporal: pd.DataFrame,
    exposure: pd.DataFrame,
    future_max_exposure_models: pd.DataFrame,
    future_max_exposure_paired: pd.DataFrame,
    calendar_models: pd.DataFrame,
    calendar_paired: pd.DataFrame,
    future_ge950_fold_counts: pd.DataFrame,
) -> None:
    future_max_gradient_boosting = future_max_exposure_models[
        (future_max_exposure_models["model"] == "GradientBoosting")
        & (future_max_exposure_models["feature_set"] == "g1_current")
    ]
    future_max_gradient_boosting_paired = future_max_exposure_paired[
        future_max_exposure_paired["model"] == "GradientBoosting"
    ]
    lines = [
        "# Major-revision analysis summary",
        "",
        "All revised prospective models exclude `delta_days` from the predictor matrix.",
        "Positive paired deltas indicate improvement after adding physical tests.",
        f"Uncertainty for paired deltas is a {BOOTSTRAP_REPLICATES}-replicate athlete-cluster bootstrap percentile interval.",
        "",
        "## Incremental regression value of physical tests",
        "",
        "```text",
        paired_regression.round(4).to_string(index=False),
        "```",
        "",
        "## Incremental classification value of physical tests",
        "",
        "```text",
        paired_classification.round(4).to_string(index=False),
        "```",
        "",
        "## Strict chronological holdout",
        "",
        "```text",
        temporal.round(4).to_string(index=False),
        "```",
        "",
        "## Follow-up exposure",
        "",
        "```text",
        exposure.round(2).to_string(index=False),
        "```",
        "",
        "## Future-maximum regression by minimum follow-up exposure",
        "",
        "Gradient boosting with current score, context, implements, and physical tests:",
        "",
        "```text",
        future_max_gradient_boosting.round(4).to_string(index=False),
        "```",
        "",
        "Paired gradient-boosting increment after adding physical tests:",
        "",
        "```text",
        future_max_gradient_boosting_paired.round(4).to_string(index=False),
        "```",
        "",
        "## Calendar-period sensitivity",
        "",
        "The full-cohort out-of-fold predictions were summarized by index-observation period; these are not independent temporal validations.",
        "",
        "```text",
        calendar_models.round(4).to_string(index=False),
        "```",
        "",
        "Paired within-period increment after adding the four physical tests:",
        "",
        "```text",
        calendar_paired.round(4).to_string(index=False),
        "```",
        "",
        "## Future >=950 validation-fold event counts",
        "",
        "```text",
        future_ge950_fold_counts.to_string(index=False),
        "```",
        "",
        "Cross-validation intervals are descriptive t intervals across the five fixed folds; they are not inferential confidence intervals.",
    ]
    (ml.RESULTS_DIR / "MAJOR_REVISION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ml.ensure_dirs()
    features = ml.get_feature_sets()
    g1, _, _ = ml.load_throwers()
    next_cohort = ml.build_next_observation_cohort(g1, ml.HORIZON_DAYS)
    horizon_cohort = ml.build_horizon_max_cohort(g1, ml.HORIZON_DAYS)
    baseline = ml.build_baseline_cohort(g1, ml.HORIZON_DAYS, features.g1_tests).reset_index(drop=True)

    paired_regression_features = {
        "context_current_implements": features.current_anchor + features.context,
        "g1_current": features.current_anchor + features.context + features.g1_tests,
    }
    regression_runs = [
        (next_cohort, "next_points", "future_next_observation"),
        (horizon_cohort, "future_max_points", "future_horizon_max"),
        (baseline, "future_delta_points", "baseline_continuous_change"),
    ]
    regression_fold_tables = []
    regression_aggregate_tables = []
    regression_paired_tables = []
    horizon_regression_predictions = {}
    for data, target, analysis in regression_runs:
        fold, aggregate, paired, predictions = evaluate_paired_regression(
            data,
            target,
            analysis,
            paired_regression_features,
        )
        regression_fold_tables.append(fold)
        regression_aggregate_tables.append(aggregate)
        regression_paired_tables.append(paired)
        if analysis == "future_horizon_max":
            horizon_regression_predictions = predictions
    regression_folds = pd.concat(regression_fold_tables, ignore_index=True)
    regression_aggregate = pd.concat(regression_aggregate_tables, ignore_index=True)
    regression_paired = pd.concat(regression_paired_tables, ignore_index=True)

    delta_days_feature_sets = {
        "current_only": features.current_anchor,
        "context_current": features.current_anchor + ["Edad", "Sexo", "Categoria", "Prueba"],
        "context_current_implements": features.current_anchor + features.context,
        "g1_current": features.current_anchor + features.context + features.g1_tests,
    }
    delta_days_impact = pd.concat(
        [
            evaluate_delta_days_removal(
                next_cohort,
                "next_points",
                "future_next_observation",
                delta_days_feature_sets,
            ),
            evaluate_delta_days_removal(
                horizon_cohort,
                "future_max_points",
                "future_horizon_max",
                delta_days_feature_sets,
            ),
        ],
        ignore_index=True,
    )

    classification_feature_sets = {
        "context_current": features.context + ["PUNTOS"],
        "g1_current": features.context + features.g1_tests + ["PUNTOS"],
    }
    outcomes = {
        "improve_ge25": baseline["future_delta_points"] >= 25,
        "improve_ge50": baseline["future_delta_points"] >= 50,
        "improve_ge75": baseline["future_delta_points"] >= 75,
        "future_ge900": baseline["future_max_points"] >= 900,
        "future_ge950": baseline["future_max_points"] >= 950,
    }
    classification_folds, classification_aggregate, classification_paired, classification_predictions = evaluate_classification(
        baseline,
        classification_feature_sets,
        outcomes,
    )
    repeated_folds, repeated_summary = repeated_cv_future_ge950(baseline, classification_feature_sets)
    temporal = strict_temporal_validation(baseline, classification_feature_sets["g1_current"])
    followup_descriptives, followup_counts = followup_exposure_tables(baseline)
    followup_results = followup_sensitivity(baseline, classification_feature_sets["g1_current"])
    (
        future_max_followup_folds,
        future_max_followup_models,
        future_max_followup_paired,
    ) = future_max_followup_sensitivity(horizon_cohort, paired_regression_features)
    participant_continuous, participant_categorical = participant_characteristics(baseline, features)
    ball_loads = medicine_ball_loads(baseline)
    quality_audit = data_quality_audit(g1, features)
    calendar_models, calendar_paired = calendar_period_performance(
        horizon_cohort,
        baseline,
        horizon_regression_predictions,
        classification_predictions,
    )
    future_ge950_fold_counts = (
        classification_folds[classification_folds["outcome"] == "future_ge950"]
        [["fold", "n", "n_positive", "n_negative"]]
        .drop_duplicates()
        .sort_values("fold")
        .reset_index(drop=True)
    )

    regression_folds.to_csv(ml.RESULTS_DIR / "reviewer_regression_fold_metrics.csv", index=False)
    regression_aggregate.to_csv(ml.RESULTS_DIR / "reviewer_regression_paired_models.csv", index=False)
    regression_paired.to_csv(ml.RESULTS_DIR / "reviewer_regression_incremental_value.csv", index=False)
    delta_days_impact.to_csv(ml.RESULTS_DIR / "delta_days_removal_impact.csv", index=False)
    classification_folds.to_csv(ml.RESULTS_DIR / "reviewer_classification_fold_metrics.csv", index=False)
    classification_aggregate.to_csv(ml.RESULTS_DIR / "reviewer_classification_paired_models.csv", index=False)
    classification_paired.to_csv(ml.RESULTS_DIR / "reviewer_classification_incremental_value.csv", index=False)
    repeated_folds.to_csv(ml.RESULTS_DIR / "future_ge950_repeated_cv_repeats.csv", index=False)
    repeated_summary.to_csv(ml.RESULTS_DIR / "future_ge950_repeated_cv_summary.csv", index=False)
    temporal.to_csv(ml.RESULTS_DIR / "strict_temporal_validation.csv", index=False)
    followup_descriptives.to_csv(ml.RESULTS_DIR / "followup_exposure_descriptives.csv", index=False)
    followup_counts.to_csv(ml.RESULTS_DIR / "followup_exposure_counts.csv", index=False)
    followup_results.to_csv(ml.RESULTS_DIR / "followup_observation_sensitivity.csv", index=False)
    future_max_followup_folds.to_csv(
        ml.RESULTS_DIR / "future_max_followup_sensitivity_fold_metrics.csv",
        index=False,
    )
    future_max_followup_models.to_csv(
        ml.RESULTS_DIR / "future_max_followup_sensitivity_models.csv",
        index=False,
    )
    future_max_followup_paired.to_csv(
        ml.RESULTS_DIR / "future_max_followup_sensitivity_incremental.csv",
        index=False,
    )
    participant_continuous.to_csv(ml.RESULTS_DIR / "participant_characteristics_continuous.csv", index=False)
    participant_categorical.to_csv(ml.RESULTS_DIR / "participant_characteristics_categorical.csv", index=False)
    ball_loads.to_csv(ml.RESULTS_DIR / "medicine_ball_loads_by_sex_category.csv", index=False)
    (ml.RESULTS_DIR / "data_quality_audit.json").write_text(
        json.dumps(quality_audit, indent=2),
        encoding="utf-8",
    )
    calendar_models.to_csv(ml.RESULTS_DIR / "calendar_period_sensitivity.csv", index=False)
    calendar_paired.to_csv(
        ml.RESULTS_DIR / "calendar_period_sensitivity_incremental.csv", index=False
    )
    future_ge950_fold_counts.to_csv(
        ml.RESULTS_DIR / "future_ge950_fold_positive_counts.csv", index=False
    )

    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    (ml.RESULTS_DIR / "software_versions.json").write_text(json.dumps(software, indent=2), encoding="utf-8")
    plot_calibration_curves(baseline, classification_predictions)
    plot_paired_regression_increment(regression_paired)
    write_revision_summary(
        regression_paired,
        classification_paired,
        temporal,
        followup_counts,
        future_max_followup_models,
        future_max_followup_paired,
        calendar_models,
        calendar_paired,
        future_ge950_fold_counts,
    )
    print(f"Wrote major-revision analyses to {ml.RESULTS_DIR}")


if __name__ == "__main__":
    main()
