from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("ML_THROWERS_RAW_DIR", ROOT / "data" / "raw"))
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

RANDOM_STATE = 42
HORIZON_DAYS = 730


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    mean = float(np.nanmean(arr))
    if len(arr) <= 1:
        return mean, float("nan"), float("nan")
    valid = arr[np.isfinite(arr)]
    if len(valid) <= 1:
        return mean, float("nan"), float("nan")
    margin = float(t.ppf(0.975, df=len(valid) - 1)) * float(np.std(valid, ddof=1)) / np.sqrt(len(valid))
    return mean, mean - margin, mean + margin


@dataclass(frozen=True)
class FeatureSets:
    g1_tests: list[str]
    g2_strength: list[str]
    context: list[str]
    broader: list[str]
    current_anchor: list[str]


def ensure_dirs() -> None:
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, prefix: str) -> str:
    for col in df.columns:
        if str(col).lower().startswith(prefix.lower()):
            return str(col)
    raise KeyError(f"No column starts with {prefix!r}")


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def load_throwers() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    g1 = pd.read_excel(RAW_DIR / "DatosGrupo1.xlsx")
    g2 = pd.read_excel(RAW_DIR / "DatosGrupo2.xlsx")
    g3 = pd.read_excel(RAW_DIR / "DatosGrupo3.xlsx")
    g2_equals_g3 = g2.equals(g3)

    category_col = find_column(g1, "Categor")
    concentration_col = find_column(g1, "Concentr")

    for df in (g1, g2):
        local_category = find_column(df, "Categor")
        local_concentration = find_column(df, "Concentr")
        rename = {}
        if local_category != "Categoria":
            rename[local_category] = "Categoria"
        if local_concentration != "Concentracion":
            rename[local_concentration] = "Concentracion"
        df.rename(columns=rename, inplace=True)

        for col in ["Atleta", "Prueba", "Sexo", "Categoria", "Concentracion"]:
            df[col] = normalize_text(df[col])
        df["Prueba"] = df["Prueba"].str.upper()
        df["athlete_key"] = (
            normalize_text(df["Atleta"]).str.upper()
            + "|"
            + pd.to_datetime(df["Nacimiento"], errors="coerce")
            .dt.strftime("%Y-%m-%d")
            .fillna("missing_birth")
        )
        df["Comienzo"] = pd.to_datetime(df["Comienzo"], errors="coerce")
        df["year"] = df["Comienzo"].dt.year

    return g1, g2, g2_equals_g3


def get_feature_sets() -> FeatureSets:
    return FeatureSets(
        g1_tests=["Dorsal", "Salto Vertical", "Longitud PJ", "Triple PJ"],
        g2_strength=["Cargada", "Pectoral", "Sentadilla"],
        context=["Edad", "Sexo", "Categoria", "Prueba", "BOLA", "ARTEFACTO"],
        broader=["30ml", "Flexibilidad", "Talla", "Peso", "Envergadura"],
        # Only information available at the index assessment is permitted.
        # delta_days describes the subsequently observed follow-up and is not a predictor.
        current_anchor=["PUNTOS"],
    )


def build_next_observation_cohort(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    rows = []
    for _, group in df.sort_values(["athlete_key", "Comienzo"]).groupby("athlete_key"):
        group = group.sort_values("Comienzo")
        for pos in range(len(group) - 1):
            current = group.iloc[pos].copy()
            nxt = group.iloc[pos + 1]
            delta_days = int((nxt["Comienzo"] - current["Comienzo"]).days)
            if 0 < delta_days <= horizon_days:
                current["next_points"] = nxt["PUNTOS"]
                current["next_date"] = nxt["Comienzo"]
                current["delta_days"] = delta_days
                rows.append(current)
    return pd.DataFrame(rows)


def build_horizon_max_cohort(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    rows = []
    for _, group in df.sort_values(["athlete_key", "Comienzo"]).groupby("athlete_key"):
        group = group.sort_values("Comienzo")
        dates = pd.to_datetime(group["Comienzo"])
        for idx, current in group.iterrows():
            future = group[dates > current["Comienzo"]].copy()
            if future.empty:
                continue
            future_delta = (future["Comienzo"] - current["Comienzo"]).dt.days
            future = future[(future_delta > 0) & (future_delta <= horizon_days)]
            if future.empty:
                continue
            best_idx = future["PUNTOS"].idxmax()
            row = current.copy()
            row["future_max_points"] = future.loc[best_idx, "PUNTOS"]
            row["future_max_date"] = future.loc[best_idx, "Comienzo"]
            row["future_delta_points"] = row["future_max_points"] - row["PUNTOS"]
            row["n_future_observations"] = len(future)
            row["delta_days"] = int((future.loc[best_idx, "Comienzo"] - row["Comienzo"]).days)
            row["last_future_date"] = future["Comienzo"].max()
            row["observed_followup_days"] = int((row["last_future_date"] - row["Comienzo"]).days)
            rows.append(row)
    return pd.DataFrame(rows)


def build_baseline_cohort(df: pd.DataFrame, horizon_days: int, g1_tests: list[str]) -> pd.DataFrame:
    rows = []
    for _, group in df.sort_values(["athlete_key", "Comienzo"]).groupby("athlete_key"):
        group = group.sort_values("Comienzo")
        dates = pd.to_datetime(group["Comienzo"])
        for _, current in group.iterrows():
            if not current[g1_tests].notna().all():
                continue
            future = group[dates > current["Comienzo"]].copy()
            if future.empty:
                continue
            future_delta = (future["Comienzo"] - current["Comienzo"]).dt.days
            future = future[(future_delta > 0) & (future_delta <= horizon_days)]
            if future.empty:
                continue
            best_idx = future["PUNTOS"].idxmax()
            row = current.copy()
            row["future_max_points"] = future.loc[best_idx, "PUNTOS"]
            row["future_max_date"] = future.loc[best_idx, "Comienzo"]
            row["future_delta_points"] = row["future_max_points"] - row["PUNTOS"]
            row["n_future_observations"] = len(future)
            row["delta_days"] = int((future.loc[best_idx, "Comienzo"] - row["Comienzo"]).days)
            row["last_future_date"] = future["Comienzo"].max()
            row["observed_followup_days"] = int((row["last_future_date"] - row["Comienzo"]).days)
            rows.append(row)
            break
    return pd.DataFrame(rows)


def feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in features if col not in numeric]
    return numeric, categorical


def make_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric, categorical = feature_types(df, features)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def make_pipeline(df: pd.DataFrame, features: list[str], model) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(df, features)), ("model", model)])


def regression_models() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def classification_models() -> dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def clean_for_features(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    cols = list(dict.fromkeys(features + [target, "athlete_key", "Comienzo", "year"]))
    available = [col for col in cols if col in df.columns]
    return df[available].dropna(subset=[target]).copy()


def evaluate_grouped_regression(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target: str,
    label: str,
) -> pd.DataFrame:
    rows = []
    for feature_set_name, features in feature_sets.items():
        data = clean_for_features(df, features, target)
        groups = data["athlete_key"]
        n_splits = min(5, groups.nunique())
        if len(data) < 20 or n_splits < 2:
            continue
        cv = GroupKFold(n_splits=n_splits)
        for model_name, model in regression_models().items():
            pipe = make_pipeline(data, features, model)
            scores = cross_validate(
                pipe,
                data[features],
                data[target],
                groups=groups,
                cv=cv,
                scoring={
                    "r2": "r2",
                    "mae": "neg_mean_absolute_error",
                    "rmse": "neg_root_mean_squared_error",
                },
                error_score="raise",
            )
            r2_mean, r2_low, r2_high = mean_ci(scores["test_r2"])
            mae_values = -scores["test_mae"]
            rmse_values = -scores["test_rmse"]
            mae_mean, mae_low, mae_high = mean_ci(mae_values)
            rmse_mean, rmse_low, rmse_high = mean_ci(rmse_values)
            rows.append(
                {
                    "analysis": label,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n_rows": len(data),
                    "n_athletes": groups.nunique(),
                    "target": target,
                    "r2_mean": r2_mean,
                    "r2_ci_low": r2_low,
                    "r2_ci_high": r2_high,
                    "r2_sd": float(scores["test_r2"].std()),
                    "mae_mean": mae_mean,
                    "mae_ci_low": mae_low,
                    "mae_ci_high": mae_high,
                    "rmse_mean": rmse_mean,
                    "rmse_ci_low": rmse_low,
                    "rmse_ci_high": rmse_high,
                }
            )
    return pd.DataFrame(rows)


def evaluate_classification_cv(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    outcomes: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for outcome_name, y in outcomes.items():
        y = pd.Series(y, index=df.index).astype(int)
        if y.nunique() < 2:
            continue
        for feature_set_name, features in feature_sets.items():
            data = df[features].copy()
            valid = y.notna()
            data = data.loc[valid]
            y_valid = y.loc[valid]
            min_class = int(y_valid.value_counts().min())
            n_splits = min(5, min_class)
            if n_splits < 2:
                continue
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            for model_name, model in classification_models().items():
                pipe = make_pipeline(df.loc[valid], features, model)
                scores = cross_validate(
                    pipe,
                    data,
                    y_valid,
                    cv=cv,
                    scoring={
                        "auc": "roc_auc",
                        "balanced_accuracy": "balanced_accuracy",
                        "f1": "f1",
                    },
                    error_score="raise",
                )
                auc_mean, auc_low, auc_high = mean_ci(scores["test_auc"])
                ba_mean, ba_low, ba_high = mean_ci(scores["test_balanced_accuracy"])
                f1_mean, f1_low, f1_high = mean_ci(scores["test_f1"])
                rows.append(
                    {
                        "outcome": outcome_name,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "n_rows": len(data),
                        "positive_rate": float(y_valid.mean()),
                        "auc_mean": auc_mean,
                        "auc_ci_low": auc_low,
                        "auc_ci_high": auc_high,
                        "auc_sd": float(scores["test_auc"].std()),
                        "balanced_accuracy_mean": ba_mean,
                        "balanced_accuracy_ci_low": ba_low,
                        "balanced_accuracy_ci_high": ba_high,
                        "f1_mean": f1_mean,
                        "f1_ci_low": f1_low,
                        "f1_ci_high": f1_high,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_temporal_holdout(
    df: pd.DataFrame,
    features: list[str],
    outcome_name: str,
    y: pd.Series,
    train_fraction: float = 0.70,
) -> pd.DataFrame:
    data = df.copy()
    y = pd.Series(y, index=df.index).astype(int)
    order = data["Comienzo"].sort_values()
    cutoff_date = order.iloc[int(np.floor(len(order) * train_fraction))]
    train_mask = data["Comienzo"] <= cutoff_date
    test_mask = data["Comienzo"] > cutoff_date
    rows = []
    for model_name, model in classification_models().items():
        pipe = make_pipeline(data, features, model)
        pipe.fit(data.loc[train_mask, features], y.loc[train_mask])
        proba = predict_probability(pipe, data.loc[test_mask, features])
        pred = (proba >= 0.5).astype(int)
        rows.append(
            {
                "outcome": outcome_name,
                "model": model_name,
                "feature_set": "g1_current",
                "cutoff_date": str(pd.Timestamp(cutoff_date).date()),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "test_positive_rate": float(y.loc[test_mask].mean()),
                "auc": safe_auc(y.loc[test_mask], proba),
                "balanced_accuracy": float(balanced_accuracy_score(y.loc[test_mask], pred)),
                "f1": float(f1_score(y.loc[test_mask], pred, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def predict_probability(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    model = pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    decision = pipe.decision_function(x)
    return 1.0 / (1.0 + np.exp(-decision))


def safe_auc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y_arr = np.asarray(list(y_true))
    if len(np.unique(y_arr)) < 2:
        return float("nan")
    return float(roc_auc_score(y_arr, scores))


def permutation_importance_table(
    df: pd.DataFrame,
    features: list[str],
    y: pd.Series,
    model,
    scoring: str,
) -> pd.DataFrame:
    y = pd.Series(y, index=df.index).astype(int)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(df[features], y), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        pipe = make_pipeline(df, features, model)
        pipe.fit(train[features], y_train)
        result = permutation_importance(
            pipe,
            test[features],
            y_test,
            n_repeats=8,
            random_state=RANDOM_STATE + fold,
            scoring=scoring,
        )
        for feature, mean, sd in zip(features, result.importances_mean, result.importances_std):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_sd": float(sd),
                }
            )
    out = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_sd=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
    )
    return out


def describe_dataset(g1: pd.DataFrame, g2: pd.DataFrame, g2_equals_g3: bool) -> dict:
    counts = g1.groupby("athlete_key").size()
    return {
        "g1_rows": int(len(g1)),
        "g1_athletes": int(g1["athlete_key"].nunique()),
        "g1_date_min": str(g1["Comienzo"].min().date()),
        "g1_date_max": str(g1["Comienzo"].max().date()),
        "g1_sex_counts": g1["Sexo"].value_counts().to_dict(),
        "g1_event_counts": g1["Prueba"].value_counts().to_dict(),
        "g1_category_counts": g1["Categoria"].value_counts().to_dict(),
        "athletes_with_2plus": int((counts >= 2).sum()),
        "athletes_with_3plus": int((counts >= 3).sum()),
        "athletes_with_5plus": int((counts >= 5).sum()),
        "g2_rows": int(len(g2)),
        "g2_athletes": int(g2["athlete_key"].nunique()),
        "g2_equals_g3": bool(g2_equals_g3),
    }


def plot_cohort_flow(summary: dict, next_cohort: pd.DataFrame, baseline: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [
        ("Raw cleaned records", f"{summary['g1_rows']} records\n{summary['g1_athletes']} athletes"),
        ("Repeated-observation pool", f"{summary['athletes_with_2plus']} athletes with >=2 records"),
        ("Next-observation cohort", f"{len(next_cohort)} transitions\n{next_cohort['athlete_key'].nunique()} athletes"),
        ("Athlete baseline cohort", f"{len(baseline)} athletes\none baseline record each"),
    ]
    xs = [0.07, 0.32, 0.58, 0.82]
    for i, ((title, body), x) in enumerate(zip(boxes, xs)):
        ax.text(
            x,
            0.48,
            f"{title}\n\n{body}",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F5F7FA", ec="#1F4E79", lw=1.4),
        )
        if i < len(xs) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.105, 0.48),
                xytext=(x + 0.105, 0.48),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#1F4E79"),
            )
    ax.set_title("Athlete-aware longitudinal cohort construction", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.20, 0.78)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_1_cohort_flow.png", dpi=300)
    plt.close(fig)


def pretty_label(label: str) -> str:
    mapping = {
        "context_current": "Context +\ncurrent score",
        "g1_current": "Context + current score\n+ physical tests",
        "g1_tests_only": "Physical tests\nonly",
        "context_g1_no_current": "Context +\nphysical tests",
        "current_only": "Current score",
        "context_current_implements": "Current + context\n+ implement",
        "broader_g1_current": "Current + context + tests\n+ broad features",
        "future_ge900": "Future score >= 900",
        "future_ge950": "Future score >= 950",
        "improve_ge50": "Improvement >= 50",
        "PUNTOS": "Current score",
        "Sexo": "Sex",
        "Edad": "Age",
        "ARTEFACTO": "Implement weight",
        "Prueba": "Event",
        "Categoria": "Age category",
        "Longitud PJ": "Standing long jump",
        "BOLA": "Medicine ball weight",
        "Triple PJ": "Standing triple jump",
        "Salto Vertical": "Vertical jump",
        "Dorsal": "Backward overhead throw",
    }
    return mapping.get(label, label.replace("_", " "))


def plot_classification_auc(metrics: pd.DataFrame) -> None:
    best = (
        metrics[metrics["feature_set"].isin(["context_current", "g1_current"])]
        .sort_values(["outcome", "auc_mean"], ascending=[True, False])
        .groupby(["outcome", "feature_set"], as_index=False)
        .first()
    )
    outcomes = list(best["outcome"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    width = 0.35
    x = np.arange(len(outcomes))
    palette = {"context_current": "#64748B", "g1_current": "#0F766E"}
    for offset, feature_set in [(-width / 2, "context_current"), (width / 2, "g1_current")]:
        vals = []
        labels = []
        for outcome in outcomes:
            row = best[(best["outcome"] == outcome) & (best["feature_set"] == feature_set)]
            vals.append(float(row["auc_mean"].iloc[0]) if len(row) else np.nan)
            labels.append(str(row["model"].iloc[0]) if len(row) else "")
        bars = ax.bar(x + offset, vals, width, label=pretty_label(feature_set), color=palette[feature_set])
        for bar, model_name in zip(bars, labels):
            if np.isfinite(bar.get_height()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    model_name,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=20,
                )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Cross-validated AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(outcome) for outcome in outcomes], rotation=0, ha="center")
    ax.legend(title="Feature set")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_2_classification_auc.png", dpi=300)
    plt.close(fig)


def plot_permutation_importance(importance: pd.DataFrame) -> None:
    top = importance.head(12).sort_values("importance_mean").copy()
    top["feature_pretty"] = top["feature"].map(pretty_label)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(top["feature_pretty"], top["importance_mean"], xerr=top["importance_sd"], color="#2563EB", alpha=0.85)
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_xlabel("Mean decrease in ROC AUC after permutation")
    ax.set_title("Permutation importance for future high-performance classification")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_3_permutation_importance.png", dpi=300)
    plt.close(fig)


def plot_regression_results(regression: pd.DataFrame) -> None:
    subset = regression[regression["analysis"] == "future_next_observation"].copy()
    if subset.empty:
        return
    order = [
        "g1_tests_only",
        "current_only",
        "context_current",
        "context_current_implements",
        "g1_current",
    ]
    best = subset.sort_values("r2_mean", ascending=False).groupby("feature_set", as_index=False).first()
    best = best[best["feature_set"].isin(order)].copy()
    best["feature_set"] = pd.Categorical(best["feature_set"], categories=order, ordered=True)
    best = best.sort_values("feature_set")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(best))
    ax.bar(x, best["r2_mean"], color="#7C3AED", alpha=0.85)
    for i, row in enumerate(best.itertuples(index=False)):
        ax.text(i, row.r2_mean + 0.015, row.model, ha="center", va="bottom", fontsize=8, rotation=15)
    ax.set_ylabel("Grouped CV R2")
    ax.set_ylim(0, max(0.75, best["r2_mean"].max() + 0.08))
    ax.set_title("Future next-observation regression by feature set")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_label(label) for label in best["feature_set"]], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_4_future_regression_r2.png", dpi=300)
    plt.close(fig)


def write_summary(
    summary: dict,
    current_reg: pd.DataFrame,
    future_reg: pd.DataFrame,
    cls: pd.DataFrame,
    temporal: pd.DataFrame,
    importance: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> None:
    best_cls = cls.sort_values("auc_mean", ascending=False).groupby("outcome", as_index=False).first()
    best_future_reg = (
        future_reg[future_reg["analysis"] == "future_next_observation"]
        .sort_values("r2_mean", ascending=False)
        .head(5)
    )
    lines = [
        "# Results summary",
        "",
        "## Dataset",
        "",
        f"- Records in DatosGrupo1: {summary['g1_rows']}",
        f"- Unique athletes: {summary['g1_athletes']}",
        f"- Period: {summary['g1_date_min']} to {summary['g1_date_max']}",
        f"- Sex counts: {summary['g1_sex_counts']}",
        f"- Event counts: {summary['g1_event_counts']}",
        f"- DatosGrupo2 equals DatosGrupo3: {summary['g2_equals_g3']}",
        "",
        "## Best future classification models",
        "",
        "```text",
        best_cls.round(3).to_string(index=False),
        "```",
        "",
        "## Best future next-observation regression models",
        "",
        "```text",
        best_future_reg.round(3).to_string(index=False),
        "```",
        "",
        "## Chronological holdout for high performance (future_ge900)",
        "",
        "```text",
        temporal.round(3).to_string(index=False),
        "```",
        "",
        "## Permutation importance for future_ge900",
        "",
        "```text",
        importance.round(4).to_string(index=False),
        "```",
        "",
        "## Threshold summary",
        "",
        "```text",
        threshold_summary.round(3).to_string(index=False),
        "```",
        "",
    ]
    (RESULTS_DIR / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def write_descriptive_tables(baseline: pd.DataFrame, outcomes: dict[str, pd.Series], features: FeatureSets) -> None:
    baseline = baseline.copy()
    for name, outcome in outcomes.items():
        baseline[name] = pd.Series(outcome, index=baseline.index).astype(int)

    outcome_counts = []
    for grouping in ["Sexo", "Prueba", "Categoria"]:
        counts = (
            baseline.groupby(grouping)
            .agg(
                n=("athlete_key", "count"),
                improve_ge50_rate=("improve_ge50", "mean"),
                future_ge900_rate=("future_ge900", "mean"),
                future_ge950_rate=("future_ge950", "mean"),
            )
            .reset_index()
            .rename(columns={grouping: "group"})
        )
        counts.insert(0, "grouping", grouping)
        outcome_counts.append(counts)
    pd.concat(outcome_counts, ignore_index=True).to_csv(
        RESULTS_DIR / "baseline_outcome_distribution.csv",
        index=False,
    )

    descriptive_features = ["PUNTOS", "Edad"] + features.g1_tests + ["BOLA", "ARTEFACTO"]
    desc = (
        baseline.groupby("future_ge900")[descriptive_features]
        .agg(["count", "mean", "std"])
        .round(3)
    )
    desc.columns = [f"{col}_{stat}" for col, stat in desc.columns]
    desc.reset_index().to_csv(
        RESULTS_DIR / "baseline_descriptives_by_future_ge900.csv",
        index=False,
    )


def write_threshold_summary(baseline: pd.DataFrame) -> pd.DataFrame:
    future = baseline["future_max_points"].dropna()
    current = baseline["PUNTOS"].dropna()
    rows = []
    for threshold in [900, 950]:
        rows.append(
            {
                "threshold": threshold,
                "future_max_percentile_rank": float((future <= threshold).mean() * 100),
                "current_score_percentile_rank": float((current <= threshold).mean() * 100),
                "n_future_positive": int((future >= threshold).sum()),
                "future_positive_rate": float((future >= threshold).mean()),
            }
        )
    delta = baseline["future_delta_points"].dropna()
    rows.append(
        {
            "threshold": "improvement >=50",
            "future_max_percentile_rank": float("nan"),
            "current_score_percentile_rank": float("nan"),
            "n_future_positive": int((delta >= 50).sum()),
            "future_positive_rate": float((delta >= 50).mean()),
        }
    )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "threshold_summary.csv", index=False)
    return out


def main() -> None:
    ensure_dirs()
    features = get_feature_sets()
    g1, g2, g2_equals_g3 = load_throwers()

    summary = describe_dataset(g1, g2, g2_equals_g3)
    (RESULTS_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    next_cohort = build_next_observation_cohort(g1, HORIZON_DAYS)
    horizon_cohort = build_horizon_max_cohort(g1, HORIZON_DAYS)
    baseline = build_baseline_cohort(g1, HORIZON_DAYS, features.g1_tests)

    next_cohort.to_csv(DATA_DIR / "cohort_next_observation_730d.csv", index=False)
    horizon_cohort.to_csv(DATA_DIR / "cohort_horizon_max_730d.csv", index=False)
    baseline.to_csv(DATA_DIR / "cohort_athlete_baseline_730d.csv", index=False)

    current_feature_sets = {
        "g1_tests": features.g1_tests,
        "context_g1": features.context + features.g1_tests,
        "broader_context_g1": features.context + features.broader + features.g1_tests,
    }
    current_reg = evaluate_grouped_regression(g1, current_feature_sets, "PUNTOS", "same_observation")
    current_reg.to_csv(RESULTS_DIR / "current_grouped_regression_metrics.csv", index=False)

    future_feature_sets = {
        "g1_tests_only": features.g1_tests,
        "context_g1_no_current": features.context + features.g1_tests,
        "current_only": features.current_anchor,
        "context_current": features.current_anchor + ["Edad", "Sexo", "Categoria", "Prueba"],
        "context_current_implements": features.current_anchor + features.context,
        "g1_current": features.current_anchor + features.context + features.g1_tests,
        "broader_g1_current": features.current_anchor + features.context + features.g1_tests + features.broader,
    }
    future_reg_next = evaluate_grouped_regression(
        next_cohort,
        future_feature_sets,
        "next_points",
        "future_next_observation",
    )
    future_reg_max = evaluate_grouped_regression(
        horizon_cohort,
        future_feature_sets,
        "future_max_points",
        "future_horizon_max",
    )
    future_reg = pd.concat([future_reg_next, future_reg_max], ignore_index=True)
    future_reg.to_csv(RESULTS_DIR / "future_grouped_regression_metrics.csv", index=False)

    cls_feature_sets = {
        "context_no_current": features.context,
        "g1_no_current": features.context + features.g1_tests,
        "context_current": features.context + ["PUNTOS"],
        "g1_current": features.context + features.g1_tests + ["PUNTOS"],
    }
    outcomes = {
        "improve_ge50": baseline["future_delta_points"] >= 50,
        "future_ge900": baseline["future_max_points"] >= 900,
        "future_ge950": baseline["future_max_points"] >= 950,
    }
    threshold_summary = write_threshold_summary(baseline)
    write_descriptive_tables(baseline, outcomes, features)
    cls = evaluate_classification_cv(baseline, cls_feature_sets, outcomes)
    cls.to_csv(RESULTS_DIR / "baseline_classification_metrics.csv", index=False)

    temporal = evaluate_temporal_holdout(
        baseline,
        cls_feature_sets["g1_current"],
        "future_ge900",
        outcomes["future_ge900"],
    )
    temporal.to_csv(RESULTS_DIR / "temporal_holdout_future_ge900.csv", index=False)

    importance = permutation_importance_table(
        baseline,
        cls_feature_sets["g1_current"],
        outcomes["future_ge900"],
        RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        scoring="roc_auc",
    )
    importance.to_csv(RESULTS_DIR / "permutation_importance_future_ge900.csv", index=False)

    plot_cohort_flow(summary, next_cohort, baseline)
    plot_classification_auc(cls)
    plot_permutation_importance(importance)

    write_summary(summary, current_reg, future_reg, cls, temporal, importance, threshold_summary)
    print(f"Wrote results to {RESULTS_DIR}")
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
