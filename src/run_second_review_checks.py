"""Generate aggregate checks requested during the second review round."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "cohort_athlete_baseline_730d.csv"
RESULTS = ROOT / "results"

FEATURE_LABELS = {
    "g1_tests_only": "Physical tests only",
    "context_g1_no_current": "Context + physical tests, no current score",
    "current_only": "Current score only",
    "context_current": "Current score + athlete/event context",
    "context_current_implements": "Current + context + implements",
    "g1_current": "Current + context + implements + physical tests",
    "broader_g1_current": "Full broader sensitivity set",
}

FEATURE_ORDER = list(FEATURE_LABELS)

G1_CURRENT = [
    "PUNTOS",
    "Edad",
    "Sexo",
    "Categoria",
    "Prueba",
    "BOLA",
    "ARTEFACTO",
    "Dorsal",
    "Salto Vertical",
    "Longitud PJ",
    "Triple PJ",
]


def baseline_threshold_status() -> pd.DataFrame:
    baseline = pd.read_csv(DATA)
    rows = []
    for threshold in (900, 950):
        baseline_positive = baseline["PUNTOS"] >= threshold
        future_positive = baseline["future_max_points"] >= threshold
        rows.append(
            {
                "threshold": threshold,
                "n_cohort": len(baseline),
                "baseline_at_or_above": int(baseline_positive.sum()),
                "future_positive": int(future_positive.sum()),
                "already_at_or_above_among_future_positive": int(
                    (baseline_positive & future_positive).sum()
                ),
                "below_baseline_then_attained": int(
                    (~baseline_positive & future_positive).sum()
                ),
                "baseline_at_or_above_but_future_negative": int(
                    (baseline_positive & ~future_positive).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def best_algorithms() -> pd.DataFrame:
    metrics = pd.read_csv(RESULTS / "future_grouped_regression_metrics.csv")
    best = metrics.loc[
        metrics.groupby(["target", "feature_set"], sort=False)["r2_mean"].idxmax(),
        ["target", "feature_set", "model", "r2_mean", "mae_mean", "rmse_mean"],
    ].copy()
    best.insert(2, "predictor_set_label", best["feature_set"].map(FEATURE_LABELS))
    best["feature_set"] = pd.Categorical(
        best["feature_set"], categories=FEATURE_ORDER, ordered=True
    )
    return best.sort_values(["target", "feature_set"]).reset_index(drop=True)


def followup_table_oof() -> pd.DataFrame:
    models = pd.read_csv(RESULTS / "future_max_followup_sensitivity_models.csv")
    paired = pd.read_csv(RESULTS / "future_max_followup_sensitivity_incremental.csv")
    models = models[
        (models["model"] == "GradientBoosting")
        & models["feature_set"].isin(["context_current_implements", "g1_current"])
    ].copy()
    paired = paired[paired["model"] == "GradientBoosting"].set_index(
        "minimum_future_observations"
    )
    models["physical_tests"] = models["feature_set"].map(
        {"context_current_implements": "Without", "g1_current": "With"}
    )
    models["paired_delta_r2_oof"] = models["minimum_future_observations"].map(
        paired["delta_r2"]
    )
    models["paired_delta_r2_ci_low"] = models["minimum_future_observations"].map(
        paired["delta_r2_ci_low"]
    )
    models["paired_delta_r2_ci_high"] = models["minimum_future_observations"].map(
        paired["delta_r2_ci_high"]
    )
    columns = [
        "minimum_future_observations",
        "n_rows",
        "n_athletes",
        "physical_tests",
        "r2_fold_mean",
        "r2_oof",
        "mae_oof",
        "rmse_oof",
        "paired_delta_r2_oof",
        "paired_delta_r2_ci_low",
        "paired_delta_r2_ci_high",
    ]
    return models[columns].sort_values(
        ["minimum_future_observations", "physical_tests"], ascending=[True, False]
    )


def strict_temporal_predictors() -> pd.DataFrame:
    temporal = pd.read_csv(RESULTS / "strict_temporal_validation.csv")
    outcomes = temporal[["outcome", "feature_set"]].drop_duplicates()
    assert set(outcomes["feature_set"]) == {"g1_current"}
    outcomes["predictors"] = "; ".join(G1_CURRENT)
    return outcomes.sort_values("outcome").reset_index(drop=True)


def main() -> None:
    baseline_threshold_status().to_csv(
        RESULTS / "second_review_baseline_threshold_status.csv", index=False
    )
    best_algorithms().to_csv(
        RESULTS / "second_review_best_algorithms_by_predictor_set.csv", index=False
    )
    followup_table_oof().to_csv(
        RESULTS / "second_review_followup_sensitivity_oof.csv", index=False
    )
    strict_temporal_predictors().to_csv(
        RESULTS / "second_review_strict_temporal_predictors.csv", index=False
    )


if __name__ == "__main__":
    main()
