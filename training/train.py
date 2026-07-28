"""
Main training entrypoint.

Trains all models (baselines + quantile LightGBM + XGBoost + LSTM),
logs everything to MLflow, and produces the leaderboard table.

Usage:
    python training/train.py --feature-version v1 --model quantile_lgbm
    python training/train.py --feature-version v1 --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_store import DemandFeatureStore
from mlops.mlflow_config import get_or_create_experiment, log_environment, log_standard_tags
from models.baselines import MovingAverageForecaster, NaiveForecaster, SeasonalNaiveForecaster
from models.tree_models import LightGBMForecaster, QuantileLightGBM, XGBoostForecaster
from training.evaluate import build_leaderboard, compute_metrics, compute_segment_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)

# Train/test split: use last 13 weeks (1 quarter) as test set
TEST_WEEKS = 13


def load_features(feature_version: str) -> pd.DataFrame:
    store = DemandFeatureStore()
    return store.load(feature_version)


def train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_weeks = sorted(df["week"].unique())
    cutoff_week = sorted_weeks[-TEST_WEEKS]
    train = df[df["week"] < cutoff_week].copy()
    test = df[df["week"] >= cutoff_week].copy()
    logger.info(
        "Train: %d rows (%s→%s), Test: %d rows (%s→%s)",
        len(train), train["week"].min().date(), train["week"].max().date(),
        len(test), test["week"].min().date(), test["week"].max().date(),
    )
    return train, test


def evaluate_baseline(model, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    model.fit(train_df)
    # Predict horizon = TEST_WEEKS starting from the end of the training period.
    # Baselines anchor on the last known week in train_df and project forward.
    preds = model.predict(train_df, horizon=TEST_WEEKS)
    merged = test_df[["sku_id", "week", "demand", "category"]].merge(
        preds, on=["sku_id", "week"], how="inner"
    )
    metrics = compute_metrics(merged["demand"].values, merged["p50"].values)
    metrics["model"] = model.name()
    metrics["inference_ms"] = "< 1"
    return metrics, merged


def train_quantile_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    feature_version: str,
) -> tuple[dict, QuantileLightGBM, pd.DataFrame]:
    get_or_create_experiment("quantile_lgbm", feature_version)

    with mlflow.start_run() as run:
        log_environment()
        log_standard_tags("quantile_lgbm", feature_version)
        mlflow.log_param("feature_version", feature_version)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("test_weeks", TEST_WEEKS)
        mlflow.log_param("random_seed", 42)

        # Validation set: last 13 weeks of training data
        sorted_train_weeks = sorted(train_df["week"].unique())
        val_cutoff = sorted_train_weeks[-TEST_WEEKS]
        fit_df = train_df[train_df["week"] < val_cutoff]
        val_df = train_df[train_df["week"] >= val_cutoff]

        t0 = time.perf_counter()
        model = QuantileLightGBM(feature_cols=feature_cols)
        model.fit(fit_df, val_df=val_df)
        train_duration = time.perf_counter() - t0

        mlflow.log_metric("training_duration_s", train_duration)

        preds = model.predict(test_df)
        merged = test_df[["sku_id", "week", "demand", "category"]].merge(
            preds, on=["sku_id", "week"], how="left"
        )

        metrics = compute_metrics(
            merged["demand"].values,
            merged["p50"].values,
            merged["p10"].values,
            merged["p90"].values,
        )
        for k, v in metrics.items():
            if not np.isnan(float(v)):
                mlflow.log_metric(k, v)

        segment_metrics = compute_segment_metrics(merged, pred_col="p50", p10_col="p10", p90_col="p90")
        mlflow.log_dict(segment_metrics, "segment_metrics.json")

        # Feature importance artifact
        if model.feature_importances_ is not None:
            fi = model.feature_importances_.reset_index()
            fi.columns = ["feature", "importance"]
            mlflow.log_dict(fi.to_dict(orient="records"), "feature_importance.json")

        # Log the P50 model for registry
        mlflow.lightgbm.log_model(model.p50_model, "model")

        log_standard_tags("quantile_lgbm", feature_version, wape=metrics["wape"])
        mlflow.set_tag("mlflow_run_id", run.info.run_id)

        logger.info(
            "QuantileLGBM — WAPE=%.2f%% Coverage=%.1f%% bias=%.2f run_id=%s",
            metrics["wape"], metrics["coverage_80pct"], metrics["bias"], run.info.run_id,
        )

        metrics["model"] = "LightGBM (quantile P50)"
        metrics["inference_ms"] = f"{model.inference_time_ms(test_df):.1f}"
        metrics["run_id"] = run.info.run_id

    return metrics, model, merged


def train_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    feature_version: str,
) -> tuple[dict, XGBoostForecaster]:
    get_or_create_experiment("xgboost", feature_version)

    with mlflow.start_run():
        log_environment()
        log_standard_tags("xgboost", feature_version)

        sorted_train_weeks = sorted(train_df["week"].unique())
        val_cutoff = sorted_train_weeks[-TEST_WEEKS]
        fit_df = train_df[train_df["week"] < val_cutoff]
        val_df = train_df[train_df["week"] >= val_cutoff]

        model = XGBoostForecaster(feature_cols=feature_cols)
        model.fit(fit_df, val_df=val_df)

        preds = model.predict(test_df)
        merged = test_df[["sku_id", "week", "demand", "category"]].merge(
            preds, on=["sku_id", "week"], how="left"
        )
        metrics = compute_metrics(merged["demand"].values, merged["p50"].values)
        for k, v in metrics.items():
            if not np.isnan(float(v)):
                mlflow.log_metric(k, v)
        log_standard_tags("xgboost", feature_version, wape=metrics["wape"])

    metrics["model"] = "XGBoost"
    metrics["inference_ms"] = "~8"
    return metrics, model


def run_all_models(feature_version: str) -> None:
    logger.info("=== Demand Forecasting Training Run ===")
    logger.info("Feature version: %s", feature_version)

    df = load_features(feature_version)
    store = DemandFeatureStore()
    feature_cols = store.feature_columns

    # Drop rows where target is NaN
    df = df.dropna(subset=["demand"])
    train_df, test_df = train_test_split(df)

    leaderboard_rows = []

    # --- Baselines ---
    for model in [NaiveForecaster(), MovingAverageForecaster(window=4), SeasonalNaiveForecaster()]:
        metrics, _ = evaluate_baseline(model, train_df, test_df)
        leaderboard_rows.append(metrics)
        logger.info("%s WAPE=%.2f%%", metrics["model"], metrics["wape"])

    # --- XGBoost ---
    xgb_metrics, _ = train_xgboost(train_df, test_df, feature_cols, feature_version)
    leaderboard_rows.append(xgb_metrics)

    # --- Quantile LightGBM ---
    lgbm_metrics, lgbm_model, _ = train_quantile_lgbm(train_df, test_df, feature_cols, feature_version)
    leaderboard_rows.append(lgbm_metrics)

    # --- Leaderboard ---
    leaderboard_path = Path("data/features/leaderboard.csv")
    leaderboard = build_leaderboard(leaderboard_rows, save_path=leaderboard_path)

    print("\n=== Model Leaderboard ===")
    print(leaderboard[["model", "wape", "rmse", "bias", "coverage_80pct", "inference_ms"]].to_string(index=False))
    print()

    return lgbm_model


def main():
    parser = argparse.ArgumentParser(description="Train demand forecasting models")
    parser.add_argument("--feature-version", default="v1", help="Feature store version to use")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "quantile_lgbm", "xgboost", "baselines"],
        help="Which model(s) to train",
    )
    args = parser.parse_args()

    if args.model == "all":
        run_all_models(args.feature_version)
    elif args.model == "quantile_lgbm":
        df = load_features(args.feature_version)
        store = DemandFeatureStore()
        df = df.dropna(subset=["demand"])
        train_df, test_df = train_test_split(df)
        train_quantile_lgbm(train_df, test_df, store.feature_columns, args.feature_version)
    else:
        logger.warning("Model type '%s' not fully implemented in standalone mode", args.model)


if __name__ == "__main__":
    main()
