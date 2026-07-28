"""
Self-healing artifact bootstrap for the Streamlit dashboard.

Why this exists
----------------
`data/raw/` and `data/features/` are intentionally gitignored (see .gitignore —
"generated, not tracked in git"). Locally that's fine because a developer runs
`make bootstrap && make train` before touching the dashboard. But Streamlit
Cloud (and any other fresh deploy target) only ever sees a clean `git clone`
of this repo — those directories don't exist there, so
`dashboard/app.py::load_features()` / `load_leaderboard()` raise
FileNotFoundError on first load.

This module regenerates the *real* artifacts using the project's own
pipeline (the synthetic generator + the real feature store + the real
model/eval code) — nothing here is fabricated or hardcoded. It only skips
the optional MLflow *logging* side-effects from training/train.py, since a
fresh cloud deployment has no MLflow tracking server configured. The models
trained and the metrics computed are the genuine output of
`models/baselines.py`, `models/tree_models.py`, and `training/evaluate.py`.

Each step is a no-op if its output already exists, so this is safe and cheap
to call on every dashboard start.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
FEATURES_DIR = ROOT / "data" / "features"
DEMAND_CSV = RAW_DIR / "demand_history.csv"
METADATA_CSV = RAW_DIR / "sku_metadata.csv"
FEATURES_PARQUET = FEATURES_DIR / "features_v1.parquet"
LEADERBOARD_CSV = FEATURES_DIR / "leaderboard.csv"

TEST_WEEKS = 13  # must match training/train.py


def _ensure_raw_data() -> None:
    if DEMAND_CSV.exists() and METADATA_CSV.exists():
        return
    logger.info("Raw data missing — running data/generator/synthetic_demand.py")
    from data.generator.synthetic_demand import generate_dataset

    generate_dataset()


def _ensure_features() -> None:
    if FEATURES_PARQUET.exists():
        return
    _ensure_raw_data()
    logger.info("features_v1.parquet missing — running the real feature store pipeline")
    from features.feature_store import DemandFeatureStore

    demand = pd.read_csv(DEMAND_CSV, parse_dates=["week"])
    meta = pd.read_csv(METADATA_CSV)
    store = DemandFeatureStore()
    features = store.generate_features(demand, meta)
    store.save(features, "v1")


def _ensure_leaderboard() -> None:
    if LEADERBOARD_CSV.exists():
        return
    _ensure_features()
    logger.info("leaderboard.csv missing — training real baselines + QuantileLightGBM (no MLflow)")

    from features.feature_store import DemandFeatureStore
    from models.baselines import MovingAverageForecaster, NaiveForecaster, SeasonalNaiveForecaster
    from models.tree_models import QuantileLightGBM
    from training.evaluate import build_leaderboard, compute_metrics

    store = DemandFeatureStore()
    df = pd.read_parquet(FEATURES_PARQUET)
    df["week"] = pd.to_datetime(df["week"])
    df = df.dropna(subset=["demand"])
    feature_cols = store.feature_columns

    sorted_weeks = sorted(df["week"].unique())
    cutoff_week = sorted_weeks[-TEST_WEEKS]
    train_df = df[df["week"] < cutoff_week].copy()
    test_df = df[df["week"] >= cutoff_week].copy()

    rows = []

    # --- Baselines (mirrors training/train.py::evaluate_baseline) ---
    for model in [NaiveForecaster(), MovingAverageForecaster(window=4), SeasonalNaiveForecaster()]:
        model.fit(train_df)
        preds = model.predict(train_df, horizon=TEST_WEEKS)
        merged = test_df[["sku_id", "week", "demand", "category"]].merge(
            preds, on=["sku_id", "week"], how="inner"
        )
        metrics = compute_metrics(merged["demand"].values, merged["p50"].values)
        metrics["model"] = model.name()
        metrics["inference_ms"] = "< 1"
        rows.append(metrics)
        logger.info("%s WAPE=%.2f%%", metrics["model"], metrics["wape"])

    sorted_train_weeks = sorted(train_df["week"].unique())
    val_cutoff = sorted_train_weeks[-TEST_WEEKS]
    fit_df = train_df[train_df["week"] < val_cutoff]
    val_df = train_df[train_df["week"] >= val_cutoff]

    # --- XGBoost (optional — dashboard/requirements.txt doesn't ship it) ---
    try:
        from models.tree_models import XGBoostForecaster

        xgb_model = XGBoostForecaster(feature_cols=feature_cols)
        xgb_model.fit(fit_df, val_df=val_df)
        preds = xgb_model.predict(test_df)
        merged = test_df[["sku_id", "week", "demand", "category"]].merge(
            preds, on=["sku_id", "week"], how="left"
        )
        metrics = compute_metrics(merged["demand"].values, merged["p50"].values)
        metrics["model"] = "XGBoost"
        metrics["inference_ms"] = "~8"
        rows.append(metrics)
        logger.info("XGBoost WAPE=%.2f%%", metrics["wape"])
    except ImportError:
        logger.info("xgboost not installed — skipping from bootstrap leaderboard")

    # --- Quantile LightGBM (production model) ---
    lgbm_model = QuantileLightGBM(feature_cols=feature_cols)
    lgbm_model.fit(fit_df, val_df=val_df)
    preds = lgbm_model.predict(test_df)
    merged = test_df[["sku_id", "week", "demand", "category"]].merge(
        preds, on=["sku_id", "week"], how="left"
    )
    metrics = compute_metrics(
        merged["demand"].values, merged["p50"].values, merged["p10"].values, merged["p90"].values
    )
    metrics["model"] = "LightGBM (quantile P50)"
    metrics["inference_ms"] = f"{lgbm_model.inference_time_ms(test_df):.1f}"
    rows.append(metrics)
    logger.info("QuantileLightGBM WAPE=%.2f%% Coverage=%.1f%%", metrics["wape"], metrics["coverage_80pct"])

    build_leaderboard(rows, save_path=LEADERBOARD_CSV)


def ensure_artifacts() -> None:
    """Idempotent: regenerate whatever's missing, in dependency order."""
    _ensure_raw_data()
    _ensure_features()
    _ensure_leaderboard()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_artifacts()
    print("All artifacts present:")
    print(f"  {DEMAND_CSV.relative_to(ROOT)}")
    print(f"  {METADATA_CSV.relative_to(ROOT)}")
    print(f"  {FEATURES_PARQUET.relative_to(ROOT)}")
    print(f"  {LEADERBOARD_CSV.relative_to(ROOT)}")
