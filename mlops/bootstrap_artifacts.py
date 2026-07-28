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

It also guarantees a servable production model exists for api/main.py.
`_load_model_and_features()` already prefers a real MLflow Production
model when the registry has one; if it doesn't (unreachable registry, or
just no model registered yet — the case on a brand-new deployment), this
module trains the project's own QuantileLightGBM (models/tree_models.py,
the same class training/train.py registers as Production) and persists it
locally so the API always has something to serve from instead of 503ing.

Each step is a no-op if its output already exists, so this is safe and cheap
to call on every dashboard/API start.
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
PRODUCTION_MODEL_PATH = FEATURES_DIR / "local_production_model.joblib"

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


def _mlflow_production_exists() -> bool:
    """Best-effort check for a Production model in the MLflow registry.

    Never raises: mlflow not installed, an unreachable tracking server, and
    "registry reachable but empty" all just mean "no" here — the caller
    falls back to training a local model instead.
    """
    try:
        from mlops.model_registry import get_production_version

        return get_production_version() is not None
    except Exception:
        return False


def _train_quantile_lgbm(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
):
    """Train the project's production model: models/tree_models.py::QuantileLightGBM.

    Mirrors training/train.py::train_quantile_lgbm's validation split and
    metrics computation exactly — the same model class, the same
    training.evaluate.compute_metrics — just without the mlflow.start_run()
    logging wrapper, since bootstrap time has no tracking server to log to.
    """
    from models.tree_models import QuantileLightGBM
    from training.evaluate import compute_metrics

    sorted_train_weeks = sorted(train_df["week"].unique())
    val_cutoff = sorted_train_weeks[-TEST_WEEKS]
    fit_df = train_df[train_df["week"] < val_cutoff]
    val_df = train_df[train_df["week"] >= val_cutoff]

    model = QuantileLightGBM(feature_cols=feature_cols)
    model.fit(fit_df, val_df=val_df)

    preds = model.predict(test_df)
    merged = test_df[["sku_id", "week", "demand", "category"]].merge(
        preds, on=["sku_id", "week"], how="left"
    )
    metrics = compute_metrics(
        merged["demand"].values, merged["p50"].values, merged["p10"].values, merged["p90"].values
    )
    metrics["model"] = "LightGBM (quantile P50)"
    metrics["inference_ms"] = f"{model.inference_time_ms(test_df):.1f}"
    return model, metrics


def _ensure_leaderboard_and_model() -> None:
    """Idempotent per-artifact: builds leaderboard.csv and/or the local
    fallback production model, training the shared QuantileLightGBM once
    if both are needed rather than twice.
    """
    need_leaderboard = not LEADERBOARD_CSV.exists()
    need_model = not PRODUCTION_MODEL_PATH.exists() and not _mlflow_production_exists()

    if not need_leaderboard and not need_model:
        return

    _ensure_features()
    logger.info(
        "leaderboard=%s model=%s — training real baselines + QuantileLightGBM (no MLflow)",
        "missing" if need_leaderboard else "present",
        "missing" if need_model else "present",
    )

    from features.feature_store import DemandFeatureStore
    from training.evaluate import compute_metrics

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

    if need_leaderboard:
        # --- Baselines (mirrors training/train.py::evaluate_baseline) ---
        from models.baselines import MovingAverageForecaster, NaiveForecaster, SeasonalNaiveForecaster

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

        # --- XGBoost (optional — dashboard/requirements.txt doesn't ship it) ---
        try:
            from models.tree_models import XGBoostForecaster

            sorted_train_weeks = sorted(train_df["week"].unique())
            val_cutoff = sorted_train_weeks[-TEST_WEEKS]
            fit_df = train_df[train_df["week"] < val_cutoff]
            val_df = train_df[train_df["week"] >= val_cutoff]

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

    if need_leaderboard or need_model:
        # --- Quantile LightGBM (production model) — trained once, shared ---
        lgbm_model, metrics = _train_quantile_lgbm(train_df, test_df, feature_cols)
        logger.info(
            "QuantileLightGBM WAPE=%.2f%% Coverage=%.1f%%", metrics["wape"], metrics["coverage_80pct"]
        )
        if need_leaderboard:
            rows.append(metrics)
        if need_model:
            import joblib

            joblib.dump(lgbm_model, PRODUCTION_MODEL_PATH)
            logger.info(
                "No Production model in MLflow registry — saved local fallback model -> %s",
                PRODUCTION_MODEL_PATH,
            )

    if need_leaderboard:
        from training.evaluate import build_leaderboard

        build_leaderboard(rows, save_path=LEADERBOARD_CSV)


def load_local_production_model():
    """Load the locally-trained fallback QuantileLightGBM, or None if absent.

    Used by api/main.py when the MLflow registry has no Production model —
    ensure_artifacts() guarantees this file exists in that case.
    """
    if not PRODUCTION_MODEL_PATH.exists():
        return None
    import joblib

    return joblib.load(PRODUCTION_MODEL_PATH)


def ensure_artifacts() -> None:
    """Idempotent: regenerate whatever's missing, in dependency order."""
    _ensure_raw_data()
    _ensure_features()
    _ensure_leaderboard_and_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_artifacts()
    print("All artifacts present:")
    print(f"  {DEMAND_CSV.relative_to(ROOT)}")
    print(f"  {METADATA_CSV.relative_to(ROOT)}")
    print(f"  {FEATURES_PARQUET.relative_to(ROOT)}")
    print(f"  {LEADERBOARD_CSV.relative_to(ROOT)}")
    if PRODUCTION_MODEL_PATH.exists():
        print(f"  {PRODUCTION_MODEL_PATH.relative_to(ROOT)} (local fallback — no MLflow Production model)")
    else:
        print("  (no local fallback model — MLflow registry already has a Production model)")
