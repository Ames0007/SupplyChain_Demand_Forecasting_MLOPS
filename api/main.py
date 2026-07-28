"""
FastAPI entrypoint for the demand forecasting service.

Serves quantile (P10/P50/P90) forecasts from the production
QuantileLightGBM model registered in MLflow.

Middleware:
  - Request logging with latency
  - x-model-version header on every response
  - x-drift-alert: true/false header
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Global application state — set during lifespan startup
_APP_STATE: dict = {}


def get_app_state() -> dict:
    return _APP_STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bootstrap data artifacts, then load model and features on startup; clean up on shutdown."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    startup_time = time.time()
    logger.info("Starting demand forecasting API...")

    try:
        from mlops.bootstrap_artifacts import ensure_artifacts

        ensure_artifacts()
    except Exception as exc:
        logger.error("Artifact bootstrap failed: %s", exc)
        logger.warning("Continuing startup — _load_model_and_features will run in degraded mode")

    try:
        _load_model_and_features()
    except Exception as exc:
        logger.error("Failed to load model/features on startup: %s", exc)
        logger.warning("API starting in degraded mode — /forecast will return 503")

    _APP_STATE["startup_time"] = startup_time
    logger.info("API startup complete. Model: %s", _APP_STATE.get("model_version", "none"))

    yield

    logger.info("Shutting down demand forecasting API...")


def _load_model_and_features() -> None:
    """Load the production quantile model and feature store into memory."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Load features
    from features.feature_store import DemandFeatureStore

    store = DemandFeatureStore()
    try:
        features_df, feature_version = store.get_latest()
        _APP_STATE["features_df"] = features_df
        _APP_STATE["feature_version"] = feature_version
        logger.info("Loaded features version %s (%d rows)", feature_version, len(features_df))
    except FileNotFoundError:
        logger.warning("No feature files found — run generate_features first")
        _APP_STATE["features_df"] = pd.DataFrame()
        _APP_STATE["feature_version"] = "none"

    # Load metadata
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    meta_path = raw_dir / "sku_metadata.csv"
    if meta_path.exists():
        _APP_STATE["metadata_df"] = pd.read_csv(meta_path)
    else:
        _APP_STATE["metadata_df"] = pd.DataFrame(columns=["sku_id", "category", "supplier_region"])

    # Load model from MLflow or local fallback
    try:
        from mlops.model_registry import get_production_version
        from mlops.mlflow_config import REGISTERED_MODEL_NAME
        import mlflow

        version = get_production_version()
        if version:
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
            lgb_model = mlflow.lightgbm.load_model(model_uri)

            # Wrap in QuantileLightGBM interface for predict()
            from models.tree_models import QuantileLightGBM
            q_model = QuantileLightGBM(feature_cols=store.feature_cols)
            q_model._models["p50"] = lgb_model
            # P10/P90 will be approximated if only P50 loaded — warn once
            logger.warning("Only P50 model loaded from registry. P10/P90 will be approximated.")
            _APP_STATE["model"] = q_model
            _APP_STATE["model_version"] = f"v{version}"
        else:
            logger.warning("No Production model in MLflow registry — model not loaded")
            _APP_STATE["model"] = None
            _APP_STATE["model_version"] = None

    except Exception as exc:
        logger.error("MLflow model load failed: %s", exc)
        _APP_STATE["model"] = None
        _APP_STATE["model_version"] = None

    # Check drift alert status
    from pathlib import Path as _Path
    reports_dir = _Path(__file__).parent.parent / "monitoring" / "evidently_reports"
    summaries = sorted(reports_dir.glob("drift_summary_*.json"))
    if summaries:
        import json
        with open(summaries[-1]) as f:
            drift_data = json.load(f)
        action = drift_data.get("action", "no_action")
        _APP_STATE["drift_alert"] = action in ("immediate_retrain", "immediate_retrain_and_alert")
        _APP_STATE["last_drift_check"] = drift_data.get("timestamp")
    else:
        _APP_STATE["drift_alert"] = False
        _APP_STATE["last_drift_check"] = None


app = FastAPI(
    title="Supply Chain Demand Forecasting API",
    description=(
        "Production-grade demand forecasting for spare parts. "
        "Rebuilt from the Caterpillar SDSA prototype (2023) with full MLOps. "
        "Serves P10/P50/P90 quantile forecasts from LightGBM with drift monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_model_headers(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    model_version = _APP_STATE.get("model_version") or "none"
    drift_alert = str(_APP_STATE.get("drift_alert", False)).lower()

    response.headers["x-model-version"] = model_version
    response.headers["x-drift-alert"] = drift_alert
    response.headers["x-latency-ms"] = f"{latency_ms:.1f}"

    logger.info(
        "method=%s path=%s status=%d latency_ms=%.1f model=%s",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        model_version,
    )
    return response


@app.get("/uptime")
async def uptime():
    startup_time = _APP_STATE.get("startup_time", time.time())
    _APP_STATE["uptime_seconds"] = time.time() - startup_time
    return {"uptime_seconds": _APP_STATE["uptime_seconds"]}


# Mount static drift reports
reports_dir = Path(__file__).parent.parent / "monitoring" / "evidently_reports"
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/drift-reports", StaticFiles(directory=str(reports_dir)), name="drift-reports")

# Register routers
from api.routers import forecast, health, model_info, drift_report  # noqa: E402

app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(forecast.router)
app.include_router(drift_report.router)


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
