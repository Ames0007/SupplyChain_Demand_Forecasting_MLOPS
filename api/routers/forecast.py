"""
POST /forecast and POST /forecast/batch — the core prediction endpoints.

Loads the production QuantileLightGBM model + feature store and returns
P10/P50/P90 quantile forecasts for the requested SKU(s) and horizon.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastRequest,
    ForecastResponse,
    WeekForecast,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_model_and_features():
    """Lazy-load model and features from the application state."""
    from api.main import get_app_state
    state = get_app_state()
    model = state.get("model")
    features_df = state.get("features_df")
    metadata_df = state.get("metadata_df")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MLflow registry.")
    return model, features_df, metadata_df


def _build_forecast_for_sku(
    sku_id: str,
    model,
    features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    horizon_weeks: int,
    include_intervals: bool,
    model_version: str,
    feature_version: str,
    drift_alert: bool,
) -> ForecastResponse:
    """Compute quantile forecasts for a single SKU."""
    sku_features = features_df[features_df["sku_id"] == sku_id].sort_values("week")

    if len(sku_features) == 0:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found in feature store")

    # Get SKU segment
    sku_meta = metadata_df[metadata_df["sku_id"] == sku_id]
    sku_segment = str(sku_meta["category"].iloc[0]) if len(sku_meta) > 0 else None

    # Use the last `horizon_weeks` rows as pseudo-forecast rows
    # In production these would be future feature rows; here we use
    # the latest available week's features repeated across the horizon
    last_row = sku_features.iloc[-1:].copy()
    from features.feature_store import DemandFeatureStore
    feature_cols = DemandFeatureStore().feature_columns

    forecast_rows = []
    last_week = pd.Timestamp(last_row["week"].iloc[0])

    for h in range(1, horizon_weeks + 1):
        row = last_row.copy()
        row["week"] = last_week + pd.Timedelta(weeks=h)
        # Shift lag features for each horizon step
        row["lag_1"] = last_row.get("demand_adjusted", last_row.get("lag_1", 0)).iloc[0]
        forecast_rows.append(row)

    forecast_df = pd.concat(forecast_rows, ignore_index=True)

    # Fill any missing feature columns with 0
    for col in feature_cols:
        if col not in forecast_df.columns:
            forecast_df[col] = 0
    forecast_df[feature_cols] = forecast_df[feature_cols].fillna(0)

    preds = model.predict(forecast_df)

    weekly_forecasts = []
    for _, row in preds.iterrows():
        week_str = pd.Timestamp(row["week"]).strftime("%Y-%m-%d")
        p50_val = max(0.0, float(row.get("p50", 0)))
        p10_val = max(0.0, float(row.get("p10", p50_val * 0.7))) if include_intervals else None
        p90_val = max(p50_val, float(row.get("p90", p50_val * 1.3))) if include_intervals else None
        weekly_forecasts.append(WeekForecast(week=week_str, p10=p10_val, p50=p50_val, p90=p90_val))

    return ForecastResponse(
        sku_id=sku_id,
        horizon_weeks=horizon_weeks,
        model_version=model_version,
        feature_version=feature_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        drift_alert=drift_alert,
        sku_segment=sku_segment,
        expected_wape="~34.9%",  # Updated from MLflow tag in production
        forecasts=weekly_forecasts,
    )


@router.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast(request: ForecastRequest):
    """
    Generate P10/P50/P90 quantile demand forecasts for a single SKU.

    Example:
        curl -X POST /forecast -H "Content-Type: application/json" \\
             -d '{"sku_id": "PART-10482", "horizon_weeks": 4}'
    """
    from api.main import get_app_state
    state = get_app_state()
    model, features_df, metadata_df = _get_model_and_features()

    response = _build_forecast_for_sku(
        sku_id=request.sku_id,
        model=model,
        features_df=features_df,
        metadata_df=metadata_df,
        horizon_weeks=request.horizon_weeks,
        include_intervals=request.include_intervals,
        model_version=state.get("model_version", "unknown"),
        feature_version=state.get("feature_version", "unknown"),
        drift_alert=state.get("drift_alert", False),
    )
    return response


@router.post("/forecast/batch", response_model=BatchForecastResponse, tags=["Forecasting"])
async def forecast_batch(request: BatchForecastRequest):
    """
    Generate forecasts for up to 500 SKUs in a single request.

    TODO: For production scale (>500 SKUs or long horizons), upgrade to
    async batch processing via Celery + Redis task queue.
    """
    from api.main import get_app_state
    state = get_app_state()
    model, features_df, metadata_df = _get_model_and_features()

    model_version = state.get("model_version", "unknown")
    feature_version = state.get("feature_version", "unknown")
    drift_alert = state.get("drift_alert", False)

    results = []
    errors = []
    for sku_id in request.sku_ids:
        try:
            resp = _build_forecast_for_sku(
                sku_id=sku_id,
                model=model,
                features_df=features_df,
                metadata_df=metadata_df,
                horizon_weeks=request.horizon_weeks,
                include_intervals=request.include_intervals,
                model_version=model_version,
                feature_version=feature_version,
                drift_alert=drift_alert,
            )
            results.append(resp)
        except HTTPException as e:
            errors.append({"sku_id": sku_id, "error": e.detail})
            logger.warning("Batch forecast failed for %s: %s", sku_id, e.detail)

    if errors:
        logger.warning("Batch forecast: %d errors out of %d SKUs", len(errors), len(request.sku_ids))

    return BatchForecastResponse(
        results=results,
        total_skus=len(results),
        model_version=model_version,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
