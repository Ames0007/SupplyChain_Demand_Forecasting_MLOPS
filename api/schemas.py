"""Pydantic request/response models for the forecasting API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    sku_id: str = Field(..., description="SKU identifier (e.g. PART-10482)")
    horizon_weeks: int = Field(4, ge=1, le=52, description="Number of weeks to forecast")
    include_intervals: bool = Field(True, description="Include P10/P90 prediction intervals")

    @field_validator("sku_id")
    @classmethod
    def sku_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sku_id cannot be empty")
        return v.strip()


class BatchForecastRequest(BaseModel):
    sku_ids: List[str] = Field(..., min_length=1, max_length=500)
    horizon_weeks: int = Field(4, ge=1, le=52)
    include_intervals: bool = True
    # TODO: async batch via Celery+Redis is the production upgrade path
    # for large batches (>500 SKUs or horizon>13 weeks)


class WeekForecast(BaseModel):
    week: str
    p10: Optional[float] = None
    p50: float
    p90: Optional[float] = None


class ForecastResponse(BaseModel):
    sku_id: str
    horizon_weeks: int
    model_version: str
    feature_version: str
    generated_at: str
    drift_alert: bool
    sku_segment: Optional[str] = None
    expected_wape: Optional[str] = None
    forecasts: List[WeekForecast]


class BatchForecastResponse(BaseModel):
    results: List[ForecastResponse]
    total_skus: int
    model_version: str
    generated_at: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_version: Optional[str]
    feature_version: Optional[str]
    uptime_seconds: float
    last_drift_check: Optional[str]


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    version: Optional[str]
    stage: str
    run_id: Optional[str]
    wape: Optional[str]
    feature_version: Optional[str]
    registered_at: Optional[str]


class DriftReportResponse(BaseModel):
    timestamp: str
    data_drift_status: str
    data_drift_fraction: Optional[float]
    prediction_drift_status: str
    performance_drift_status: str
    current_wape: Optional[float]
    baseline_wape: Optional[float]
    action: str
    html_report_url: Optional[str]
