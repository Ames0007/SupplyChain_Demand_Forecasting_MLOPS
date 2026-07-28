"""API integration tests using TestClient (no live MLflow required)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_app_state(tmp_path):
    """Create a minimal app state with fake features and a mock model."""
    weeks = pd.date_range("2024-01-01", periods=20, freq="W-MON")
    features_df = pd.DataFrame({
        "sku_id": ["PART-00001"] * 20,
        "week": weeks,
        "demand": np.random.randint(10, 100, 20).astype(float),
        "demand_adjusted": np.random.randint(10, 100, 20).astype(float),
        **{f"lag_{lag}": np.random.rand(20) * 50 for lag in [1, 2, 4, 8, 12, 26, 52]},
        "rolling_mean_4w": np.random.rand(20) * 50,
        "rolling_mean_12w": np.random.rand(20) * 50,
        "rolling_std_4w": np.random.rand(20) * 10,
        "rolling_std_12w": np.random.rand(20) * 10,
        "rolling_min_4w": np.random.rand(20) * 20,
        "rolling_max_4w": np.random.rand(20) * 80,
        "demand_change_1w": np.zeros(20),
        "demand_change_4w": np.zeros(20),
        "week_of_year": np.arange(20) % 52 + 1,
        "month": [1] * 20,
        "quarter": [1] * 20,
        "is_q1": [1] * 20,
        "is_q4": [0] * 20,
        "holiday_flag": [0] * 20,
        "promotion_flag": [0] * 20,
        "lead_time_weeks": [3] * 20,
        "log_unit_cost": [5.0] * 20,
        "zero_demand_ratio_12w": [0.0] * 20,
        "supplier_region_enc": [0] * 20,
        "category_enc": [0] * 20,
        "cat_fast_moving": [1] * 20,
        "cat_slow_moving": [0] * 20,
        "cat_intermittent": [0] * 20,
        "cat_seasonal": [0] * 20,
        "region_APAC": [1] * 20,
        "region_EMEA": [0] * 20,
        "region_NA": [0] * 20,
    })

    metadata_df = pd.DataFrame({
        "sku_id": ["PART-00001"],
        "category": ["fast_moving"],
        "supplier_region": ["APAC"],
        "unit_cost": [100.0],
        "lead_time_weeks": [3],
    })

    # Mock model that returns quantile predictions
    mock_model = MagicMock()
    def fake_predict(df):
        result = df[["sku_id", "week"]].copy()
        n = len(result)
        result["p10"] = np.random.rand(n) * 50
        result["p50"] = np.random.rand(n) * 100 + 50
        result["p90"] = np.random.rand(n) * 50 + 150
        return result
    mock_model.predict.side_effect = fake_predict

    return {
        "model": mock_model,
        "features_df": features_df,
        "metadata_df": metadata_df,
        "model_version": "v1",
        "feature_version": "v1",
        "drift_alert": False,
        "startup_time": 0,
        "uptime_seconds": 0,
        "last_drift_check": None,
    }


@pytest.fixture
def client(mock_app_state):
    from fastapi.testclient import TestClient
    import api.main as main_module
    main_module._APP_STATE.update(mock_app_state)
    # Skip lifespan loading
    from api.main import app
    return TestClient(app, raise_server_exceptions=True)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_forecast_endpoint(client):
    resp = client.post("/forecast", json={"sku_id": "PART-00001", "horizon_weeks": 4})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sku_id"] == "PART-00001"
    assert len(data["forecasts"]) == 4
    for fc in data["forecasts"]:
        assert "p50" in fc
        assert fc["p50"] >= 0


def test_forecast_unknown_sku(client):
    resp = client.post("/forecast", json={"sku_id": "UNKNOWN-SKU", "horizon_weeks": 4})
    assert resp.status_code == 404


def test_forecast_invalid_horizon(client):
    resp = client.post("/forecast", json={"sku_id": "PART-00001", "horizon_weeks": 0})
    assert resp.status_code == 422


def test_batch_forecast(client):
    resp = client.post(
        "/forecast/batch",
        json={"sku_ids": ["PART-00001"], "horizon_weeks": 2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_skus"] == 1


def test_model_version_header(client):
    resp = client.get("/health")
    assert "x-model-version" in resp.headers


def test_drift_alert_header(client):
    resp = client.get("/health")
    assert "x-drift-alert" in resp.headers
    assert resp.headers["x-drift-alert"] in ("true", "false")
