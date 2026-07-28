"""Tests for the feature store."""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_store import DemandFeatureStore, CATEGORY_MAP, SUPPLIER_REGION_MAP


@pytest.fixture
def tiny_dataset():
    """Minimal 3-SKU, 60-week dataset for fast unit tests."""
    np.random.seed(0)
    skus = ["PART-00001", "PART-00002", "PART-00003"]
    weeks = pd.date_range("2022-01-03", periods=60, freq="W-MON")

    rows = []
    for sku in skus:
        for week in weeks:
            rows.append({
                "sku_id": sku,
                "week": week,
                "demand": float(np.random.randint(0, 50)),
                "stockout_flag": int(np.random.random() < 0.05),
                "holiday_flag": int(np.random.random() < 0.08),
                "promotion_flag": int(np.random.random() < 0.12),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_metadata():
    return pd.DataFrame({
        "sku_id": ["PART-00001", "PART-00002", "PART-00003"],
        "category": ["fast_moving", "intermittent", "seasonal"],
        "supplier_region": ["APAC", "EMEA", "NA"],
        "unit_cost": [100.0, 500.0, 250.0],
        "lead_time_weeks": [2, 6, 3],
    })


def test_generate_features_shape(tiny_dataset, tiny_metadata):
    store = DemandFeatureStore()
    features = store.generate_features(tiny_dataset, tiny_metadata)
    assert len(features) == len(tiny_dataset)
    assert "lag_1" in features.columns
    assert "rolling_mean_4w" in features.columns
    assert "zero_demand_ratio_12w" in features.columns
    assert "demand_adjusted" in features.columns


def test_lag_features_present(tiny_dataset, tiny_metadata):
    store = DemandFeatureStore()
    features = store.generate_features(tiny_dataset, tiny_metadata)
    for lag in [1, 2, 4, 8, 12, 26, 52]:
        assert f"lag_{lag}" in features.columns, f"Missing lag_{lag}"


def test_category_encodings(tiny_dataset, tiny_metadata):
    store = DemandFeatureStore()
    features = store.generate_features(tiny_dataset, tiny_metadata)
    assert "category_enc" in features.columns
    assert "supplier_region_enc" in features.columns
    for cat in CATEGORY_MAP:
        assert f"cat_{cat}" in features.columns
    for region in SUPPLIER_REGION_MAP:
        assert f"region_{region}" in features.columns


def test_demand_adjusted_fills_stockout(tiny_dataset, tiny_metadata):
    store = DemandFeatureStore()
    features = store.generate_features(tiny_dataset, tiny_metadata)
    # demand_adjusted should never be NaN
    assert features["demand_adjusted"].isna().sum() == 0


def test_feature_columns_list(tiny_dataset, tiny_metadata):
    store = DemandFeatureStore()
    features = store.generate_features(tiny_dataset, tiny_metadata)
    for col in store.feature_columns:
        assert col in features.columns, f"Feature column '{col}' missing from output"


def test_save_and_load(tmp_path, tiny_dataset, tiny_metadata, monkeypatch):
    from features import feature_store as fs_module
    monkeypatch.setattr(fs_module, "FEATURES_DIR", tmp_path)

    store = DemandFeatureStore()
    features = store.generate_features(tiny_dataset, tiny_metadata)
    store.save(features, "test_v1")

    loaded = store.load("test_v1")
    assert len(loaded) == len(features)
    assert list(loaded.columns) == list(features.columns)


def test_get_schema(tmp_path, tiny_dataset, tiny_metadata, monkeypatch):
    from features import feature_store as fs_module
    monkeypatch.setattr(fs_module, "FEATURES_DIR", tmp_path)

    store = DemandFeatureStore()
    features = store.generate_features(tiny_dataset, tiny_metadata)
    store.save(features, "v1")

    schema = store.get_schema()
    assert isinstance(schema, dict)
    assert len(schema) > 0
