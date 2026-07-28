"""Tests for evaluation metrics."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.evaluate import compute_metrics, compute_segment_metrics, _wape, _smape


def test_perfect_forecast():
    y = np.array([10.0, 20.0, 30.0])
    m = compute_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["wape"] == pytest.approx(0.0)
    assert m["bias"] == pytest.approx(0.0)


def test_constant_over_forecast():
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([15.0, 15.0, 15.0])
    m = compute_metrics(y_true, y_pred)
    assert m["bias"] == pytest.approx(5.0)
    assert m["wape"] == pytest.approx(50.0)


def test_wape_denominator_safety():
    # All-zero actuals should not raise ZeroDivisionError
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(y_true, y_pred)
    assert np.isfinite(m["wape"])


def test_mape_zero_handling():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 10.0, 20.0])
    m = compute_metrics(y_true, y_pred)
    # MAPE should ignore zero-denominator entries
    assert np.isfinite(m["mape"])


def test_coverage_metric():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_p10 = np.array([5.0, 15.0, 25.0, 35.0, 45.0])
    y_p90 = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
    m = compute_metrics(y_true, y_pred=y_true, y_p10=y_p10, y_p90=y_p90)
    assert m["coverage_80pct"] == pytest.approx(100.0)


def test_interval_width():
    y_p10 = np.array([0.0, 10.0])
    y_p90 = np.array([20.0, 30.0])
    m = compute_metrics(
        y_true=np.array([10.0, 20.0]),
        y_pred=np.array([10.0, 20.0]),
        y_p10=y_p10,
        y_p90=y_p90,
    )
    assert m["interval_width"] == pytest.approx(20.0)


def test_nan_handling():
    y_true = np.array([10.0, np.nan, 30.0])
    y_pred = np.array([10.0, 20.0, np.nan])
    m = compute_metrics(y_true, y_pred)
    # Only the first row is valid after dropping NaN in both arrays
    assert m["mae"] == pytest.approx(0.0)


def test_segment_metrics_structure():
    import pandas as pd
    df = pd.DataFrame({
        "demand": [10.0, 20.0, 5.0, 50.0],
        "p50": [10.0, 18.0, 5.0, 55.0],
        "category": ["fast_moving", "fast_moving", "intermittent", "seasonal"],
    })
    result = compute_segment_metrics(df)
    assert "fast_moving" in result
    assert "intermittent" in result
    assert "wape" in result["fast_moving"]
