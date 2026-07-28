"""
Drift monitoring using Evidently AI.

Generates three separate reports:
  1. Data Drift — input feature distribution shifts
  2. Prediction Drift — P50 forecast distribution shifts
  3. Performance Drift — rolling WAPE degradation vs. promotion baseline

Concept Drift vs. Data Drift:
  - Data Drift: input feature distributions shift (e.g., lead_time_weeks
    suddenly increases due to a supplier disruption). The model sees data
    it was not trained on.
  - Concept Drift: the relationship between features and demand changes
    (e.g., a new product line makes historical seasonality patterns invalid).
    Evidenced by Performance Drift even when Data Drift is absent.

This system monitors both separately and responds differently:
  - Data Drift alone → warning + scheduled retraining
  - Performance Drift → immediate retraining trigger
  - Both → immediate retraining + alert
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "monitoring" / "evidently_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Features to monitor for data drift
DATA_DRIFT_FEATURES = [
    "lag_1",
    "rolling_mean_4w",
    "rolling_mean_12w",
    "zero_demand_ratio_12w",
    "lead_time_weeks",
    "promotion_flag",
]

# Thresholds
DATA_DRIFT_WARNING_THRESHOLD = 0.30   # >30% features drift → warning
DATA_DRIFT_RETRAIN_THRESHOLD = 0.50   # >50% features drift → trigger retraining
PRED_DRIFT_PVALUE_THRESHOLD = 0.05    # KS p-value < 0.05 → prediction drift
PERF_DRIFT_DEGRADATION_THRESHOLD = 0.15  # >15% WAPE degradation → immediate retrain


class DriftMonitor:
    """
    Runs all three drift reports against a reference (training) and
    current (production) dataset.
    """

    def __init__(self, reference_df: pd.DataFrame, baseline_wape: float):
        self.reference_df = reference_df
        self.baseline_wape = baseline_wape
        self._timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def run_all(self, current_df: pd.DataFrame, predictions_df: pd.DataFrame | None = None) -> dict:
        """
        Run all drift reports and return a combined summary.

        Parameters
        ----------
        current_df : recent production data (same schema as reference_df)
        predictions_df : DataFrame with columns [sku_id, week, p50, demand]
                         used for performance drift
        """
        summary = {
            "timestamp": self._timestamp,
            "baseline_wape": self.baseline_wape,
        }

        data_result = self.check_data_drift(current_df)
        summary["data_drift"] = data_result

        pred_result = self.check_prediction_drift(current_df)
        summary["prediction_drift"] = pred_result

        if predictions_df is not None:
            perf_result = self.check_performance_drift(predictions_df)
            summary["performance_drift"] = perf_result
        else:
            summary["performance_drift"] = {"status": "skipped", "reason": "No predictions provided"}

        summary["action"] = self._determine_action(summary)
        logger.info("Drift check complete — action: %s", summary["action"])

        # Save summary
        summary_path = REPORTS_DIR / f"drift_summary_{self._timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def check_data_drift(self, current_df: pd.DataFrame) -> dict:
        """
        KS-test for continuous features, chi-square for categorical features.
        Returns drift status per feature and overall drift fraction.
        """
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            from evidently.metrics import ColumnDriftMetric

            available_features = [f for f in DATA_DRIFT_FEATURES if f in current_df.columns and f in self.reference_df.columns]
            if not available_features:
                return self._ks_fallback(current_df)

            report = Report(metrics=[DataDriftPreset(columns=available_features)])
            report.run(
                reference_data=self.reference_df[available_features].dropna(),
                current_data=current_df[available_features].dropna(),
            )
            result = report.as_dict()
            html_path = REPORTS_DIR / f"data_drift_{self._timestamp}.html"
            report.save_html(str(html_path))

            # Extract drift fraction
            drift_results = {}
            drifted_count = 0
            for metric in result.get("metrics", []):
                if "column_name" in metric.get("result", {}):
                    col = metric["result"]["column_name"]
                    drifted = metric["result"].get("drift_detected", False)
                    drift_results[col] = drifted
                    if drifted:
                        drifted_count += 1

            drift_fraction = drifted_count / len(available_features) if available_features else 0
            status = (
                "retrain" if drift_fraction >= DATA_DRIFT_RETRAIN_THRESHOLD
                else "warning" if drift_fraction >= DATA_DRIFT_WARNING_THRESHOLD
                else "ok"
            )

            return {
                "status": status,
                "drift_fraction": drift_fraction,
                "drifted_features": drift_results,
                "html_report": str(html_path),
            }

        except ImportError:
            logger.warning("Evidently not installed — using SciPy KS-test fallback for data drift")
            return self._ks_fallback(current_df)
        except Exception as exc:
            logger.error("Data drift check failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _ks_fallback(self, current_df: pd.DataFrame) -> dict:
        """Fallback KS-test implementation when Evidently is unavailable."""
        from scipy import stats

        drift_results = {}
        available = [f for f in DATA_DRIFT_FEATURES if f in current_df.columns and f in self.reference_df.columns]

        for feature in available:
            ref_vals = self.reference_df[feature].dropna().values
            cur_vals = current_df[feature].dropna().values
            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue
            _, p_value = stats.ks_2samp(ref_vals, cur_vals)
            drift_results[feature] = p_value < 0.05

        drifted = sum(drift_results.values())
        drift_fraction = drifted / len(available) if available else 0.0
        status = (
            "retrain" if drift_fraction >= DATA_DRIFT_RETRAIN_THRESHOLD
            else "warning" if drift_fraction >= DATA_DRIFT_WARNING_THRESHOLD
            else "ok"
        )
        return {
            "status": status,
            "drift_fraction": drift_fraction,
            "drifted_features": drift_results,
            "method": "scipy_ks_fallback",
        }

    def check_prediction_drift(self, current_df: pd.DataFrame) -> dict:
        """
        KS-test on P50 forecast distribution vs. reference P50 distribution.
        """
        ref_col = "p50" if "p50" in self.reference_df else "demand"
        cur_col = "p50" if "p50" in current_df else "demand"

        ref_vals = self.reference_df[ref_col].dropna().values
        cur_vals = current_df[cur_col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            return {"status": "skipped", "reason": "Insufficient data"}

        try:
            from scipy import stats
            stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
            drifted = p_value < PRED_DRIFT_PVALUE_THRESHOLD
            return {
                "status": "drift" if drifted else "ok",
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drifted": drifted,
            }
        except ImportError:
            return {"status": "skipped", "reason": "scipy not installed"}

    def check_performance_drift(self, predictions_df: pd.DataFrame) -> dict:
        """
        Compute rolling WAPE on recent 4 weeks and compare to baseline.
        Triggers retraining if WAPE degrades by >15%.
        """
        from training.evaluate import _wape

        required_cols = {"demand", "p50"}
        if not required_cols.issubset(predictions_df.columns):
            return {"status": "skipped", "reason": "Missing demand or p50 columns"}

        # Use last 4 weeks
        if "week" in predictions_df.columns:
            sorted_weeks = sorted(predictions_df["week"].unique())[-4:]
            recent = predictions_df[predictions_df["week"].isin(sorted_weeks)]
        else:
            recent = predictions_df.tail(int(len(predictions_df) * 0.1))

        recent_wape = _wape(recent["demand"].values, recent["p50"].values)
        degradation = (recent_wape - self.baseline_wape) / (self.baseline_wape + 1e-8)

        status = "retrain" if degradation > PERF_DRIFT_DEGRADATION_THRESHOLD else "ok"
        return {
            "status": status,
            "baseline_wape": self.baseline_wape,
            "recent_wape": float(recent_wape),
            "degradation_pct": round(float(degradation) * 100, 2),
            "threshold_pct": PERF_DRIFT_DEGRADATION_THRESHOLD * 100,
        }

    def _determine_action(self, summary: dict) -> str:
        data_status = summary.get("data_drift", {}).get("status", "ok")
        perf_status = summary.get("performance_drift", {}).get("status", "ok")
        pred_status = summary.get("prediction_drift", {}).get("status", "ok")

        if perf_status == "retrain" and data_status in ("retrain", "warning"):
            return "immediate_retrain_and_alert"
        if perf_status == "retrain":
            return "immediate_retrain"
        if data_status == "retrain":
            return "scheduled_retrain"
        if data_status == "warning" or pred_status == "drift":
            return "warning"
        return "no_action"
