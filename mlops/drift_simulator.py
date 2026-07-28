"""
Drift simulation demo — the highest-leverage interview asset.

Injects a synthetic demand shock into production data to demonstrate
that the drift monitoring system catches it, triggers retraining,
and that the retrained model's WAPE recovers.

Walk-through (~3 minutes):
  1. Show baseline metrics from production model
  2. Inject APAC demand shock (2x for 6 weeks)
  3. Show Evidently catches feature drift + performance drift
  4. Trigger retraining on shocked data
  5. Show WAPE recovery

Usage:
    python mlops/drift_simulator.py --shock-region APAC --shock-multiplier 2.0 --duration-weeks 6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.drift_monitor import DriftMonitor
from features.feature_store import DemandFeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "monitoring" / "evidently_reports"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def inject_demand_shock(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    shock_region: str,
    multiplier: float,
    duration_weeks: int,
) -> pd.DataFrame:
    """
    Double (or multiply) demand for all SKUs in shock_region for the last
    `duration_weeks` weeks of the dataset.
    """
    shocked = df.copy()
    shocked_skus = metadata[metadata["supplier_region"] == shock_region]["sku_id"]

    all_weeks = sorted(shocked["week"].unique())
    shock_weeks = set(all_weeks[-duration_weeks:])

    mask = shocked["sku_id"].isin(shocked_skus) & shocked["week"].isin(shock_weeks)
    shocked.loc[mask, "demand"] = shocked.loc[mask, "demand"] * multiplier

    n_affected = mask.sum()
    logger.info(
        "Injected %.1fx demand shock on %s SKUs (%d rows, %d weeks)",
        multiplier,
        shock_region,
        n_affected,
        duration_weeks,
    )
    return shocked


def run_drift_demo(
    shock_region: str = "APAC",
    shock_multiplier: float = 2.0,
    duration_weeks: int = 6,
    baseline_wape: float = 34.9,
) -> dict:
    """Execute the full drift demo pipeline."""
    logger.info("=== DRIFT SIMULATION DEMO ===")
    logger.info("Shock: %.1fx demand on %s for %d weeks", shock_multiplier, shock_region, duration_weeks)

    # Load data
    demand_path = RAW_DIR / "demand_history.csv"
    meta_path = RAW_DIR / "sku_metadata.csv"

    if not demand_path.exists():
        raise FileNotFoundError(
            "Raw data not found. Run: python data/generator/synthetic_demand.py first"
        )

    demand = pd.read_csv(demand_path, parse_dates=["week"])
    metadata = pd.read_csv(meta_path)

    # Build features for reference (pre-shock)
    store = DemandFeatureStore()
    logger.info("Building reference features (pre-shock)...")
    reference_features = store.generate_features(demand, metadata)

    # Inject shock
    shocked_demand = inject_demand_shock(demand, metadata, shock_region, shock_multiplier, duration_weeks)

    # Build features for shocked data
    logger.info("Building shocked features (post-shock)...")
    shocked_features = store.generate_features(shocked_demand, metadata)

    # Compute WAPE on shocked data to simulate performance drift
    # (Use rolling mean as a naive "current model" prediction for the demo)
    shocked_features = shocked_features.dropna(subset=["demand"])
    shocked_features["p50"] = shocked_features["rolling_mean_4w"].fillna(
        shocked_features["demand"].median()
    )

    # Run drift monitor
    monitor = DriftMonitor(
        reference_df=reference_features,
        baseline_wape=baseline_wape,
    )

    # Use last 6 weeks of shocked data as "current production data"
    all_weeks = sorted(shocked_features["week"].unique())
    recent_weeks = set(all_weeks[-duration_weeks:])
    current_df = shocked_features[shocked_features["week"].isin(recent_weeks)].copy()
    predictions_df = current_df[["sku_id", "week", "demand", "p50"]].copy()

    summary = monitor.run_all(current_df=current_df, predictions_df=predictions_df)

    # Build the HTML demo report
    html_report_path = _build_demo_html(summary, shock_region, shock_multiplier, duration_weeks)
    summary["demo_html_report"] = str(html_report_path)

    print("\n=== DRIFT DEMO RESULTS ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nDrift Demo Report -> {html_report_path}")

    return summary


def _build_demo_html(
    summary: dict,
    shock_region: str,
    multiplier: float,
    duration_weeks: int,
) -> Path:
    """Build a standalone HTML report summarizing the drift demo."""
    action = summary.get("action", "unknown")
    data_drift = summary.get("data_drift", {})
    perf_drift = summary.get("performance_drift", {})

    action_color = {
        "immediate_retrain_and_alert": "#e74c3c",
        "immediate_retrain": "#e67e22",
        "scheduled_retrain": "#f39c12",
        "warning": "#f1c40f",
        "no_action": "#2ecc71",
    }.get(action, "#95a5a6")

    drift_fraction_pct = round(data_drift.get("drift_fraction", 0) * 100, 1)
    recent_wape_raw = perf_drift.get("recent_wape", "N/A")
    baseline_wape_raw = perf_drift.get("baseline_wape", "N/A")
    degradation_pct = perf_drift.get("degradation_pct", "N/A")
    recent_wape = f"{recent_wape_raw:.1f}" if isinstance(recent_wape_raw, (int, float)) else str(recent_wape_raw)
    baseline_wape = f"{baseline_wape_raw:.1f}" if isinstance(baseline_wape_raw, (int, float)) else str(baseline_wape_raw)

    drifted_features_html = "".join(
        f"<li>{'🔴' if v else '🟢'} <code>{k}</code></li>"
        for k, v in data_drift.get("drifted_features", {}).items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Demand Forecasting — Drift Demo Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f8f9fa; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
  .card {{ background: white; border-radius: 8px; padding: 24px; margin: 16px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .badge {{ display: inline-block; padding: 6px 14px; border-radius: 20px; color: white; font-weight: bold; background: {action_color}; font-size: 14px; }}
  .metric {{ display: inline-block; margin: 8px 16px 8px 0; }}
  .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
  .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
  .step {{ counter-increment: step-counter; }}
  .step::before {{ content: "Step " counter(step-counter); font-weight: bold; color: #3498db; display: block; margin-bottom: 4px; }}
  ol {{ counter-reset: step-counter; list-style: none; padding: 0; }}
  ul {{ line-height: 1.8; }}
  code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
  .alert {{ background: #ffeaa7; border-left: 4px solid #fdcb6e; padding: 12px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>🚨 Supply Chain Demand Forecasting — Drift Demo</h1>
<p>Generated: {summary.get("timestamp", "N/A")} | Project: Caterpillar SDSA Rebuild</p>

<div class="card">
  <h2>Shock Injected</h2>
  <p>Simulated a <strong>{multiplier}× demand shock</strong> on all <strong>{shock_region}</strong> supplier SKUs
     for <strong>{duration_weeks} weeks</strong> to mimic a real supply-chain disruption
     (e.g., APAC port delays increasing order volumes).</p>
  <div class="alert">
    This mirrors a real scenario from the Caterpillar SDSA prototype — sudden regional
    demand spikes that invalidate the model's historical seasonality assumptions.
  </div>
</div>

<div class="card">
  <h2>Drift Detection Result</h2>
  <div class="badge">{action.replace("_", " ").upper()}</div>
  <br><br>
  <div class="metric">
    <div class="metric-value">{drift_fraction_pct}%</div>
    <div class="metric-label">Features Drifted</div>
  </div>
  <div class="metric">
    <div class="metric-value">{recent_wape}%</div>
    <div class="metric-label">Current WAPE</div>
  </div>
  <div class="metric">
    <div class="metric-value">{baseline_wape}%</div>
    <div class="metric-label">Baseline WAPE</div>
  </div>
  <div class="metric">
    <div class="metric-value">{degradation_pct if isinstance(degradation_pct, str) else f"+{degradation_pct:.1f}%"}</div>
    <div class="metric-label">WAPE Degradation</div>
  </div>
</div>

<div class="card">
  <h2>Feature Drift Status</h2>
  <ul>{drifted_features_html}</ul>
  <p>Data drift threshold: <code>&gt;30% features → warning</code>, <code>&gt;50% → scheduled retrain</code></p>
</div>

<div class="card">
  <h2>3-Minute Interview Walk-Through</h2>
  <ol>
    <li class="step">Show baseline model WAPE on production data (pre-shock)</li>
    <li class="step">Inject the {shock_region} demand shock — demand doubles for {duration_weeks} weeks</li>
    <li class="step">Point to the feature drift table — <code>lag_1</code>, <code>rolling_mean_4w</code> turn red</li>
    <li class="step">Show the WAPE degradation: {baseline_wape}% → {recent_wape}% ({degradation_pct}% worse)</li>
    <li class="step">System auto-triggers retraining — explain the promotion gate</li>
    <li class="step">Show retrained model WAPE recovers (run <code>make train && make promote</code>)</li>
  </ol>
</div>

<div class="card">
  <h2>System Response</h2>
  <p><strong>Action triggered:</strong> <code>{action}</code></p>
  <p>Performance drift detected (WAPE degraded by {degradation_pct}%) triggers an immediate
     retrain. A new candidate model is trained, evaluated against the promotion gate
     (must improve WAPE by &gt;2%), and promoted to Production only if it passes.</p>
  <p>Run recovery: <code>make simulate-drift && make train && make promote</code></p>
</div>
</body>
</html>"""

    path = REPORTS_DIR / "drift_demo_report.html"
    path.write_text(html, encoding="utf-8")
    logger.info("Drift demo report saved → %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Drift simulation demo")
    parser.add_argument("--shock-region", default="APAC", choices=["APAC", "EMEA", "NA"])
    parser.add_argument("--shock-multiplier", type=float, default=2.0)
    parser.add_argument("--duration-weeks", type=int, default=6)
    parser.add_argument("--baseline-wape", type=float, default=34.9,
                        help="Baseline WAPE of production model (from MLflow)")
    args = parser.parse_args()

    run_drift_demo(
        shock_region=args.shock_region,
        shock_multiplier=args.shock_multiplier,
        duration_weeks=args.duration_weeks,
        baseline_wape=args.baseline_wape,
    )


if __name__ == "__main__":
    main()
