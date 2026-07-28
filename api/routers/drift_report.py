"""GET /drift-report — latest Evidently drift report summary."""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from api.schemas import DriftReportResponse

router = APIRouter()
REPORTS_DIR = Path(__file__).parent.parent.parent / "monitoring" / "evidently_reports"


@router.get("/drift-report", response_model=DriftReportResponse, tags=["Monitoring"])
async def drift_report():
    """Returns the most recent drift report summary as JSON."""
    summaries = sorted(REPORTS_DIR.glob("drift_summary_*.json"))
    if not summaries:
        raise HTTPException(status_code=404, detail="No drift reports found. Run drift_monitor.py first.")

    latest = summaries[-1]
    with open(latest) as f:
        data = json.load(f)

    data_drift = data.get("data_drift", {})
    perf_drift = data.get("performance_drift", {})
    pred_drift = data.get("prediction_drift", {})

    html_report = data_drift.get("html_report")
    html_url = f"/drift-reports/{Path(html_report).name}" if html_report else None

    return DriftReportResponse(
        timestamp=data.get("timestamp", ""),
        data_drift_status=data_drift.get("status", "unknown"),
        data_drift_fraction=data_drift.get("drift_fraction"),
        prediction_drift_status=pred_drift.get("status", "unknown"),
        performance_drift_status=perf_drift.get("status", "unknown"),
        current_wape=perf_drift.get("recent_wape"),
        baseline_wape=perf_drift.get("baseline_wape"),
        action=data.get("action", "unknown"),
        html_report_url=html_url,
    )
