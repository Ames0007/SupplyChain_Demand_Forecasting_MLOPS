"""GET /health — service liveness and model version."""

from fastapi import APIRouter
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health(request=None):
    """Service health check. Returns model version, uptime, last drift check."""
    from api.main import get_app_state
    state = get_app_state()
    return HealthResponse(
        status="ok",
        model_version=state.get("model_version"),
        feature_version=state.get("feature_version"),
        uptime_seconds=state.get("uptime_seconds", 0.0),
        last_drift_check=state.get("last_drift_check"),
    )
