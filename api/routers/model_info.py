"""GET /model-info — current production model metadata from MLflow."""

from fastapi import APIRouter, HTTPException
from api.schemas import ModelInfoResponse

router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info():
    """Returns metadata for the current production model from MLflow registry."""
    try:
        from mlops.model_registry import get_client, get_production_version
        from mlops.mlflow_config import REGISTERED_MODEL_NAME

        version = get_production_version()
        if version is None:
            return ModelInfoResponse(
                model_name=REGISTERED_MODEL_NAME,
                version=None,
                stage="none",
                run_id=None,
                wape=None,
                feature_version=None,
                registered_at=None,
            )

        client = get_client()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if not versions:
            raise HTTPException(status_code=404, detail="No production model found")

        mv = versions[0]
        run = client.get_run(mv.run_id)
        return ModelInfoResponse(
            model_name=REGISTERED_MODEL_NAME,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id,
            wape=run.data.tags.get("wape"),
            feature_version=run.data.tags.get("feature_version"),
            registered_at=str(mv.creation_timestamp),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")
