"""PlantIQ AI Brain - API Route: Worker Performance"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Worker Performance"])


@router.get("/analyze/worker/{worker_id}")
async def analyze_worker(worker_id: str, nursery_id: str = Query(...), period_days: int = Query(30)):
    """Get performance scorecard for a specific worker."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.worker.get_worker_scorecard(
        data["attendance"], data["task"], worker_id, period_days
    )


@router.get("/analyze/workers")
async def analyze_all_workers(nursery_id: str = Query(...), period_days: int = Query(30)):
    """Get performance scores for all workers."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    scores = model_service.worker.get_all_scores(
        data["attendance"], data["task"], period_days
    )
    return {"workers": sorted(scores, key=lambda x: x["score"], reverse=True)}


@router.get("/predict/absenteeism")
async def predict_absenteeism(nursery_id: str = Query(...)):
    """Predict workers likely to be absent."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"predictions": model_service.worker.predict_absenteeism(data["attendance"])}


@router.get("/detect/burnout")
@router.get("/predict/worker-burnout")
async def detect_burnout(nursery_id: str = Query(...)):
    """Detect workers at risk of burnout."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"worker_predictions": model_service.worker.detect_burnout_risks(
        data["attendance"], data["task"]
    )}


@router.get("/analyze/workload")
async def analyze_workload(nursery_id: str = Query(...), period_days: int = Query(7)):
    """Analyze and recommend workload balancing."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.worker.get_workload_balance(data["task"], period_days)


@router.get("/training-needs")
async def training_needs(nursery_id: str = Query(...)):
    """Identify worker training needs."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"needs": model_service.worker.get_training_needs(
        data["task"], data["graft"]
    )}
