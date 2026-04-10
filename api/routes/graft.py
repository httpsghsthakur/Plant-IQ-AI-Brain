"""PlantIQ AI Brain - API Route: Graft Prediction"""
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Graft Prediction"])


class GraftRequest(BaseModel):
    worker_id: str
    method: str
    rootstock_variety: str
    scion_variety: str
    rootstock_age_days: int = 180
    scion_freshness_days: int = 1
    temperature: float = 23
    humidity: float = 65


@router.post("/predict/graft-success")
async def predict_graft_success(request: GraftRequest, nursery_id: str = Query(...)):
    """Predict success probability for a planned graft."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.graft.predict_success(
        data["graft"], request.worker_id, request.method,
        request.rootstock_variety, request.scion_variety,
        request.rootstock_age_days, request.scion_freshness_days,
        request.temperature, request.humidity,
    )


@router.get("/analyze/graft-workers")
async def analyze_graft_workers(nursery_id: str = Query(...)):
    """Analyze each worker's grafting performance per method."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"analysis": model_service.graft.get_worker_method_analysis(data["graft"])}


@router.get("/optimize/graft-batch")
async def optimize_graft_batch(
    nursery_id: str = Query(...),
    grafts_needed: int = Query(20),
    method: Optional[str] = Query(None),
):
    """Optimize worker assignments for a batch of grafts."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.graft.optimize_batch_assignment(
        data["graft"], grafts_needed, method
    )


@router.get("/monitor/graft/{graft_id}")
async def monitor_graft(graft_id: str, nursery_id: str = Query(...)):
    """Monitor a specific graft's progress."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.graft.monitor_graft(data["graft"], graft_id)
