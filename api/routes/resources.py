"""PlantIQ AI Brain - API Route: Resource Optimization"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Resource Optimization"])


@router.get("/optimize/water")
async def optimize_water(nursery_id: str = Query(...)):
    """Get water usage optimization per zone."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"zones": model_service.resource.optimize_water(
        data["sensor"], data["inventory"]
    )}


@router.get("/optimize/fertilizer")
async def optimize_fertilizer(nursery_id: str = Query(...)):
    """Get fertilizer plan per zone and plant stage."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"plans": model_service.resource.get_fertilizer_plan(data["plant_inventory"])}


@router.get("/predict/inventory")
async def predict_inventory(nursery_id: str = Query(...)):
    """Predict inventory levels and reorder alerts."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"forecasts": model_service.resource.predict_inventory(data["inventory"])}


@router.get("/optimize/resources/summary")
async def resource_summary(nursery_id: str = Query(...)):
    """Get overall resource optimization summary."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.resource.get_resource_summary(
        data["inventory"], data["sensor"], data["plant_inventory"]
    )
