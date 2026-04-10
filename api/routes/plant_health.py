"""PlantIQ AI Brain - API Route: Plant Health"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Plant Health"])


@router.get("/analyze/plant/{plant_id}")
async def analyze_plant_growth(plant_id: str, nursery_id: str = Query(...)):
    """Predict growth status for a specific plant."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.plant_health.predict_growth(
        data["plant_inventory"], data["growth"], plant_id
    )


@router.get("/predict/disease-risk/{zone_id}")
async def predict_disease_risk(zone_id: str, nursery_id: str = Query(...)):
    """Assess disease risk for a zone."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"high_risk_diseases": model_service.plant_health.assess_disease_risk(
        data["sensor"], data["disease"], zone_id
    )}


@router.get("/detect/stress/{plant_id}")
async def detect_plant_stress(plant_id: str, nursery_id: str = Query(...)):
    """Detect stress conditions for a plant."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.plant_health.detect_stress(
        data["plant_inventory"], data["growth"],
        data["sensor"], plant_id
    )


@router.get("/predict/mortality/{plant_id}")
async def predict_mortality(plant_id: str, nursery_id: str = Query(...)):
    """Predict survival probability for a plant."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.plant_health.predict_mortality(
        data["plant_inventory"], data["growth"],
        data["disease"], plant_id
    )


@router.get("/analyze/zone-health/{zone_id}")
async def zone_health_summary(zone_id: str, nursery_id: str = Query(...)):
    """Get health summary for a zone."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.plant_health.get_zone_health_summary(
        data["plant_inventory"], data["growth"],
        data["disease"], data["sensor"], zone_id
    )
