"""PlantIQ AI Brain - API Route: Environmental Optimization"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Environmental"])


@router.get("/analyze/environment/{zone_id}")
async def analyze_environment(zone_id: str, nursery_id: str = Query(...)):
    """Analyze environmental conditions for a specific zone."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.environmental.analyze_zone(data["sensor"], zone_id)


@router.get("/analyze/environment")
async def analyze_all_environments(nursery_id: str = Query(...)):
    """Analyze environmental conditions for all zones."""
    from services.model_service import model_service
    from services.data_service import data_service
    import config
    data = data_service.load_nursery_data(nursery_id)
    results = []
    for zone in config.ZONES:
        result = model_service.environmental.analyze_zone(data["sensor"], zone["id"])
        results.append(result)
    return {"zones": results}


@router.get("/optimize/irrigation")
async def optimize_irrigation(nursery_id: str = Query(...)):
    """Get optimized irrigation schedule for all zones."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.environmental.get_irrigation_schedule(data["sensor"])


@router.get("/alerts/weather")
async def weather_alerts(nursery_id: str = Query(...)):
    """Get weather-based proactive alerts."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"alerts": model_service.environmental.get_weather_alerts(data["sensor"])}
