"""PlantIQ AI Brain - API Route: Yield Forecasting"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Yield Forecasting"])


@router.get("/forecast/production")
async def forecast_production(nursery_id: str = Query(...), days_ahead: int = Query(90)):
    """Forecast production for the next N days."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.yield_forecast.forecast_production(
        data["plant_inventory"], days_ahead
    )


@router.get("/forecast/demand-supply")
async def demand_supply_analysis(nursery_id: str = Query(...), months: int = Query(3)):
    """Analyze demand-supply gap."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.yield_forecast.analyze_demand_supply(
        data["plant_inventory"], data["sales"], months
    )


@router.get("/forecast/quality")
async def quality_forecast(nursery_id: str = Query(...)):
    """Predict quality grade distribution."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.yield_forecast.predict_quality(data["plant_inventory"])
