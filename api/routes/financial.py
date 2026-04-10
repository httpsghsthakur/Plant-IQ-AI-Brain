"""PlantIQ AI Brain - API Route: Financial Analytics"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Financial Analytics"])


@router.get("/analyze/financials")
async def profit_loss_report(nursery_id: str = Query(...), months: int = Query(3)):
    """Generate Profit & Loss report."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.financial.get_profit_loss(
        data["sales"], data["expense"], months
    )


@router.get("/forecast/cashflow")
async def cashflow_forecast(nursery_id: str = Query(...), months_ahead: int = Query(3)):
    """Forecast cash flow."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.financial.forecast_cashflow(
        data["sales"], data["expense"], months_ahead
    )


@router.get("/optimize/costs")
async def optimize_costs(nursery_id: str = Query(...)):
    """Identify cost optimization opportunities."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.financial.optimize_costs(data["expense"])


@router.get("/recommend/pricing")
async def recommend_pricing(nursery_id: str = Query(...)):
    """Generate dynamic pricing recommendations."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"recommendations": model_service.financial.recommend_pricing(
        data["sales"], data["plant_inventory"]
    )}
