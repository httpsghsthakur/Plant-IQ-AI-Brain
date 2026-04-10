"""PlantIQ AI Brain - API Route: Recommendations & Reports"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Recommendations"])


@router.get("/reports/daily")
async def daily_report(nursery_id: str = Query(...)):
    """Generate comprehensive daily AI advisor report."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.recommendation.generate_daily_report(
        sensor_df=data["sensor"],
        plant_df=data["plant_inventory"],
        growth_df=data["growth"],
        disease_df=data["disease"],
        graft_df=data["graft"],
        attendance_df=data["attendance"],
        task_df=data["task"],
        sales_df=data["sales"],
        expense_df=data["expense"],
        inventory_df=data["inventory"],
    )


@router.get("/dashboard")
async def performance_dashboard(nursery_id: str = Query(...)):
    """Get performance dashboard data."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.recommendation.get_performance_dashboard(
        sensor_df=data["sensor"],
        plant_df=data["plant_inventory"],
        attendance_df=data["attendance"],
        sales_df=data["sales"],
        expense_df=data["expense"],
    )
