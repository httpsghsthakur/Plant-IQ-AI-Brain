"""PlantIQ AI Brain - API Route: Anomaly Detection"""
from fastapi import APIRouter, Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Anomaly Detection"])


@router.get("/anomalies/sensors")
async def sensor_anomalies(nursery_id: str = Query(...)):
    """Detect anomalies in sensor readings."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"anomalies": model_service.anomaly.detect_sensor_anomalies(data["sensor"])}


@router.get("/anomalies/workers")
async def worker_anomalies(nursery_id: str = Query(...)):
    """Detect unusual worker activity patterns."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"anomalies": model_service.anomaly.detect_worker_anomalies(
        data["attendance"], data["task"]
    )}


@router.get("/anomalies/inventory")
async def inventory_anomalies(nursery_id: str = Query(...)):
    """Detect inventory consumption anomalies."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return {"anomalies": model_service.anomaly.detect_inventory_anomalies(data["inventory"])}


@router.get("/anomalies/report")
async def anomaly_report(nursery_id: str = Query(...)):
    """Get comprehensive anomaly report."""
    from services.model_service import model_service
    from services.data_service import data_service
    data = data_service.load_nursery_data(nursery_id)
    return model_service.anomaly.get_anomaly_report(
        data["sensor"], data["attendance"],
        data["task"], data["inventory"],
    )
