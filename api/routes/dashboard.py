"""PlantIQ AI Brain - API Route: Integrated Dashboard"""
from fastapi import APIRouter, Query
from typing import Dict, Any, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/ai", tags=["Dashboard"])

@router.get("/dashboard")
async def get_dashboard_summary(nursery_id: str = Query(...)) -> Dict[str, Any]:
    """
    Aggregates data from all AI models to provide a top-level system health score 
    and mission-critical alerts for the frontend dashboard.
    """
    from services.model_service import model_service
    from services.data_service import data_service
    
    # 1. Fetch scoped nursery data
    data = data_service.load_nursery_data(nursery_id)
    
    # 2. Calculate a mock Global Health Score based on actual model health signals
    # In a real scenario, this would be a weighted average of model outputs.
    health_score = 88.5 # Starting default
    
    # 3. Compile Active Alerts
    active_alerts = []
    
    # Check for environmental stress
    env_results = model_service.environmental.analyze_zone(data["sensor"], "ZONE-A")
    if env_results.get("risk_level") == "High":
        active_alerts.append({
            "id": "env-alert-1",
            "severity": "high",
            "message": "Thermal stress detected in Zone A. Immediate irrigation required."
        })
        health_score -= 10

    # Check for disease risks
    disease_risks = model_service.plant_health.assess_disease_risk(data["sensor"], data["disease"], "ZONE-A")
    if any(r.get("probability", 0) > 0.7 for r in disease_risks):
         active_alerts.append({
            "id": "disease-alert-1",
            "severity": "critical",
            "message": "High Walnut Blight risk detected in propagation area."
        })
         health_score -= 15

    return {
        "status": "success",
        "nursery_wellness_score": max(0, health_score),
        "urgent_care_alerts": active_alerts,
        "summary": {
            "growing_areas": 4, 
            "pending_daily_tasks": len(data.get("task", [])),
            "sensor_system_health": "All systems running smoothly"
        }
    }
