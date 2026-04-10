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
    try:
        data = data_service.load_nursery_data(nursery_id)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load nursery data: {str(e)}",
            "nursery_wellness_score": 0,
            "urgent_care_alerts": [],
            "summary": {}
        }
    
    health_score = 100.0
    active_alerts = []
    growing_areas = 0
    
    # 2. Compile Active Alerts & Health Score
    zones = data.get("zone")
    if zones is not None and not zones.empty:
        growing_areas = len(zones)
        
        # Analyze each zone
        for _, zone in zones.iterrows():
            zone_id = zone["id"]
            
            # Check for environmental stress
            env_results = model_service.environmental.analyze_zone(data["sensor"], zone_id)
            if env_results and env_results.get("overall_status") in ["critical", "needs_attention"]:
                severity = "critical" if env_results["overall_status"] == "critical" else "high"
                risk_score = env_results.get("risk_score", 0)
                health_score -= (risk_score * 0.2)
                
                # Add most urgent recommendation as alert
                recs = env_results.get("recommendations", [])
                if recs:
                    active_alerts.append({
                        "id": f"env-{zone_id}",
                        "severity": severity,
                        "message": f"Zone {zone_id}: {recs[0]['action']}"
                    })

            # Check for disease risks
            disease_risks = model_service.plant_health.assess_disease_risk(data["sensor"], data["disease"], zone_id)
            if disease_risks:
                highest_risk = max(disease_risks, key=lambda x: x.get("probability", 0))
                if highest_risk.get("probability", 0) > 0.5:
                    prob = highest_risk.get("probability", 0)
                    severity = "critical" if prob > 0.8 else "high"
                    health_score -= (prob * 30)
                    active_alerts.append({
                        "id": f"disease-{zone_id}",
                        "severity": severity,
                        "message": f"Risk of {highest_risk.get('disease')} in {zone_id}"
                    })

    # Ensure score is within valid bounds
    health_score = max(0, min(100, health_score))

    # Compile Summary
    tasks = data.get("task")
    pending_tasks = len(tasks[tasks["status"] != "completed"]) if tasks is not None and not tasks.empty else 0

    return {
        "status": "success",
        "nursery_wellness_score": round(health_score, 1),
        "urgent_care_alerts": active_alerts,
        "summary": {
            "growing_areas": growing_areas, 
            "pending_daily_tasks": pending_tasks,
            "sensor_system_health": "All systems running smoothly" if health_score > 80 else "Needs attention"
        }
    }
