"""PlantIQ AI Brain - Graft Success Prediction Schemas"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class GraftPrediction(BaseModel):
    graft_id: Optional[str] = None
    worker_id: str
    worker_name: str
    method: str
    rootstock_variety: str
    scion_variety: str
    success_probability: float
    risk_factors: List[str]
    recommendations: List[str]
    optimal_conditions: Dict[str, str]


class WorkerMethodStats(BaseModel):
    worker_id: str
    worker_name: str
    method: str
    total_grafts: int
    success_rate: float
    avg_callus_formation: float
    recommendation: str


class BatchOptimization(BaseModel):
    assignments: List[Dict]
    expected_success_rate: float
    improvement_over_random: str
    total_grafts: int
    estimated_cost_savings: str


class GraftMonitoring(BaseModel):
    graft_id: str
    days_post_graft: int
    callus_formation_pct: float
    cambium_alignment_pct: float
    predicted_outcome: str
    confidence: float
    action_needed: str
