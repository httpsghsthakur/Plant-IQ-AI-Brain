"""PlantIQ AI Brain - Plant Health Model Schemas"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class GrowthPrediction(BaseModel):
    plant_id: str
    variety: str
    age_days: int
    current_height: float
    expected_height: float
    growth_status: str
    deviation_pct: float
    likely_causes: List[str]
    recommendations: List[str]
    predicted_recovery_time: Optional[str] = None


class DiseaseRisk(BaseModel):
    zone_id: str
    disease_name: str
    risk_score: float
    risk_level: str
    contributing_factors: List[str]
    recommended_actions: List[str]
    prevention_cost: str
    treatment_cost_if_outbreak: str


class StressDetection(BaseModel):
    plant_id: str
    stress_level: str
    stress_type: str
    indicators: List[str]
    immediate_action: str
    long_term_fix: str


class MortalityPrediction(BaseModel):
    plant_id: str
    survival_probability: float
    status: str
    prognosis: str
    reasons: List[str]
    recommendations: List[str]
    ai_recommendation: str
