"""PlantIQ AI Brain - Yield Forecasting Schemas"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class ProductionForecast(BaseModel):
    forecast_period: str
    total_plants_ready: int
    by_variety: List[Dict]
    by_grade: Dict[str, int]
    confidence_interval: Dict[str, int]
    revenue_forecast: Dict[str, float]


class DemandSupplyAnalysis(BaseModel):
    period: str
    supply_forecast: int
    estimated_demand: int
    gap: int
    gap_status: str  # "surplus", "deficit", "balanced"
    recommendations: List[str]
    by_variety: List[Dict]


class QualityDistribution(BaseModel):
    total_plants: int
    distribution: Dict[str, Dict]
    improvement_potential: List[Dict]
