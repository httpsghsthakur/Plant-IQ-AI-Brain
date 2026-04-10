"""
PlantIQ AI Brain - Environmental Optimization Model
Schemas for API input/output validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class SensorReading(BaseModel):
    zone_id: str
    temperature: float
    humidity: float
    soil_moisture: float
    light_intensity: float = 0
    soil_ph: float = 6.8
    rainfall_mm: float = 0
    wind_speed_kmh: float = 0


class OptimalRange(BaseModel):
    parameter: str
    current_value: float
    optimal_min: float
    optimal_max: float
    status: str  # "optimal", "below", "above", "critical"
    deviation_pct: float


class Recommendation(BaseModel):
    action: str
    method: str
    urgency: str  # "critical", "high", "medium", "low"
    impact: str
    estimated_cost: Optional[float] = None


class EnvironmentalAnalysis(BaseModel):
    zone_id: str
    zone_name: str
    timestamp: str
    current_conditions: Dict[str, float]
    optimal_ranges: List[OptimalRange]
    recommendations: List[Recommendation]
    overall_status: str  # "optimal", "needs_attention", "critical"
    risk_score: float  # 0-100


class IrrigationScheduleItem(BaseModel):
    zone_id: str
    zone_name: str
    next_irrigation: str
    duration_minutes: int
    water_amount_liters: float
    reason: str
    urgency: str


class IrrigationSchedule(BaseModel):
    schedule: List[IrrigationScheduleItem]
    total_water_liters: float
    water_savings_pct: float
    cost_savings_monthly: float


class WeatherAlert(BaseModel):
    alert_type: str
    severity: str
    message: str
    affected_zones: List[str]
    recommended_actions: List[str]
    valid_until: str
