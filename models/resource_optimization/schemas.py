"""PlantIQ AI Brain - Resource Optimization Schemas"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class WaterOptimization(BaseModel):
    zone_id: str
    zone_name: str
    current_usage_liters: float
    optimized_usage_liters: float
    savings_pct: float
    schedule: List[Dict]
    recommendations: List[str]


class FertilizerPlan(BaseModel):
    zone_id: str
    variety: str
    stage: str
    fertilizer_type: str
    dosage: str
    frequency: str
    estimated_cost: str
    expected_impact: str


class InventoryForecast(BaseModel):
    item_name: str
    current_stock: float
    daily_usage_rate: float
    days_until_stockout: int
    reorder_date: str
    reorder_quantity: float
    estimated_cost: str
    urgency: str


class ResourceSummary(BaseModel):
    total_monthly_cost: float
    optimized_monthly_cost: float
    savings_pct: float
    top_savings_opportunities: List[Dict]
    inventory_alerts: List[Dict]
