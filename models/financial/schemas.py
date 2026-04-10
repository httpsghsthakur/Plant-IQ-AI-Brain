"""PlantIQ AI Brain - Financial Analytics Schemas"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class ProfitLossReport(BaseModel):
    period: str
    total_revenue: float
    total_expenses: float
    gross_profit: float
    profit_margin: float
    revenue_trend: str
    expense_breakdown: List[Dict]
    insights: List[str]
    comparison_to_target: Dict


class CashFlowForecast(BaseModel):
    forecast_months: int
    monthly_forecasts: List[Dict]
    projected_balance: float
    cash_flow_status: str
    alerts: List[str]


class CostOptimization(BaseModel):
    total_current_cost: float
    total_optimized_cost: float
    savings_potential: float
    opportunities: List[Dict]


class PricingRecommendation(BaseModel):
    variety: str
    current_avg_price: float
    recommended_price: float
    change_pct: float
    reasoning: str
    expected_impact: str
