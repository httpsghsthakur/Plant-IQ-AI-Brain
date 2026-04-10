"""PlantIQ AI Brain - Recommendation Engine Schemas"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class ActionItem(BaseModel):
    priority: str  # "urgent", "important", "optimization"
    category: str  # "environmental", "plant_health", "worker", "financial", etc.
    title: str
    description: str
    impact: str
    estimated_cost: Optional[str] = None
    estimated_roi: Optional[str] = None
    assigned_to: Optional[str] = None
    deadline: Optional[str] = None


class DailyReport(BaseModel):
    report_date: str
    nursery_name: str
    summary: Dict
    urgent_actions: List[Dict]
    important_actions: List[Dict]
    optimization_actions: List[Dict]
    performance_snapshot: Dict
    generated_at: str


class PerformanceDashboard(BaseModel):
    overall_score: float
    category_scores: Dict[str, float]
    trends: Dict[str, str]
    key_metrics: Dict
    alerts_count: int
