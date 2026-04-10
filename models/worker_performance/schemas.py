"""
PlantIQ AI Brain - Worker Performance Analytics Model
Schemas for worker performance API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class AttendanceScore(BaseModel):
    score: float
    days_present: int
    days_total: int
    on_time_rate: str
    early_departures: int
    late_arrivals: int


class ProductivityScore(BaseModel):
    score: float
    tasks_completed: int
    tasks_assigned: int
    completion_rate: str
    avg_completion_time_pct: str


class QualityScore(BaseModel):
    score: float
    grafts_success_rate: Optional[str] = None
    error_rate: str
    rework_needed: int


class InitiativeScore(BaseModel):
    score: float
    proactive_issues_reported: int
    suggestions_submitted: int


class WorkerPerformance(BaseModel):
    worker_id: str
    worker_name: str
    period: str
    performance_score: float
    breakdown: Dict[str, dict]
    rank: str
    trend: str
    recommendations: List[str]


class ProductivityInsight(BaseModel):
    finding: str
    impact: str
    recommendation: str
    expected_improvement: str


class WorkloadBalance(BaseModel):
    current_workload: Dict[str, List[str]]
    recommended_redistribution: List[Dict]
    expected_outcome: str


class AbsenteeismPrediction(BaseModel):
    worker_id: str
    worker_name: str
    absence_probability: float
    predicted_date: str
    reason: str
    recommended_action: str


class BurnoutAlert(BaseModel):
    worker_id: str
    worker_name: str
    burnout_risk: str
    risk_score: float
    indicators: List[str]
    recommendations: List[str]


class TrainingNeed(BaseModel):
    skill: str
    workers: List[str]
    reason: str
    training_type: str
    duration: str
    expected_improvement: str
    estimated_roi: str
