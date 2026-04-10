"""PlantIQ AI Brain - Anomaly Detection Schemas"""
from pydantic import BaseModel
from typing import List, Optional, Dict


class SensorAnomaly(BaseModel):
    sensor_type: str
    zone_id: str
    zone_name: str
    anomaly_type: str  # "spike", "drift", "offline", "stuck"
    severity: str
    current_value: float
    expected_range: Dict[str, float]
    detection_method: str
    recommended_action: str
    timestamp: str


class WorkerAnomaly(BaseModel):
    worker_id: str
    worker_name: str
    anomaly_type: str
    description: str
    severity: str
    recommended_action: str


class InventoryAnomaly(BaseModel):
    item_name: str
    anomaly_type: str
    description: str
    severity: str
    recommended_action: str


class AnomalyReport(BaseModel):
    total_anomalies: int
    critical_count: int
    sensor_anomalies: List[Dict]
    worker_anomalies: List[Dict]
    inventory_anomalies: List[Dict]
    generated_at: str
