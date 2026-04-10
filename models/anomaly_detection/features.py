"""
PlantIQ AI Brain - Anomaly Detection
Feature engineering for baseline computation and anomaly detection.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_sensor_baselines(sensor_df: pd.DataFrame, lookback_days: int = 30) -> Dict:
    """Compute baseline statistics per zone per sensor type."""
    df = sensor_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["timestamp"] >= cutoff]

    baselines = {}
    sensor_cols = ["temperature", "humidity", "soil_moisture", "light_intensity", "soil_ph",
                   "rainfall_mm", "wind_speed_kmh"]

    for zone_id in recent["zone_id"].unique():
        zone_data = recent[recent["zone_id"] == zone_id]
        baselines[zone_id] = {}
        for col in sensor_cols:
            if col in zone_data.columns:
                values = zone_data[col].dropna()
                if len(values) > 0:
                    baselines[zone_id][col] = {
                        "mean": round(values.mean(), 2),
                        "std": round(values.std(), 2),
                        "min": round(values.min(), 2),
                        "max": round(values.max(), 2),
                        "q25": round(values.quantile(0.25), 2),
                        "q75": round(values.quantile(0.75), 2),
                        "iqr": round(values.quantile(0.75) - values.quantile(0.25), 2),
                    }

    return baselines


def detect_sensor_spikes(sensor_df: pd.DataFrame, baselines: Dict,
                          threshold_sigma: float = 3.0) -> List[Dict]:
    """Detect spikes in sensor readings using statistical thresholds."""
    df = sensor_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.sort_values("timestamp").groupby("zone_id").tail(24)  # Last 24 hours

    anomalies = []
    sensor_cols = ["temperature", "humidity", "soil_moisture", "light_intensity", "soil_ph"]

    for zone_id in latest["zone_id"].unique():
        zone_data = latest[latest["zone_id"] == zone_id]
        zone_baselines = baselines.get(zone_id, {})

        for col in sensor_cols:
            if col not in zone_baselines or col not in zone_data.columns:
                continue

            baseline = zone_baselines[col]
            values = zone_data[col].dropna()

            for idx, val in values.items():
                if baseline["std"] > 0:
                    z_score = abs(val - baseline["mean"]) / baseline["std"]
                    if z_score > threshold_sigma:
                        anomalies.append({
                            "sensor_type": col,
                            "zone_id": zone_id,
                            "value": float(val),
                            "expected_mean": baseline["mean"],
                            "expected_std": baseline["std"],
                            "z_score": round(z_score, 2),
                            "timestamp": str(zone_data.loc[idx, "timestamp"]) if "timestamp" in zone_data.columns else "",
                        })

    return anomalies


def detect_stuck_sensors(sensor_df: pd.DataFrame, min_window: int = 12) -> List[Dict]:
    """Detect sensors producing identical readings (possibly malfunctioning)."""
    df = sensor_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    stuck = []
    sensor_cols = ["temperature", "humidity", "soil_moisture"]

    for zone_id in df["zone_id"].unique():
        zone_data = df[df["zone_id"] == zone_id].sort_values("timestamp").tail(48)
        for col in sensor_cols:
            if col not in zone_data.columns:
                continue
            values = zone_data[col].dropna().values
            if len(values) >= min_window:
                # Check if last N readings are identical
                last_readings = values[-min_window:]
                if np.std(last_readings) < 0.01:
                    stuck.append({
                        "sensor_type": col,
                        "zone_id": zone_id,
                        "stuck_value": float(last_readings[0]),
                        "duration_hours": min_window,
                    })

    return stuck


def compute_worker_baseline(attendance_df: pd.DataFrame, task_df: pd.DataFrame) -> Dict:
    """Compute baseline worker behavior patterns."""
    att = attendance_df.copy()
    att["date"] = pd.to_datetime(att["date"])

    baselines = {}
    for wid in att["worker_id"].unique():
        worker_att = att[att["worker_id"] == wid]
        present = worker_att[worker_att["status"] == "present"]

        if len(present) > 0:
            baselines[wid] = {
                "avg_hours": round(present["work_hours"].mean(), 2),
                "std_hours": round(present["work_hours"].std(), 2),
                "absence_rate": round(1 - len(present) / max(1, len(worker_att)), 4),
                "avg_overtime": round(present["overtime_hours"].mean(), 2),
            }

    return baselines
