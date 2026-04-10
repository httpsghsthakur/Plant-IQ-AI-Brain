"""
PlantIQ AI Brain - Plant Health Prediction Model
Feature engineering from growth measurements, disease records, and environmental data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_growth_features(plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                            plant_id: str) -> Dict:
    """Compute growth-related features for a single plant."""
    plant = plant_df[plant_df["plant_id"] == plant_id]
    if plant.empty:
        return {}
    plant = plant.iloc[0]

    measurements = growth_df[growth_df["plant_id"] == plant_id].copy()
    if measurements.empty:
        return {
            "age_days": int(plant.get("age_days", 0)),
            "current_height": float(plant.get("current_height_cm", 0)),
            "expected_height": float(plant.get("expected_height_cm", 0)),
            "height_deviation_pct": float(plant.get("height_deviation_pct", 0)),
            "growth_trend": 0,
            "growth_acceleration": 0,
            "avg_weekly_growth": 0,
            "recent_weekly_growth": 0,
            "growth_consistency": 0,
            "health_score_latest": 75,
            "health_score_trend": 0,
            "stress_count": 0,
            "disease_detected_count": 0,
        }

    measurements["measurement_date"] = pd.to_datetime(measurements["measurement_date"])
    measurements = measurements.sort_values("measurement_date")

    # Growth trend (slope of weekly growth over time)
    weekly_growth = measurements["weekly_growth_cm"].values
    growth_trend = 0
    growth_acceleration = 0
    if len(weekly_growth) > 2:
        x = np.arange(len(weekly_growth))
        coeffs = np.polyfit(x, weekly_growth, 1)
        growth_trend = coeffs[0]  # slope
        if len(weekly_growth) > 4:
            coeffs2 = np.polyfit(x, weekly_growth, 2)
            growth_acceleration = coeffs2[0]

    # Recent vs historical growth
    avg_weekly = weekly_growth.mean() if len(weekly_growth) > 0 else 0
    recent_weekly = weekly_growth[-4:].mean() if len(weekly_growth) >= 4 else avg_weekly

    # Growth consistency (coefficient of variation)
    growth_consistency = 0
    if avg_weekly > 0 and len(weekly_growth) > 2:
        growth_consistency = 1 - min(1, weekly_growth.std() / avg_weekly)

    # Health score analysis
    health_scores = measurements["health_score"].values
    health_latest = health_scores[-1] if len(health_scores) > 0 else 75
    health_trend = 0
    if len(health_scores) > 3:
        recent_health = health_scores[-3:].mean()
        older_health = health_scores[:-3].mean()
        health_trend = recent_health - older_health

    # Stress and disease counts
    stress_count = int((measurements["stress_indicators"] != "none").sum())
    disease_count = int(measurements["disease_detected"].notna().sum())

    return {
        "age_days": int(plant.get("age_days", 0)),
        "current_height": float(plant.get("current_height_cm", 0)),
        "expected_height": float(plant.get("expected_height_cm", 0)),
        "height_deviation_pct": float(plant.get("height_deviation_pct", 0)),
        "growth_trend": round(growth_trend, 4),
        "growth_acceleration": round(growth_acceleration, 4),
        "avg_weekly_growth": round(avg_weekly, 3),
        "recent_weekly_growth": round(recent_weekly, 3),
        "growth_consistency": round(growth_consistency, 3),
        "health_score_latest": round(float(health_latest), 1),
        "health_score_trend": round(health_trend, 2),
        "stress_count": stress_count,
        "disease_detected_count": disease_count,
    }


def compute_disease_risk_features(sensor_df: pd.DataFrame, disease_df: pd.DataFrame,
                                   zone_id: str, lookback_days: int = 30) -> Dict:
    """Compute disease risk features for a zone based on environmental conditions."""
    # Recent environmental conditions
    zone_sensors = sensor_df[sensor_df["zone_id"] == zone_id].copy()
    if zone_sensors.empty:
        return {
            "avg_temp": 23, "avg_humidity": 65, "avg_moisture": 50,
            "high_humidity_hours": 0, "high_temp_hours": 0,
            "rainfall_total": 0, "disease_history_count": 0,
            "disease_recurrence_rate": 0,
        }

    zone_sensors["timestamp"] = pd.to_datetime(zone_sensors["timestamp"])
    cutoff = zone_sensors["timestamp"].max() - pd.Timedelta(days=lookback_days)
    recent = zone_sensors[zone_sensors["timestamp"] >= cutoff]

    avg_temp = recent["temperature"].mean()
    avg_humidity = recent["humidity"].mean()
    avg_moisture = recent["soil_moisture"].mean()

    # Environmental risk indicators
    high_humidity_hours = int((recent["humidity"] > 80).sum())
    high_temp_hours = int((recent["temperature"] > 32).sum())
    low_temp_hours = int((recent["temperature"] < 5).sum())
    rainfall_total = recent["rainfall_mm"].sum()

    # Disease history for this zone
    zone_diseases = disease_df[disease_df["zone_id"] == zone_id]
    disease_count = len(zone_diseases)
    if disease_count > 0:
        treated = zone_diseases["treatment_applied"].sum()
        successful = zone_diseases["treatment_success"].sum()
        recurrence = 1 - (successful / max(1, treated))
    else:
        recurrence = 0

    return {
        "avg_temp": round(avg_temp, 1),
        "avg_humidity": round(avg_humidity, 1),
        "avg_moisture": round(avg_moisture, 1),
        "high_humidity_hours": high_humidity_hours,
        "high_temp_hours": high_temp_hours,
        "low_temp_hours": low_temp_hours,
        "rainfall_total": round(rainfall_total, 1),
        "disease_history_count": disease_count,
        "disease_recurrence_rate": round(recurrence, 3),
    }


def compute_mortality_features(plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                                disease_df: pd.DataFrame, plant_id: str) -> Dict:
    """Compute features for mortality/survival prediction."""
    plant = plant_df[plant_df["plant_id"] == plant_id]
    if plant.empty:
        return {}
    plant = plant.iloc[0]

    # Growth features
    growth_feats = compute_growth_features(plant_df, growth_df, plant_id)

    # Disease history for this plant
    plant_diseases = disease_df[disease_df["plant_id"] == plant_id]
    total_diseases = len(plant_diseases)
    severe_diseases = len(plant_diseases[plant_diseases["severity"] == "severe"])
    untreated = len(plant_diseases[plant_diseases["treatment_applied"] == False])
    treatment_failures = len(plant_diseases[
        (plant_diseases["treatment_applied"] == True) &
        (plant_diseases["treatment_success"] == False)
    ])

    # Health status encoding
    health_status = plant.get("health_status", "healthy")
    health_code = {"healthy": 0, "at_risk": 1, "critical": 2}.get(health_status, 0)

    # Is grafted and graft outcome
    is_grafted = bool(plant.get("is_grafted", False))
    graft_success = plant.get("graft_success", None)
    graft_factor = 0
    if is_grafted:
        graft_factor = 1 if graft_success else -1

    return {
        **growth_feats,
        "total_diseases": total_diseases,
        "severe_diseases": severe_diseases,
        "untreated_diseases": untreated,
        "treatment_failures": treatment_failures,
        "health_code": health_code,
        "is_grafted": int(is_grafted),
        "graft_factor": graft_factor,
        "variety": plant.get("variety_name", "Unknown"),
        "stage": plant.get("stage", "seedling"),
        "zone_id": plant.get("zone_id", "Unknown"),
    }


def prepare_training_data(plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                           disease_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix for model training from all plant data."""
    records = []
    for _, plant in plant_df.iterrows():
        pid = plant["plant_id"]
        feats = compute_mortality_features(plant_df, growth_df, disease_df, pid)
        if feats:
            # Target: survival probability approximation
            survival = 1.0
            if feats["health_code"] == 2:
                survival -= 0.3
            if feats["severe_diseases"] > 0:
                survival -= 0.15 * feats["severe_diseases"]
            if feats["treatment_failures"] > 0:
                survival -= 0.1 * feats["treatment_failures"]
            if feats["height_deviation_pct"] < -30:
                survival -= 0.1
            if feats["graft_factor"] < 0:
                survival -= 0.15
            survival = max(0.05, min(1.0, survival))

            feats["survival_probability"] = survival
            feats["is_alive"] = 1 if survival > 0.3 else 0
            records.append(feats)

    return pd.DataFrame(records)
