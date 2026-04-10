"""
PlantIQ AI Brain - Resource Optimization
Feature engineering from inventory, consumption, and environmental data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_consumption_features(inventory_df: pd.DataFrame, item_name: str,
                                  lookback_days: int = 30) -> Dict:
    """Compute consumption pattern features for an inventory item."""
    item_data = inventory_df[inventory_df["item_name"] == item_name].copy()
    if item_data.empty:
        return {"daily_avg": 0, "daily_std": 0, "trend": 0, "current_stock": 0}

    item_data["date"] = pd.to_datetime(item_data["date"])
    cutoff = item_data["date"].max() - pd.Timedelta(days=lookback_days)
    recent = item_data[item_data["date"] >= cutoff]

    consumed = recent["consumed"].values
    daily_avg = consumed.mean() if len(consumed) > 0 else 0
    daily_std = consumed.std() if len(consumed) > 1 else 0

    # Trend (is consumption increasing or decreasing?)
    trend = 0
    if len(consumed) > 7:
        first_half = consumed[:len(consumed)//2].mean()
        second_half = consumed[len(consumed)//2:].mean()
        trend = second_half - first_half

    current_stock = item_data.sort_values("date").iloc[-1]["closing_stock"]

    return {
        "daily_avg": round(daily_avg, 3),
        "daily_std": round(daily_std, 3),
        "trend": round(trend, 3),
        "current_stock": round(current_stock, 1),
        "below_reorder": bool(item_data.sort_values("date").iloc[-1].get("below_reorder", False)),
    }


def compute_water_usage_features(sensor_df: pd.DataFrame, zone_id: str) -> Dict:
    """Compute water usage features for optimization."""
    zone_data = sensor_df[sensor_df["zone_id"] == zone_id].copy()
    if zone_data.empty:
        return {"avg_moisture": 50, "moisture_variability": 5, "irrigation_events": 0}

    zone_data["timestamp"] = pd.to_datetime(zone_data["timestamp"])
    recent = zone_data.sort_values("timestamp").tail(168)  # Last 7 days

    return {
        "avg_moisture": round(recent["soil_moisture"].mean(), 1),
        "moisture_variability": round(recent["soil_moisture"].std(), 2),
        "avg_temp": round(recent["temperature"].mean(), 1),
        "avg_humidity": round(recent["humidity"].mean(), 1),
        "rainfall_total": round(recent["rainfall_mm"].sum(), 1),
        "evaporation_estimate": round(
            max(0, recent["temperature"].mean() * 0.1 - recent["humidity"].mean() * 0.05), 3
        ),
    }


def compute_fertilizer_needs(plant_df: pd.DataFrame, zone_id: str) -> List[Dict]:
    """Compute fertilizer recommendations based on plant stages in a zone."""
    zone_plants = plant_df[plant_df["zone_id"] == zone_id]
    if zone_plants.empty:
        return []

    # Group by stage and variety
    stage_groups = zone_plants.groupby(["stage", "variety_name"]).size().reset_index(name="count")
    needs = []

    stage_nutrients = {
        "seedling": {"type": "NPK 20-20-20", "dose": "2g/plant", "freq": "Weekly"},
        "young": {"type": "NPK 15-15-15", "dose": "5g/plant", "freq": "Bi-weekly"},
        "growing": {"type": "NPK 10-26-26", "dose": "10g/plant", "freq": "Monthly"},
        "mature": {"type": "NPK 10-10-10 + Micronutrients", "dose": "15g/plant", "freq": "Monthly"},
        "ready_to_sell": {"type": "NPK 10-10-10", "dose": "10g/plant", "freq": "Bi-monthly"},
    }

    for _, row in stage_groups.iterrows():
        stage = row["stage"]
        nutrient = stage_nutrients.get(stage, stage_nutrients["growing"])
        needs.append({
            "stage": stage,
            "variety": row["variety_name"],
            "plant_count": int(row["count"]),
            "fertilizer_type": nutrient["type"],
            "dosage_per_plant": nutrient["dose"],
            "frequency": nutrient["freq"],
        })

    return needs
