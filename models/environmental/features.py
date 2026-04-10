"""
PlantIQ AI Brain - Environmental Optimization Model
Feature engineering for sensor data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_zone_features(sensor_df: pd.DataFrame, zone_id: str = None) -> pd.DataFrame:
    """Compute environmental features for a zone from sensor data."""
    if zone_id:
        df = sensor_df[sensor_df["zone_id"] == zone_id].copy()
    else:
        df = sensor_df.copy()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Temperature features
    df["temp_deviation"] = df["temperature"] - 23  # deviation from ideal
    df["temp_stress"] = ((df["temperature"] < 10) | (df["temperature"] > 35)).astype(int)
    df["temp_optimal"] = ((df["temperature"] >= 18) & (df["temperature"] <= 28)).astype(int)

    # Humidity features
    df["humidity_deviation"] = df["humidity"] - 65
    df["humidity_stress"] = ((df["humidity"] < 30) | (df["humidity"] > 90)).astype(int)
    df["humidity_optimal"] = ((df["humidity"] >= 55) & (df["humidity"] <= 75)).astype(int)

    # Soil moisture features
    df["moisture_deviation"] = df["soil_moisture"] - 50
    df["moisture_stress"] = ((df["soil_moisture"] < 25) | (df["soil_moisture"] > 80)).astype(int)
    df["moisture_optimal"] = ((df["soil_moisture"] >= 40) & (df["soil_moisture"] <= 60)).astype(int)
    df["needs_irrigation"] = (df["soil_moisture"] < 38).astype(int)

    # Light features
    df["is_daylight"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    df["light_stress"] = ((df["light_intensity"] > 50000) & df["is_daylight"].astype(bool)).astype(int)

    # Composite risk score (0-100)
    df["risk_score"] = np.clip(
        df["temp_stress"] * 25
        + df["humidity_stress"] * 20
        + df["moisture_stress"] * 30
        + df["light_stress"] * 10
        + np.abs(df["temp_deviation"]) * 1.5
        + np.abs(df["humidity_deviation"]) * 0.8,
        0, 100
    )

    return df


def compute_daily_aggregates(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily aggregate features per zone for model training."""
    df = sensor_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    agg = df.groupby(["date", "zone_id"]).agg(
        temp_mean=("temperature", "mean"),
        temp_min=("temperature", "min"),
        temp_max=("temperature", "max"),
        temp_std=("temperature", "std"),
        humidity_mean=("humidity", "mean"),
        humidity_min=("humidity", "min"),
        humidity_max=("humidity", "max"),
        moisture_mean=("soil_moisture", "mean"),
        moisture_min=("soil_moisture", "min"),
        moisture_max=("soil_moisture", "max"),
        light_mean=("light_intensity", "mean"),
        light_max=("light_intensity", "max"),
        ph_mean=("soil_ph", "mean"),
        rainfall_total=("rainfall_mm", "sum"),
        wind_mean=("wind_speed_kmh", "mean"),
        wind_max=("wind_speed_kmh", "max"),
    ).reset_index()

    # Temperature range
    agg["temp_range"] = agg["temp_max"] - agg["temp_min"]
    agg["humidity_range"] = agg["humidity_max"] - agg["humidity_min"]

    # Risk indicators
    agg["heat_stress_hours"] = df.groupby(["date", "zone_id"]).apply(
        lambda x: (x["temperature"] > 32).sum()
    ).values
    agg["frost_risk"] = (agg["temp_min"] < 5).astype(int)
    agg["drought_risk"] = (agg["moisture_min"] < 30).astype(int)

    return agg


def get_irrigation_features(sensor_df: pd.DataFrame, zone_id: str) -> Dict:
    """Extract features relevant for irrigation scheduling."""
    df = sensor_df[sensor_df["zone_id"] == zone_id].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Last 24 hours
    latest = df.sort_values("timestamp").tail(24)

    return {
        "current_moisture": latest["soil_moisture"].iloc[-1] if len(latest) > 0 else 50,
        "moisture_trend": latest["soil_moisture"].diff().mean() if len(latest) > 1 else 0,
        "avg_temp_24h": latest["temperature"].mean(),
        "total_rainfall_24h": latest["rainfall_mm"].sum(),
        "humidity_avg": latest["humidity"].mean(),
        "evaporation_estimate": max(0, latest["temperature"].mean() * 0.1 - latest["humidity"].mean() * 0.05),
    }
