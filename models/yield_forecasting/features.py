"""
PlantIQ AI Brain - Yield Forecasting
Feature engineering from plant inventory and sales data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_production_pipeline(plant_df: pd.DataFrame, days_ahead: int = 90) -> Dict:
    """Compute production pipeline — plants expected to be ready in the forecast period."""
    df = plant_df.copy()

    # Estimate days to ready_to_sell based on current stage
    stage_days_remaining = {
        "seedling": 240,
        "young": 180,
        "growing": 90,
        "mature": 30,
        "ready_to_sell": 0,
    }

    df["days_to_ready"] = df["stage"].map(stage_days_remaining).fillna(120)
    df["days_to_ready"] = df["days_to_ready"] - (df["age_days"] * 0.3)  # Adjust by current age
    df["days_to_ready"] = df["days_to_ready"].clip(lower=0)

    # Plants ready within forecast period
    ready_mask = df["days_to_ready"] <= days_ahead
    ready_plants = df[ready_mask]

    # By variety
    by_variety = ready_plants.groupby("variety_name").agg(
        count=("plant_id", "size"),
        avg_height=("current_height_cm", "mean"),
        avg_deviation=("height_deviation_pct", "mean"),
    ).reset_index()

    return {
        "total_ready": int(ready_mask.sum()),
        "by_variety": by_variety.to_dict("records") if len(by_variety) > 0 else [],
        "currently_ready": int((df["stage"] == "ready_to_sell").sum()),
        "pipeline_30d": int((df["days_to_ready"] <= 30).sum()),
        "pipeline_60d": int((df["days_to_ready"] <= 60).sum()),
        "pipeline_90d": int((df["days_to_ready"] <= 90).sum()),
    }


def compute_demand_features(sales_df: pd.DataFrame, lookback_days: int = 90) -> Dict:
    """Compute demand features from historical sales data."""
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    recent = df[df["date"] >= cutoff]

    if recent.empty:
        return {"monthly_avg_orders": 0, "monthly_avg_quantity": 0, "trend": 0}

    # Monthly aggregation
    recent["month"] = recent["date"].dt.to_period("M")
    monthly = recent.groupby("month").agg(
        orders=("order_id", "size"),
        quantity=("quantity", "sum"),
        revenue=("total_amount", "sum"),
    ).reset_index()

    # Demand by variety
    variety_demand = recent.groupby("variety_name")["quantity"].sum().to_dict()

    # Trend
    trend = 0
    if len(monthly) >= 2:
        trend = (monthly["quantity"].iloc[-1] - monthly["quantity"].iloc[0]) / max(1, len(monthly))

    return {
        "monthly_avg_orders": round(monthly["orders"].mean(), 1),
        "monthly_avg_quantity": round(monthly["quantity"].mean(), 1),
        "monthly_avg_revenue": round(monthly["revenue"].mean(), 0),
        "trend": round(trend, 2),
        "variety_demand": variety_demand,
        "total_recent_orders": int(recent["order_id"].nunique()),
    }


def compute_quality_features(plant_df: pd.DataFrame) -> Dict:
    """Compute quality grade distribution predictions."""
    df = plant_df.copy()

    # Predict grade based on health and growth deviation
    def predict_grade(row):
        dev = row.get("height_deviation_pct", 0)
        health = row.get("health_status", "healthy")
        if health == "critical":
            return "C"
        if health == "at_risk":
            return "B" if dev > -10 else "C"
        if dev > 10:
            return "A+"
        if dev > -10:
            return "A"
        if dev > -25:
            return "B"
        return "C"

    df["predicted_grade"] = df.apply(predict_grade, axis=1)

    grade_dist = df["predicted_grade"].value_counts().to_dict()
    grade_by_variety = df.groupby(["variety_name", "predicted_grade"]).size().unstack(fill_value=0)

    return {
        "grade_distribution": grade_dist,
        "grade_by_variety": grade_by_variety.to_dict() if len(grade_by_variety) > 0 else {},
        "total_plants": len(df),
    }
