"""
PlantIQ AI Brain - Graft Success Prediction
Feature engineering from graft records, worker skills, and environmental data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_graft_features(graft_df: pd.DataFrame, graft_id: str = None,
                            row: pd.Series = None) -> Dict:
    """Compute features for a single graft record."""
    if row is None:
        if graft_id is None:
            return {}
        row = graft_df[graft_df["graft_id"] == graft_id]
        if row.empty:
            return {}
        row = row.iloc[0]

    return {
        "rootstock_age_days": int(row.get("rootstock_age_days", 180)),
        "scion_freshness_days": int(row.get("scion_freshness_days", 1)),
        "temperature_at_graft": float(row.get("temperature_at_graft", 23)),
        "humidity_at_graft": float(row.get("humidity_at_graft", 65)),
        "callus_formation_pct": float(row.get("callus_formation_pct", 50)),
        "cambium_alignment_pct": float(row.get("cambium_alignment_pct", 50)),
        "time_of_day_morning": 1 if row.get("time_of_day") == "morning" else 0,
        "time_of_day_afternoon": 1 if row.get("time_of_day") == "afternoon" else 0,
        "rootstock_age_optimal": 1 if 150 <= row.get("rootstock_age_days", 180) <= 240 else 0,
        "scion_fresh": 1 if row.get("scion_freshness_days", 1) <= 2 else 0,
        "temp_optimal": 1 if 18 <= row.get("temperature_at_graft", 23) <= 28 else 0,
        "humidity_optimal": 1 if 50 <= row.get("humidity_at_graft", 65) <= 75 else 0,
    }


def compute_worker_graft_stats(graft_df: pd.DataFrame, worker_id: str,
                                method: str = None) -> Dict:
    """Compute grafting statistics for a specific worker."""
    worker_grafts = graft_df[graft_df["worker_id"] == worker_id]
    if method:
        worker_grafts = worker_grafts[worker_grafts["method"] == method]

    if worker_grafts.empty:
        return {
            "total_grafts": 0, "success_rate": 0, "avg_callus": 0,
            "avg_cambium": 0, "recent_success_rate": 0, "skill_trend": 0,
        }

    total = len(worker_grafts)
    successes = worker_grafts["success"].sum()
    success_rate = successes / total

    # Recent performance (last 30 grafts)
    recent = worker_grafts.tail(30)
    recent_success_rate = recent["success"].mean()

    # Skill trend
    if total >= 20:
        first_half = worker_grafts.head(total // 2)["success"].mean()
        second_half = worker_grafts.tail(total // 2)["success"].mean()
        skill_trend = second_half - first_half
    else:
        skill_trend = 0

    return {
        "total_grafts": total,
        "success_rate": round(success_rate, 4),
        "avg_callus": round(worker_grafts["callus_formation_pct"].mean(), 1),
        "avg_cambium": round(worker_grafts["cambium_alignment_pct"].mean(), 1),
        "recent_success_rate": round(recent_success_rate, 4),
        "skill_trend": round(skill_trend, 4),
    }


def compute_method_stats(graft_df: pd.DataFrame, method: str) -> Dict:
    """Compute statistics for a specific grafting method."""
    method_grafts = graft_df[graft_df["method"] == method]
    if method_grafts.empty:
        return {"total": 0, "success_rate": 0, "avg_callus": 0}

    return {
        "total": len(method_grafts),
        "success_rate": round(method_grafts["success"].mean(), 4),
        "avg_callus": round(method_grafts["callus_formation_pct"].mean(), 1),
        "avg_cambium": round(method_grafts["cambium_alignment_pct"].mean(), 1),
    }


def prepare_graft_training_data(graft_df: pd.DataFrame) -> tuple:
    """Prepare feature matrix and labels for graft success model training."""
    features = []
    labels = []

    for _, row in graft_df.iterrows():
        feats = compute_graft_features(graft_df, row=row)
        worker_stats = compute_worker_graft_stats(graft_df, row["worker_id"], row["method"])

        feature_vector = [
            feats["rootstock_age_days"],
            feats["scion_freshness_days"],
            feats["temperature_at_graft"],
            feats["humidity_at_graft"],
            feats["callus_formation_pct"],
            feats["cambium_alignment_pct"],
            feats["time_of_day_morning"],
            feats["time_of_day_afternoon"],
            feats["rootstock_age_optimal"],
            feats["scion_fresh"],
            feats["temp_optimal"],
            feats["humidity_optimal"],
            worker_stats["success_rate"],
            worker_stats["total_grafts"],
            worker_stats["skill_trend"],
        ]
        features.append(feature_vector)
        labels.append(int(row["success"]))

    return np.array(features), np.array(labels)
