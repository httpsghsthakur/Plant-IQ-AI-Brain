"""
PlantIQ AI Brain - Worker Performance Analytics
Feature engineering from attendance, tasks, and quality data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def compute_attendance_features(attendance_df: pd.DataFrame, worker_id: str,
                                  period_days: int = 30) -> Dict:
    """Compute attendance features for a worker over a period."""
    df = attendance_df[attendance_df["worker_id"] == worker_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=period_days)
    df = df[df["date"] >= cutoff]

    total_days = len(df)
    present_days = len(df[df["status"] == "present"])
    late_days = len(df[df["is_late"] == True])
    total_hours = df["work_hours"].sum()
    overtime_hours = df["overtime_hours"].sum()

    # Pattern detection for absenteeism
    df["day_of_week"] = df["date"].dt.dayofweek
    absent_days = df[df["status"] == "absent"]
    day_absent_counts = absent_days["day_of_week"].value_counts()
    most_absent_day = day_absent_counts.index[0] if len(day_absent_counts) > 0 else -1

    return {
        "total_working_days": total_days,
        "days_present": present_days,
        "days_absent": total_days - present_days,
        "attendance_rate": present_days / max(1, total_days),
        "late_arrivals": late_days,
        "on_time_rate": 1 - late_days / max(1, present_days),
        "total_hours": total_hours,
        "avg_daily_hours": total_hours / max(1, present_days),
        "overtime_hours": overtime_hours,
        "overtime_rate": overtime_hours / max(1, total_hours),
        "most_absent_day": most_absent_day,
        "consecutive_absences_max": _max_consecutive_absences(df),
    }


def _max_consecutive_absences(df: pd.DataFrame) -> int:
    """Calculate maximum consecutive absences."""
    if df.empty:
        return 0
    absent_mask = df.sort_values("date")["status"] == "absent"
    max_consecutive = 0
    current = 0
    for is_absent in absent_mask:
        if is_absent:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0
    return max_consecutive


def compute_productivity_features(task_df: pd.DataFrame, worker_id: str,
                                    period_days: int = 30) -> Dict:
    """Compute productivity features for a worker."""
    df = task_df[task_df["worker_id"] == worker_id].copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=period_days)
    df = df[df["date"] >= cutoff]

    total_tasks = len(df)
    completed = len(df[df["status"] == "completed"])
    avg_quality = df[df["status"] == "completed"]["quality_score"].mean() if completed > 0 else 0

    # Time efficiency
    df_completed = df[df["status"] == "completed"]
    if len(df_completed) > 0:
        time_efficiency = (df_completed["estimated_hours"].sum() /
                          max(0.01, df_completed["actual_hours"].sum()))
    else:
        time_efficiency = 0

    # Errors and rework
    total_errors = df["errors_count"].sum()
    rework_count = df["rework_needed"].sum()

    # Task type breakdown
    task_type_counts = df["task_type"].value_counts().to_dict()

    return {
        "tasks_assigned": total_tasks,
        "tasks_completed": completed,
        "completion_rate": completed / max(1, total_tasks),
        "avg_quality_score": avg_quality,
        "time_efficiency": time_efficiency,
        "total_errors": int(total_errors),
        "error_rate": total_errors / max(1, completed),
        "rework_count": int(rework_count),
        "rework_rate": rework_count / max(1, completed),
        "task_type_distribution": task_type_counts,
        "tasks_per_day": total_tasks / max(1, period_days),
    }


def compute_worker_score(attendance_feats: Dict, productivity_feats: Dict) -> Dict:
    """Compute composite worker performance score (0-100)."""
    # Attendance score (0-100)
    attendance_score = (
        attendance_feats["attendance_rate"] * 60
        + attendance_feats["on_time_rate"] * 30
        + min(1, attendance_feats["avg_daily_hours"] / 9) * 10
    )

    # Productivity score (0-100)
    productivity_score = (
        productivity_feats["completion_rate"] * 40
        + min(1, productivity_feats["time_efficiency"]) * 30
        + (productivity_feats["avg_quality_score"] / 100) * 30
    )

    # Quality score (0-100)
    error_penalty = min(30, productivity_feats["error_rate"] * 100)
    rework_penalty = min(20, productivity_feats["rework_rate"] * 50)
    quality_score = max(0, 100 - error_penalty - rework_penalty)

    # Initiative score (approximated from task diversity and proactivity)
    task_diversity = min(1, len(productivity_feats.get("task_type_distribution", {})) / 8)
    initiative_score = task_diversity * 60 + 40  # Base 40 for showing up

    # Weighted composite
    weights = {
        "attendance": 0.25,
        "productivity": 0.30,
        "quality": 0.30,
        "initiative": 0.15,
    }

    composite = (
        attendance_score * weights["attendance"]
        + productivity_score * weights["productivity"]
        + quality_score * weights["quality"]
        + initiative_score * weights["initiative"]
    )

    return {
        "composite_score": round(min(100, max(0, composite)), 1),
        "attendance_score": round(min(100, attendance_score), 1),
        "productivity_score": round(min(100, productivity_score), 1),
        "quality_score": round(min(100, quality_score), 1),
        "initiative_score": round(min(100, initiative_score), 1),
    }


def compute_burnout_features(attendance_df: pd.DataFrame, task_df: pd.DataFrame,
                               worker_id: str, weeks: int = 4) -> Dict:
    """Compute burnout risk features."""
    att = attendance_df[attendance_df["worker_id"] == worker_id].copy()
    att["date"] = pd.to_datetime(att["date"])
    cutoff = att["date"].max() - pd.Timedelta(weeks=weeks)
    att = att[att["date"] >= cutoff]

    tasks = task_df[task_df["worker_id"] == worker_id].copy()
    tasks["date"] = pd.to_datetime(tasks["date"])
    tasks = tasks[tasks["date"] >= cutoff]

    # Weekly hours
    att_present = att[att["status"] == "present"]
    weekly_hours = att_present.groupby(att_present["date"].dt.isocalendar().week)["work_hours"].sum()
    max_weekly = weekly_hours.max() if len(weekly_hours) > 0 else 0
    avg_weekly = weekly_hours.mean() if len(weekly_hours) > 0 else 0

    # Quality trend
    tasks_completed = tasks[tasks["status"] == "completed"]
    if len(tasks_completed) > 10:
        recent_quality = tasks_completed.tail(len(tasks_completed)//2)["quality_score"].mean()
        old_quality = tasks_completed.head(len(tasks_completed)//2)["quality_score"].mean()
        quality_trend = recent_quality - old_quality
    else:
        quality_trend = 0

    # Overtime trend
    overtime_total = att_present["overtime_hours"].sum()

    return {
        "avg_weekly_hours": round(avg_weekly, 1),
        "max_weekly_hours": round(max_weekly, 1),
        "weeks_over_50hrs": int((weekly_hours > 50).sum()),
        "overtime_total": round(overtime_total, 1),
        "quality_trend": round(quality_trend, 1),
        "task_count_recent": len(tasks),
    }
