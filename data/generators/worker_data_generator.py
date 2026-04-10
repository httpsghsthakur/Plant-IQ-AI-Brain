"""
PlantIQ AI Brain - Worker Data Generator
Generates worker profiles, attendance records, task data, and performance metrics.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

WORKER_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Mohammad Iqbal", "Sunita Devi", "Amit Singh",
    "Kavita Devi", "Vikram Thakur", "Fatima Begum", "Ravi Shankar", "Meena Kumari",
    "Deepak Negi", "Anita Rana", "Suresh Bhatt", "Geeta Rawat", "Abdul Rashid"
]

TASK_TYPES = [
    {"name": "Watering", "avg_duration_hrs": 1.5, "skill_level": "basic"},
    {"name": "Fertilizer Application", "avg_duration_hrs": 2.0, "skill_level": "basic"},
    {"name": "Grafting - Whip and Tongue", "avg_duration_hrs": 0.5, "skill_level": "expert"},
    {"name": "Grafting - Bud", "avg_duration_hrs": 0.4, "skill_level": "intermediate"},
    {"name": "Grafting - Cleft", "avg_duration_hrs": 0.6, "skill_level": "expert"},
    {"name": "Pruning", "avg_duration_hrs": 1.0, "skill_level": "intermediate"},
    {"name": "Pest Inspection", "avg_duration_hrs": 2.5, "skill_level": "intermediate"},
    {"name": "Transplanting", "avg_duration_hrs": 1.5, "skill_level": "basic"},
    {"name": "Disease Treatment", "avg_duration_hrs": 1.0, "skill_level": "intermediate"},
    {"name": "Soil Preparation", "avg_duration_hrs": 3.0, "skill_level": "basic"},
    {"name": "Shade Net Management", "avg_duration_hrs": 1.0, "skill_level": "basic"},
    {"name": "Growth Measurement", "avg_duration_hrs": 2.0, "skill_level": "intermediate"},
    {"name": "Packing & Dispatch", "avg_duration_hrs": 2.0, "skill_level": "basic"},
    {"name": "Equipment Maintenance", "avg_duration_hrs": 1.5, "skill_level": "intermediate"},
]


def generate_worker_profiles(output_path: str = None) -> pd.DataFrame:
    """Generate worker profiles with skills and attributes."""
    np.random.seed(config.RANDOM_STATE)
    output_path = output_path or str(config.DATA_DIR / "worker_profiles.csv")

    workers = []
    for i, name in enumerate(WORKER_NAMES):
        experience_years = np.random.randint(1, 12)
        base_skill = min(0.95, 0.5 + experience_years * 0.04 + np.random.normal(0, 0.05))

        workers.append({
            "worker_id": f"W{i+1:03d}",
            "name": name,
            "age": np.random.randint(22, 55),
            "experience_years": experience_years,
            "base_skill_level": round(base_skill, 3),
            "specialization": np.random.choice(["grafting", "irrigation", "pest_management", "general"]),
            "daily_wage": 500 + experience_years * 50 + np.random.randint(0, 100),
            "hire_date": (datetime(2025, 4, 10) - timedelta(days=experience_years * 365 + np.random.randint(0, 365))).strftime("%Y-%m-%d"),
            "reliability_score": round(np.clip(np.random.normal(0.85, 0.1), 0.5, 1.0), 3),
        })

    df = pd.DataFrame(workers)
    df.to_csv(output_path, index=False)
    print(f"✅ Worker profiles generated: {len(df)} workers → {output_path}")
    return df


def generate_attendance_records(days: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate NFC attendance records with realistic patterns."""
    days = days or config.HISTORY_DAYS
    output_path = output_path or str(config.DATA_DIR / "attendance_records.csv")
    np.random.seed(config.RANDOM_STATE + 1)

    records = []
    start_date = datetime(2025, 4, 10)

    for i, name in enumerate(WORKER_NAMES):
        worker_id = f"W{i+1:03d}"
        reliability = 0.85 + np.random.uniform(-0.1, 0.1)

        for day in range(days):
            current_date = start_date + timedelta(days=day)

            # Skip Sundays (off day)
            if current_date.weekday() == 6:
                continue

            # Random absences based on reliability
            is_present = np.random.random() < reliability

            # Saturday pattern - some workers skip
            if current_date.weekday() == 5:
                is_present = is_present and (np.random.random() < 0.7)

            if is_present:
                # Check-in time (normal around 7:00 AM)
                checkin_hour = 7
                checkin_minute = int(np.clip(np.random.normal(0, 10), -15, 30))
                checkin_time = current_date.replace(hour=checkin_hour, minute=max(0, checkin_minute))

                # Late arrival probability
                if np.random.random() < 0.08:  # 8% chance of being late
                    checkin_time = current_date.replace(
                        hour=np.random.choice([8, 9, 10, 11]),
                        minute=np.random.randint(0, 59)
                    )

                # Check-out time (normal around 5:00 PM)
                checkout_hour = 17
                checkout_minute = int(np.clip(np.random.normal(0, 15), -30, 60))
                checkout_time = current_date.replace(hour=checkout_hour, minute=max(0, min(59, checkout_minute)))

                # Overtime (10% chance)
                if np.random.random() < 0.10:
                    checkout_time = current_date.replace(
                        hour=np.random.choice([18, 19, 20]),
                        minute=np.random.randint(0, 59)
                    )

                work_hours = (checkout_time - checkin_time).seconds / 3600

                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "worker_id": worker_id,
                    "worker_name": name,
                    "status": "present",
                    "checkin_time": checkin_time.strftime("%H:%M"),
                    "checkout_time": checkout_time.strftime("%H:%M"),
                    "work_hours": round(work_hours, 1),
                    "is_late": checkin_time.hour >= 8,
                    "overtime_hours": round(max(0, work_hours - 9), 1),
                    "nfc_scan_location": np.random.choice(["main_gate", "greenhouse_a", "office"]),
                })
            else:
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "worker_id": worker_id,
                    "worker_name": name,
                    "status": "absent",
                    "checkin_time": None,
                    "checkout_time": None,
                    "work_hours": 0,
                    "is_late": False,
                    "overtime_hours": 0,
                    "nfc_scan_location": None,
                })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Attendance records generated: {len(df)} records → {output_path}")
    return df


def generate_task_records(days: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate task assignment and completion records."""
    days = days or config.HISTORY_DAYS
    output_path = output_path or str(config.DATA_DIR / "task_records.csv")
    np.random.seed(config.RANDOM_STATE + 2)

    records = []
    start_date = datetime(2025, 4, 10)
    task_counter = 0

    for day in range(days):
        current_date = start_date + timedelta(days=day)
        if current_date.weekday() == 6:  # Skip Sunday
            continue

        # Each worker gets 2-5 tasks per day
        for i, name in enumerate(WORKER_NAMES):
            worker_id = f"W{i+1:03d}"
            num_tasks = np.random.randint(2, 6)

            for _ in range(num_tasks):
                task_counter += 1
                task = np.random.choice(TASK_TYPES)
                zone = np.random.choice(config.ZONES)

                # Task completion with skill-based variation
                skill_factor = 0.8 + np.random.uniform(0, 0.2)
                estimated_hours = task["avg_duration_hrs"]
                actual_hours = estimated_hours * (1 / skill_factor) + np.random.normal(0, 0.2)
                actual_hours = max(0.1, actual_hours)

                # Completion probability
                is_completed = np.random.random() < 0.92
                quality_score = np.clip(np.random.normal(80, 12), 30, 100) if is_completed else 0

                records.append({
                    "task_id": f"T{task_counter:06d}",
                    "date": current_date.strftime("%Y-%m-%d"),
                    "worker_id": worker_id,
                    "worker_name": name,
                    "task_type": task["name"],
                    "skill_level_required": task["skill_level"],
                    "zone_id": zone["id"],
                    "estimated_hours": round(estimated_hours, 2),
                    "actual_hours": round(actual_hours, 2),
                    "status": "completed" if is_completed else np.random.choice(["incomplete", "in_progress"]),
                    "quality_score": round(quality_score, 1),
                    "errors_count": max(0, int(np.random.exponential(0.3))) if is_completed else 0,
                    "rework_needed": np.random.random() < 0.05 if is_completed else False,
                })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Task records generated: {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    generate_worker_profiles()
    generate_attendance_records()
    generate_task_records()
