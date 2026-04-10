"""
PlantIQ AI Brain - Graft Data Generator
Generates graft records with rootstock/scion details, methods, outcomes.
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


def generate_graft_records(output_path: str = None) -> pd.DataFrame:
    """Generate graft records with detailed success/failure data."""
    output_path = output_path or str(config.DATA_DIR / "graft_records.csv")
    np.random.seed(config.RANDOM_STATE + 20)

    records = []
    start_date = datetime(2025, 4, 10)

    # Worker skill levels per graft method
    worker_skills = {}
    for i, name in enumerate(WORKER_NAMES):
        worker_skills[f"W{i+1:03d}"] = {
            method: np.clip(np.random.normal(0.75, 0.12), 0.4, 0.98)
            for method in config.GRAFT_METHODS
        }

    graft_counter = 0
    for day in range(config.HISTORY_DAYS):
        current_date = start_date + timedelta(days=day)

        # Skip Sundays, grafting mostly in spring/autumn
        if current_date.weekday() == 6:
            continue

        month = current_date.month
        if month in [3, 4, 5, 9, 10]:
            daily_grafts = np.random.randint(8, 20)
        elif month in [6, 7, 8]:
            daily_grafts = np.random.randint(3, 10)
        else:
            daily_grafts = np.random.randint(0, 5)

        for _ in range(daily_grafts):
            graft_counter += 1
            method = np.random.choice(config.GRAFT_METHODS)
            worker_idx = np.random.randint(0, len(WORKER_NAMES))
            worker_id = f"W{worker_idx+1:03d}"
            worker_name = WORKER_NAMES[worker_idx]

            rootstock_variety = np.random.choice(config.PLANT_VARIETIES)
            scion_variety = np.random.choice(config.PLANT_VARIETIES)
            rootstock_age_days = np.random.randint(120, 300)
            scion_freshness_days = np.random.randint(0, 7)

            # Environmental conditions at time of grafting
            temp = np.random.normal(23, 5)
            humidity = np.random.normal(65, 10)

            # Success factors
            worker_skill = worker_skills[worker_id][method]
            age_factor = 1.0 if 150 <= rootstock_age_days <= 240 else 0.85
            freshness_factor = 1.0 - scion_freshness_days * 0.03
            temp_factor = 1.0 if 18 <= temp <= 28 else 0.8
            humidity_factor = 1.0 if 50 <= humidity <= 75 else 0.85

            # Overall success probability
            success_prob = (
                worker_skill * 0.35
                + age_factor * 0.20
                + freshness_factor * 0.15
                + temp_factor * 0.15
                + humidity_factor * 0.15
            )
            success_prob = np.clip(success_prob, 0.1, 0.98)

            is_success = np.random.random() < success_prob

            # Post-graft monitoring data (7-day and 14-day)
            callus_formation = np.random.uniform(60, 100) if is_success else np.random.uniform(10, 60)
            cambium_alignment = np.random.uniform(70, 100) if is_success else np.random.uniform(20, 70)

            records.append({
                "graft_id": f"GR-2025-{graft_counter:05d}",
                "date": current_date.strftime("%Y-%m-%d"),
                "worker_id": worker_id,
                "worker_name": worker_name,
                "method": method,
                "rootstock_variety": rootstock_variety["name"],
                "rootstock_age_days": rootstock_age_days,
                "scion_variety": scion_variety["name"],
                "scion_freshness_days": scion_freshness_days,
                "temperature_at_graft": round(temp, 1),
                "humidity_at_graft": round(humidity, 1),
                "time_of_day": np.random.choice(["morning", "afternoon", "evening"], p=[0.5, 0.35, 0.15]),
                "callus_formation_pct": round(callus_formation, 1),
                "cambium_alignment_pct": round(cambium_alignment, 1),
                "success": is_success,
                "success_probability": round(success_prob, 3),
                "failure_reason": None if is_success else np.random.choice([
                    "poor_callus_formation", "desiccation", "infection",
                    "cambium_misalignment", "rootstock_rejection"
                ]),
                "zone_id": np.random.choice([z["id"] for z in config.ZONES]),
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Graft records generated: {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    generate_graft_records()
