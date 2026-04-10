"""
PlantIQ AI Brain - Plant Data Generator
Generates plant inventory, growth measurements, disease records, and mortality data.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def generate_plant_inventory(num_plants: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate plant inventory with variety, zone, and stage information."""
    num_plants = num_plants or config.NUM_PLANTS
    output_path = output_path or str(config.DATA_DIR / "plant_inventory.csv")
    np.random.seed(config.RANDOM_STATE + 10)

    plants = []
    start_date = datetime(2025, 4, 10)

    for i in range(num_plants):
        variety = np.random.choice(config.PLANT_VARIETIES,
                                   p=[0.35, 0.25, 0.15, 0.15, 0.10])
        zone = np.random.choice(config.ZONES)
        plant_age_days = np.random.randint(7, 365)
        planting_date = start_date - timedelta(days=plant_age_days)

        # Determine stage based on age
        if plant_age_days < 30:
            stage = "seedling"
        elif plant_age_days < 90:
            stage = "young"
        elif plant_age_days < 180:
            stage = "growing"
        elif plant_age_days < 270:
            stage = "mature"
        else:
            stage = "ready_to_sell"

        # Health status
        health_roll = np.random.random()
        if health_roll < 0.85:
            health_status = "healthy"
        elif health_roll < 0.95:
            health_status = "at_risk"
        else:
            health_status = "critical"

        # Growth rate based on variety + random factors
        expected_height = plant_age_days * variety["growth_rate"]
        actual_height = expected_height * np.random.normal(1.0, 0.15)
        actual_height = max(1, actual_height)

        # Girth (proportional to height with noise)
        expected_girth = plant_age_days * variety["growth_rate"] * 0.08
        actual_girth = expected_girth * np.random.normal(1.0, 0.12)
        actual_girth = max(0.1, actual_girth)

        # Is this plant grafted?
        is_grafted = plant_age_days > 120 and np.random.random() < 0.6
        graft_success = None
        if is_grafted:
            graft_success = np.random.random() < 0.82

        plants.append({
            "plant_id": f"WN-2024-{i+1:04d}",
            "variety_name": variety["name"],
            "variety_code": variety["code"],
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "planting_date": planting_date.strftime("%Y-%m-%d"),
            "age_days": plant_age_days,
            "stage": stage,
            "health_status": health_status,
            "current_height_cm": round(actual_height, 1),
            "expected_height_cm": round(expected_height, 1),
            "height_deviation_pct": round((actual_height - expected_height) / expected_height * 100, 1),
            "current_girth_mm": round(actual_girth, 1),
            "is_grafted": is_grafted,
            "graft_success": graft_success,
            "price_range_low": variety["price_range"][0],
            "price_range_high": variety["price_range"][1],
        })

    df = pd.DataFrame(plants)
    df.to_csv(output_path, index=False)
    print(f"✅ Plant inventory generated: {len(df)} plants → {output_path}")
    return df


def generate_growth_measurements(num_plants: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate weekly growth measurements for all plants."""
    num_plants = num_plants or min(config.NUM_PLANTS, 1000)  # Sample for performance
    output_path = output_path or str(config.DATA_DIR / "growth_measurements.csv")
    np.random.seed(config.RANDOM_STATE + 11)

    records = []
    start_date = datetime(2025, 4, 10)

    for i in range(num_plants):
        variety = np.random.choice(config.PLANT_VARIETIES,
                                   p=[0.35, 0.25, 0.15, 0.15, 0.10])
        plant_id = f"WN-2024-{i+1:04d}"
        zone = np.random.choice(config.ZONES)

        # Generate weekly measurements over the plant's life
        plant_age_days = np.random.randint(30, 365)
        num_weeks = plant_age_days // 7

        height = np.random.uniform(2, 5)  # Starting height
        girth = np.random.uniform(1, 3)   # Starting girth

        for week in range(num_weeks):
            measurement_date = start_date - timedelta(days=plant_age_days) + timedelta(weeks=week)

            # Weekly growth rate with seasonal and random variation
            day_of_year = measurement_date.timetuple().tm_yday
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

            # Environmental stress (random)
            stress_factor = 1.0
            if np.random.random() < 0.15:
                stress_factor = np.random.uniform(0.5, 0.8)

            weekly_growth = variety["growth_rate"] * 7 * seasonal_factor * stress_factor
            weekly_growth += np.random.normal(0, 0.3)
            weekly_growth = max(0, weekly_growth)

            height += weekly_growth
            girth += weekly_growth * 0.08 + np.random.normal(0, 0.05)

            # Disease detection (random events)
            disease_detected = None
            if np.random.random() < 0.03:
                disease_detected = np.random.choice([d["name"] for d in config.DISEASES])

            records.append({
                "plant_id": plant_id,
                "variety_name": variety["name"],
                "zone_id": zone["id"],
                "measurement_date": measurement_date.strftime("%Y-%m-%d"),
                "week_number": week + 1,
                "height_cm": round(height, 1),
                "girth_mm": round(max(0.5, girth), 1),
                "weekly_growth_cm": round(weekly_growth, 2),
                "leaf_count": max(3, int(height * 1.2 + np.random.normal(0, 2))),
                "disease_detected": disease_detected,
                "stress_indicators": "water_stress" if stress_factor < 0.9 else "none",
                "health_score": round(np.clip(
                    80 + 10 * stress_factor + np.random.normal(0, 5), 0, 100
                ), 1),
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Growth measurements generated: {len(df)} records → {output_path}")
    return df


def generate_disease_records(num_plants: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate disease detection and treatment records."""
    num_plants = num_plants or config.NUM_PLANTS
    output_path = output_path or str(config.DATA_DIR / "disease_records.csv")
    np.random.seed(config.RANDOM_STATE + 12)

    records = []
    start_date = datetime(2025, 4, 10)
    case_counter = 0

    for day in range(config.HISTORY_DAYS):
        current_date = start_date + timedelta(days=day)

        # Number of new cases per day (seasonal variation)
        month = current_date.month
        if month in [7, 8, 9]:  # Monsoon - more diseases
            daily_cases = np.random.poisson(3)
        elif month in [4, 5, 6, 10]:
            daily_cases = np.random.poisson(1)
        else:
            daily_cases = np.random.poisson(0.5)

        for _ in range(daily_cases):
            case_counter += 1
            disease = np.random.choice(config.DISEASES)
            plant_id = f"WN-2024-{np.random.randint(1, num_plants+1):04d}"
            zone = np.random.choice(config.ZONES)
            severity = np.random.choice(["mild", "moderate", "severe"], p=[0.5, 0.35, 0.15])

            # Treatment
            treatment_applied = np.random.random() < 0.90
            treatment_success = False
            if treatment_applied:
                if severity == "mild":
                    treatment_success = np.random.random() < 0.90
                elif severity == "moderate":
                    treatment_success = np.random.random() < 0.70
                else:
                    treatment_success = np.random.random() < 0.45

            records.append({
                "case_id": f"DIS-{case_counter:05d}",
                "date_detected": current_date.strftime("%Y-%m-%d"),
                "plant_id": plant_id,
                "zone_id": zone["id"],
                "disease_name": disease["name"],
                "severity": severity,
                "treatment_applied": treatment_applied,
                "treatment_type": np.random.choice(["fungicide", "bactericide", "insecticide", "cultural_practice"]) if treatment_applied else None,
                "treatment_cost": round(np.random.uniform(50, 500), 0) if treatment_applied else 0,
                "treatment_success": treatment_success,
                "days_to_recovery": np.random.randint(5, 30) if treatment_success else None,
                "plant_survived": treatment_success or (np.random.random() < 0.3),
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Disease records generated: {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    generate_plant_inventory()
    generate_growth_measurements()
    generate_disease_records()
