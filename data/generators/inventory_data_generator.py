"""
PlantIQ AI Brain - Inventory Data Generator
Generates inventory levels, consumption logs, and reorder history.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

INVENTORY_ITEMS = [
    {"name": "NPK Fertilizer 15-15-15", "unit": "kg", "daily_usage": 3, "unit_cost": 45, "reorder_point": 20, "reorder_qty": 100},
    {"name": "NPK Fertilizer 20-20-20", "unit": "kg", "daily_usage": 2, "unit_cost": 55, "reorder_point": 15, "reorder_qty": 80},
    {"name": "Urea", "unit": "kg", "daily_usage": 1.5, "unit_cost": 30, "reorder_point": 10, "reorder_qty": 50},
    {"name": "Copper Fungicide", "unit": "kg", "daily_usage": 0.5, "unit_cost": 280, "reorder_point": 3, "reorder_qty": 10},
    {"name": "Neem Oil", "unit": "liters", "daily_usage": 0.3, "unit_cost": 350, "reorder_point": 2, "reorder_qty": 10},
    {"name": "Grafting Tape", "unit": "rolls", "daily_usage": 1.5, "unit_cost": 120, "reorder_point": 10, "reorder_qty": 50},
    {"name": "Grafting Wax", "unit": "kg", "daily_usage": 0.3, "unit_cost": 200, "reorder_point": 2, "reorder_qty": 5},
    {"name": "Potting Mix", "unit": "bags", "daily_usage": 5, "unit_cost": 85, "reorder_point": 20, "reorder_qty": 100},
    {"name": "Poly Bags", "unit": "packs", "daily_usage": 2, "unit_cost": 150, "reorder_point": 10, "reorder_qty": 50},
    {"name": "Insecticide Spray", "unit": "liters", "daily_usage": 0.2, "unit_cost": 450, "reorder_point": 2, "reorder_qty": 10},
    {"name": "Root Hormone", "unit": "grams", "daily_usage": 10, "unit_cost": 5, "reorder_point": 100, "reorder_qty": 500},
    {"name": "Shade Net Material", "unit": "meters", "daily_usage": 0.5, "unit_cost": 60, "reorder_point": 20, "reorder_qty": 100},
]


def generate_inventory_data(days: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate daily inventory levels and consumption logs."""
    days = days or config.HISTORY_DAYS
    output_path = output_path or str(config.DATA_DIR / "inventory_data.csv")
    np.random.seed(config.RANDOM_STATE + 30)

    records = []
    start_date = datetime(2025, 4, 10)

    for item in INVENTORY_ITEMS:
        stock = item["reorder_qty"] * 1.5  # Start with decent stock

        for day in range(days):
            current_date = start_date + timedelta(days=day)

            # Skip Sundays
            if current_date.weekday() == 6:
                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "item_name": item["name"],
                    "unit": item["unit"],
                    "opening_stock": round(stock, 1),
                    "consumed": 0,
                    "received": 0,
                    "closing_stock": round(stock, 1),
                    "unit_cost": item["unit_cost"],
                    "daily_cost": 0,
                    "below_reorder": stock < item["reorder_point"],
                })
                continue

            # Daily consumption with variation + seasonal factors
            seasonal = 1.0
            month = current_date.month
            if item["name"].startswith("Grafting") and month in [3, 4, 5, 9, 10]:
                seasonal = 1.8
            if "Fungicide" in item["name"] and month in [7, 8, 9]:
                seasonal = 2.5

            daily_usage = item["daily_usage"] * seasonal * np.random.uniform(0.5, 1.5)
            daily_usage = max(0, daily_usage)

            # Reorder check
            received = 0
            if stock < item["reorder_point"]:
                # Order arrives with 3-7 day delay (simplified: arrives today sometimes)
                if np.random.random() < 0.3:
                    received = item["reorder_qty"]

            opening = stock
            stock = stock - daily_usage + received
            stock = max(0, stock)

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "item_name": item["name"],
                "unit": item["unit"],
                "opening_stock": round(opening, 1),
                "consumed": round(daily_usage, 2),
                "received": round(received, 1),
                "closing_stock": round(stock, 1),
                "unit_cost": item["unit_cost"],
                "daily_cost": round(daily_usage * item["unit_cost"], 0),
                "below_reorder": stock < item["reorder_point"],
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Inventory data generated: {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    generate_inventory_data()
