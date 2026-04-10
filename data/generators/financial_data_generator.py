"""
PlantIQ AI Brain - Financial Data Generator
Generates sales orders, expenses, revenue, and payments.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

CUSTOMERS = [
    "Himalayan Horticulture", "Valley Green Farms", "Kashmir Orchards Ltd",
    "Mountain View Nurseries", "Alpine Agriculture", "Green Earth Foundation",
    "Nature's Best Nursery", "Fruit Valley Co-op", "Highland Plantations",
    "Eco Green Farms", "Rural Development Society", "Individual - Farmer",
    "Government Agriculture Dept", "University Research Farm", "Export - Dubai Traders"
]

EXPENSE_CATEGORIES = [
    {"name": "Labor - Regular", "monthly_avg": 100000, "variation": 0.05},
    {"name": "Labor - Overtime", "monthly_avg": 15000, "variation": 0.40},
    {"name": "Fertilizers", "monthly_avg": 35000, "variation": 0.25},
    {"name": "Pesticides & Fungicides", "monthly_avg": 18000, "variation": 0.35},
    {"name": "Seeds & Rootstock", "monthly_avg": 25000, "variation": 0.30},
    {"name": "Electricity", "monthly_avg": 12000, "variation": 0.15},
    {"name": "Water", "monthly_avg": 6000, "variation": 0.20},
    {"name": "Equipment Maintenance", "monthly_avg": 8000, "variation": 0.40},
    {"name": "Transport", "monthly_avg": 10000, "variation": 0.25},
    {"name": "Packaging Materials", "monthly_avg": 7000, "variation": 0.20},
    {"name": "Miscellaneous", "monthly_avg": 5000, "variation": 0.50},
]


def generate_sales_data(days: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate sales order data with seasonal patterns."""
    days = days or config.HISTORY_DAYS
    output_path = output_path or str(config.DATA_DIR / "sales_data.csv")
    np.random.seed(config.RANDOM_STATE + 40)

    records = []
    start_date = datetime(2025, 4, 10)
    order_counter = 0

    for day in range(days):
        current_date = start_date + timedelta(days=day)

        # Sales have strong seasonality (peak in spring and autumn)
        month = current_date.month
        if month in [3, 4, 5]:
            daily_orders = np.random.poisson(4)
        elif month in [9, 10, 11]:
            daily_orders = np.random.poisson(3)
        elif month in [6, 7, 8]:
            daily_orders = np.random.poisson(1)
        else:
            daily_orders = np.random.poisson(2)

        for _ in range(daily_orders):
            order_counter += 1
            variety = np.random.choice(config.PLANT_VARIETIES)
            customer = np.random.choice(CUSTOMERS)
            quantity = np.random.choice([5, 10, 15, 20, 25, 50, 100], p=[0.15, 0.25, 0.20, 0.15, 0.10, 0.10, 0.05])

            # Price based on variety + quality grade
            grade = np.random.choice(["A+", "A", "B", "C"], p=[0.15, 0.45, 0.30, 0.10])
            grade_multiplier = {"A+": 1.3, "A": 1.0, "B": 0.75, "C": 0.50}[grade]
            unit_price = int(np.mean(variety["price_range"]) * grade_multiplier)
            total_amount = unit_price * quantity

            # Payment status
            payment_status = np.random.choice(
                ["paid", "partial", "pending"],
                p=[0.60, 0.15, 0.25]
            )
            amount_paid = total_amount if payment_status == "paid" else (
                total_amount * np.random.uniform(0.3, 0.7) if payment_status == "partial" else 0
            )

            records.append({
                "order_id": f"ORD-{order_counter:05d}",
                "date": current_date.strftime("%Y-%m-%d"),
                "customer_name": customer,
                "variety_name": variety["name"],
                "variety_code": variety["code"],
                "quality_grade": grade,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_amount": total_amount,
                "payment_status": payment_status,
                "amount_paid": round(amount_paid, 0),
                "amount_pending": round(total_amount - amount_paid, 0),
                "delivery_status": np.random.choice(["delivered", "in_transit", "processing"], p=[0.7, 0.15, 0.15]),
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Sales data generated: {len(df)} orders → {output_path}")
    return df


def generate_expense_data(days: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate daily expense records by category."""
    days = days or config.HISTORY_DAYS
    output_path = output_path or str(config.DATA_DIR / "expense_data.csv")
    np.random.seed(config.RANDOM_STATE + 41)

    records = []
    start_date = datetime(2025, 4, 10)

    for day in range(days):
        current_date = start_date + timedelta(days=day)

        for category in EXPENSE_CATEGORIES:
            # Daily amount = monthly / 30 with variation
            daily_avg = category["monthly_avg"] / 30
            daily_amount = daily_avg * np.random.uniform(
                1 - category["variation"],
                1 + category["variation"]
            )

            # Seasonal adjustments
            month = current_date.month
            if "Fertilizer" in category["name"] and month in [4, 5, 6]:
                daily_amount *= 1.3
            if "Pest" in category["name"] and month in [7, 8, 9]:
                daily_amount *= 2.0
            if "Overtime" in category["name"] and month in [3, 4, 5]:
                daily_amount *= 1.5

            # Sundays - only essential expenses
            if current_date.weekday() == 6:
                if "Labor" in category["name"]:
                    daily_amount = 0
                else:
                    daily_amount *= 0.1

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "category": category["name"],
                "amount": round(max(0, daily_amount), 0),
                "month": current_date.strftime("%Y-%m"),
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Expense data generated: {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    generate_sales_data()
    generate_expense_data()
