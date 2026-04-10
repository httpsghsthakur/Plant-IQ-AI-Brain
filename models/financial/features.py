"""
PlantIQ AI Brain - Financial Analytics
Feature engineering from sales, expenses, and operational data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List


def compute_revenue_features(sales_df: pd.DataFrame, months: int = 3) -> Dict:
    """Compute revenue features from sales data."""
    df = sales_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Monthly revenue
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month").agg(
        revenue=("total_amount", "sum"),
        orders=("order_id", "size"),
        quantity=("quantity", "sum"),
        avg_price=("unit_price", "mean"),
    ).reset_index()

    if monthly.empty:
        return {"monthly_avg_revenue": 0, "trend": 0, "total_revenue": 0}

    # Revenue trend
    revenues = monthly["revenue"].values
    trend = 0
    if len(revenues) >= 2:
        x = np.arange(len(revenues))
        coeffs = np.polyfit(x, revenues, 1)
        trend = coeffs[0]

    # By variety
    variety_revenue = df.groupby("variety_name")["total_amount"].sum().to_dict()

    # Payment analysis
    total_billed = df["total_amount"].sum()
    total_paid = df["amount_paid"].sum()
    collection_rate = total_paid / max(1, total_billed)

    return {
        "monthly_avg_revenue": round(monthly["revenue"].mean(), 0),
        "total_revenue": round(total_billed, 0),
        "total_collected": round(total_paid, 0),
        "outstanding": round(total_billed - total_paid, 0),
        "collection_rate": round(collection_rate, 4),
        "trend": round(trend, 0),
        "trend_direction": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
        "variety_revenue": variety_revenue,
        "monthly_data": monthly.to_dict("records") if len(monthly) > 0 else [],
        "orders_count": int(df["order_id"].nunique()),
    }


def compute_expense_features(expense_df: pd.DataFrame) -> Dict:
    """Compute expense features and breakdown."""
    df = expense_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Monthly totals
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum()

    # By category
    category_totals = df.groupby("category")["amount"].sum().to_dict()
    total_expenses = df["amount"].sum()

    # Category trends
    category_trends = {}
    for cat in df["category"].unique():
        cat_data = df[df["category"] == cat]
        cat_monthly = cat_data.groupby("month")["amount"].sum()
        if len(cat_monthly) >= 2:
            trend = cat_monthly.iloc[-1] - cat_monthly.iloc[0]
            category_trends[cat] = "increasing" if trend > 0 else "decreasing"
        else:
            category_trends[cat] = "stable"

    return {
        "total_expenses": round(total_expenses, 0),
        "monthly_avg": round(monthly.mean() if len(monthly) > 0 else 0, 0),
        "category_breakdown": category_totals,
        "category_trends": category_trends,
        "top_expense": max(category_totals, key=category_totals.get) if category_totals else "Unknown",
    }


def compute_profitability_features(sales_df: pd.DataFrame, expense_df: pd.DataFrame) -> Dict:
    """Compute profitability metrics."""
    revenue = compute_revenue_features(sales_df)
    expenses = compute_expense_features(expense_df)

    total_revenue = revenue["total_revenue"]
    total_expenses = expenses["total_expenses"]
    gross_profit = total_revenue - total_expenses
    profit_margin = gross_profit / max(1, total_revenue)

    return {
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "gross_profit": gross_profit,
        "profit_margin": round(profit_margin, 4),
        "monthly_avg_profit": round(
            revenue["monthly_avg_revenue"] - expenses["monthly_avg"], 0
        ),
        "target_margin": config.TARGET_PROFIT_MARGIN,
        "margin_gap": round(profit_margin - config.TARGET_PROFIT_MARGIN, 4),
    }


# Need config import for compute_profitability_features
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
