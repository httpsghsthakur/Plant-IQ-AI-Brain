"""
PlantIQ AI Brain - Financial Analytics Model
Profitability analysis, cash flow forecasting, cost optimization, pricing recommendations.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from models.financial.features import (
    compute_revenue_features,
    compute_expense_features,
    compute_profitability_features,
)
from models.financial.schemas import (
    ProfitLossReport, CashFlowForecast, CostOptimization, PricingRecommendation,
)


class FinancialModel:
    """Financial Analytics AI model."""

    def __init__(self):
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "financial"
        self.revenue_patterns = {}
        self.expense_patterns = {}

    def train(self, sales_df: pd.DataFrame, expense_df: pd.DataFrame) -> Dict:
        """Train financial analytics models."""
        print("  📊 Training Financial Analytics Model...")

        self.revenue_patterns = compute_revenue_features(sales_df)
        self.expense_patterns = compute_expense_features(expense_df)

        self.is_trained = True
        self._save_models()

        metrics = {
            "total_revenue_analyzed": self.revenue_patterns.get("total_revenue", 0),
            "total_expenses_analyzed": self.expense_patterns.get("total_expenses", 0),
            "collection_rate": self.revenue_patterns.get("collection_rate", 0),
            "top_expense_category": self.expense_patterns.get("top_expense", "Unknown"),
        }
        print(f"  ✅ Financial Model trained: revenue=₹{int(self.revenue_patterns.get('total_revenue', 0)):,}")
        return metrics

    def _save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.revenue_patterns, self.model_path / "revenue_patterns.joblib")
        joblib.dump(self.expense_patterns, self.model_path / "expense_patterns.joblib")

    def load_models(self):
        try:
            self.revenue_patterns = joblib.load(self.model_path / "revenue_patterns.joblib")
            self.expense_patterns = joblib.load(self.model_path / "expense_patterns.joblib")
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def get_profit_loss(self, sales_df: pd.DataFrame, expense_df: pd.DataFrame,
                        months: int = 3) -> Dict:
        """Generate P&L report."""
        revenue = compute_revenue_features(sales_df, months)
        expenses = compute_expense_features(expense_df)
        profitability = compute_profitability_features(sales_df, expense_df)

        # Expense breakdown
        expense_breakdown = []
        cat_totals = expenses.get("category_breakdown", {})
        total_exp = expenses.get("total_expenses", 1)
        for cat, amount in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
            expense_breakdown.append({
                "category": cat,
                "amount": round(amount, 0),
                "percentage": f"{amount / total_exp * 100:.1f}%",
                "trend": expenses.get("category_trends", {}).get(cat, "stable"),
            })

        # Insights
        insights = []
        margin = profitability["profit_margin"]
        target = config.TARGET_PROFIT_MARGIN

        if margin >= target:
            insights.append(f"✅ Profit margin {margin*100:.1f}% exceeds target {target*100:.0f}%")
        else:
            gap = (target - margin) * 100
            insights.append(f"⚠️ Profit margin {margin*100:.1f}% is {gap:.1f}% below target {target*100:.0f}%")

        if revenue.get("collection_rate", 1) < 0.8:
            outstanding = revenue.get("outstanding", 0)
            insights.append(f"⚠️ Collection rate {revenue['collection_rate']*100:.0f}% — ₹{int(outstanding):,} outstanding")

        if revenue.get("trend_direction") == "increasing":
            insights.append("📈 Revenue is trending upward month-over-month")
        elif revenue.get("trend_direction") == "decreasing":
            insights.append("📉 Revenue is declining — review sales strategy")

        # Labor cost ratio
        labor_cost = cat_totals.get("Labor - Regular", 0) + cat_totals.get("Labor - Overtime", 0)
        if labor_cost / max(1, total_exp) > 0.5:
            insights.append(f"💼 Labor costs are {labor_cost/total_exp*100:.0f}% of expenses — optimize workforce")

        return ProfitLossReport(
            period=f"Last {months} months",
            total_revenue=profitability["total_revenue"],
            total_expenses=profitability["total_expenses"],
            gross_profit=profitability["gross_profit"],
            profit_margin=round(margin * 100, 1),
            revenue_trend=revenue.get("trend_direction", "stable"),
            expense_breakdown=expense_breakdown,
            insights=insights,
            comparison_to_target={
                "target_margin": f"{target*100:.0f}%",
                "actual_margin": f"{margin*100:.1f}%",
                "status": "above_target" if margin >= target else "below_target",
            },
        ).model_dump()

    def forecast_cashflow(self, sales_df: pd.DataFrame, expense_df: pd.DataFrame,
                          months_ahead: int = 3) -> Dict:
        """Forecast cash flow for the next N months."""
        revenue = compute_revenue_features(sales_df)
        expenses = compute_expense_features(expense_df)

        monthly_revenue = revenue.get("monthly_avg_revenue", 0)
        monthly_expenses = expenses.get("monthly_avg", 0)
        revenue_trend = revenue.get("trend", 0)

        monthly_forecasts = []
        balance = revenue.get("total_collected", 0) - expenses.get("total_expenses", 0)
        balance = max(0, balance / 12)  # Approximate current monthly balance

        for i in range(1, months_ahead + 1):
            month_date = datetime.now() + timedelta(days=30 * i)
            # Apply trend
            projected_revenue = monthly_revenue + revenue_trend * i
            projected_expenses = monthly_expenses * (1 + 0.02 * i)  # 2% monthly cost increase

            net = projected_revenue - projected_expenses
            balance += net

            monthly_forecasts.append({
                "month": month_date.strftime("%B %Y"),
                "projected_revenue": round(projected_revenue, 0),
                "projected_expenses": round(projected_expenses, 0),
                "net_cashflow": round(net, 0),
                "running_balance": round(balance, 0),
            })

        # Status
        if balance > monthly_expenses * 2:
            status = "healthy"
        elif balance > monthly_expenses:
            status = "adequate"
        elif balance > 0:
            status = "tight"
        else:
            status = "critical"

        alerts = []
        if status in ["tight", "critical"]:
            alerts.append(f"⚠️ Cash flow projected to be {status} within {months_ahead} months")
            alerts.append("Consider accelerating collections on outstanding invoices")
        if any(m["net_cashflow"] < 0 for m in monthly_forecasts):
            neg_months = [m["month"] for m in monthly_forecasts if m["net_cashflow"] < 0]
            alerts.append(f"Negative cash flow expected in: {', '.join(neg_months)}")

        return CashFlowForecast(
            forecast_months=months_ahead,
            monthly_forecasts=monthly_forecasts,
            projected_balance=round(balance, 0),
            cash_flow_status=status,
            alerts=alerts,
        ).model_dump()

    def optimize_costs(self, expense_df: pd.DataFrame) -> Dict:
        """Identify cost optimization opportunities."""
        expenses = compute_expense_features(expense_df)
        cat_totals = expenses.get("category_breakdown", {})
        total = expenses.get("total_expenses", 1)

        opportunities = []

        # Overtime reduction
        overtime = cat_totals.get("Labor - Overtime", 0)
        if overtime > total * 0.05:
            savings = overtime * 0.3
            opportunities.append({
                "area": "Overtime Reduction",
                "current_cost": f"₹{int(overtime):,}",
                "potential_savings": f"₹{int(savings):,}",
                "action": "Optimize task scheduling, balance workload across workers",
                "difficulty": "Medium",
                "timeline": "2-4 weeks",
            })

        # Pesticide optimization
        pesticide = cat_totals.get("Pesticides & Fungicides", 0)
        if pesticide > 0:
            savings = pesticide * 0.2
            opportunities.append({
                "area": "Integrated Pest Management",
                "current_cost": f"₹{int(pesticide):,}",
                "potential_savings": f"₹{int(savings):,}",
                "action": "Implement IPM practices, reduce chemical dependency",
                "difficulty": "Medium",
                "timeline": "1-3 months",
            })

        # Energy optimization
        electricity = cat_totals.get("Electricity", 0)
        water = cat_totals.get("Water", 0)
        if electricity + water > 0:
            savings = (electricity + water) * 0.25
            opportunities.append({
                "area": "Utility Optimization",
                "current_cost": f"₹{int(electricity + water):,}",
                "potential_savings": f"₹{int(savings):,}",
                "action": "Smart irrigation scheduling, solar panel installation",
                "difficulty": "High",
                "timeline": "3-6 months",
            })

        # Bulk purchasing
        materials = sum(v for k, v in cat_totals.items()
                       if k in ["Fertilizers", "Seeds & Rootstock", "Packaging Materials"])
        if materials > 0:
            savings = materials * 0.10
            opportunities.append({
                "area": "Bulk Purchasing",
                "current_cost": f"₹{int(materials):,}",
                "potential_savings": f"₹{int(savings):,}",
                "action": "Negotiate bulk discounts with suppliers, join cooperative purchasing",
                "difficulty": "Low",
                "timeline": "1-2 weeks",
            })

        total_savings = sum(
            float(o["potential_savings"].replace("₹", "").replace(",", ""))
            for o in opportunities
        )

        return CostOptimization(
            total_current_cost=total,
            total_optimized_cost=total - total_savings,
            savings_potential=round(total_savings / max(1, total) * 100, 1),
            opportunities=opportunities,
        ).model_dump()

    def recommend_pricing(self, sales_df: pd.DataFrame, plant_df: pd.DataFrame) -> List[Dict]:
        """Generate dynamic pricing recommendations per variety."""
        df = sales_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        recommendations = []

        for variety in config.PLANT_VARIETIES:
            name = variety["name"]
            variety_sales = df[df["variety_name"] == name]
            variety_plants = plant_df[plant_df["variety_name"] == name]

            if variety_sales.empty:
                continue

            current_avg = variety_sales["unit_price"].mean()
            base_price = np.mean(variety["price_range"])

            # Demand-supply factor
            supply = len(variety_plants[variety_plants["stage"] == "ready_to_sell"])
            demand_rate = len(variety_sales) / max(1, (df["date"].max() - df["date"].min()).days) * 30

            if supply > demand_rate * 3:
                # Excess supply — lower price
                price_adj = -0.10
                reasoning = f"Excess supply ({supply} available vs {demand_rate:.0f}/month demand)"
            elif supply < demand_rate:
                # Short supply — raise price
                price_adj = 0.15
                reasoning = f"Strong demand ({demand_rate:.0f}/month) exceeds supply ({supply} available)"
            else:
                price_adj = 0
                reasoning = "Supply and demand are balanced"

            # Quality factor — if mostly A+/A grade, can charge premium
            if not variety_plants.empty:
                healthy_pct = len(variety_plants[variety_plants["health_status"] == "healthy"]) / len(variety_plants)
                if healthy_pct > 0.9:
                    price_adj += 0.05
                    reasoning += ", high quality stock (90%+ healthy)"

            recommended = current_avg * (1 + price_adj)
            recommended = np.clip(recommended, variety["price_range"][0], variety["price_range"][1] * 1.3)

            change_pct = (recommended - current_avg) / current_avg * 100

            impact = "No change expected"
            if change_pct > 5:
                impact = f"Increase revenue by ~₹{int(demand_rate * (recommended - current_avg)):,}/month"
            elif change_pct < -5:
                impact = f"Boost sales volume by 15-20% with price reduction"

            recommendations.append(PricingRecommendation(
                variety=name,
                current_avg_price=round(current_avg, 0),
                recommended_price=round(recommended, 0),
                change_pct=round(change_pct, 1),
                reasoning=reasoning,
                expected_impact=impact,
            ).model_dump())

        return recommendations
