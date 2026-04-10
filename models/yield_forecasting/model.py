"""
PlantIQ AI Brain - Yield Forecasting Model
Production forecast, demand-supply matching, quality grade prediction.
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
from models.yield_forecasting.features import (
    compute_production_pipeline,
    compute_demand_features,
    compute_quality_features,
)
from models.yield_forecasting.schemas import (
    ProductionForecast, DemandSupplyAnalysis, QualityDistribution,
)


class YieldForecastingModel:
    """Yield Forecasting AI model."""

    def __init__(self):
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "yield_forecasting"
        self.demand_patterns = {}

    def train(self, plant_df: pd.DataFrame, sales_df: pd.DataFrame) -> Dict:
        """Train yield forecasting models."""
        print("  [*] Training Yield Forecasting Model...")

        # Analyze production pipeline
        pipeline = compute_production_pipeline(plant_df)
        demand = compute_demand_features(sales_df)
        quality = compute_quality_features(plant_df)

        self.demand_patterns = demand
        self.is_trained = True
        self._save_models()

        metrics = {
            "plants_in_pipeline": pipeline["total_ready"],
            "currently_ready": pipeline["currently_ready"],
            "monthly_demand_avg": demand.get("monthly_avg_quantity", 0),
            "grade_distribution": quality.get("grade_distribution", {}),
        }
        print(f"  [OK] Yield Forecasting Model trained: {pipeline['total_ready']} plants in pipeline")
        return metrics

    def _save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.demand_patterns, self.model_path / "demand_patterns.joblib")

    def load_models(self):
        try:
            self.demand_patterns = joblib.load(self.model_path / "demand_patterns.joblib")
            self.is_trained = True
            return True
        except FileNotFoundError:
            self.demand_patterns = {}
            return False

    def forecast_production(self, plant_df: pd.DataFrame,
                             days_ahead: int = 90) -> Dict:
        """Forecast production for the next N days."""
        pipeline = compute_production_pipeline(plant_df, days_ahead)

        # By variety details
        by_variety = []
        for v in pipeline.get("by_variety", []):
            variety_info = next(
                (pv for pv in config.PLANT_VARIETIES if pv["name"] == v.get("variety_name")),
                {"price_range": (2000, 3000)}
            )
            avg_price = np.mean(variety_info["price_range"])
            count = int(v.get("count", 0))
            by_variety.append({
                "variety": v.get("variety_name", "Unknown"),
                "count": count,
                "avg_height_cm": round(v.get("avg_height", 0), 1),
                "estimated_revenue": f"₹{int(count * avg_price):,}",
            })

        # Grade distribution
        quality = compute_quality_features(plant_df)
        grade_dist = quality.get("grade_distribution", {})

        # Revenue forecast
        total_ready = pipeline["total_ready"]
        avg_price = config.AVG_PLANT_PRICE
        grade_multipliers = {"A+": 1.3, "A": 1.0, "B": 0.75, "C": 0.5}

        revenue_by_grade = {}
        for grade, count in grade_dist.items():
            mult = grade_multipliers.get(grade, 1.0)
            revenue_by_grade[grade] = round(count * avg_price * mult, 0)

        # Confidence interval (+/-15%)
        low = int(total_ready * 0.85)
        high = int(total_ready * 1.15)

        return ProductionForecast(
            forecast_period=f"Next {days_ahead} days",
            total_plants_ready=total_ready,
            by_variety=by_variety,
            by_grade=grade_dist,
            confidence_interval={"low": low, "mid": total_ready, "high": high},
            revenue_forecast={
                "pessimistic": round(low * avg_price * 0.8, 0),
                "expected": round(total_ready * avg_price, 0),
                "optimistic": round(high * avg_price * 1.1, 0),
            },
        ).model_dump()

    def analyze_demand_supply(self, plant_df: pd.DataFrame,
                               sales_df: pd.DataFrame, months: int = 3) -> Dict:
        """Analyze demand-supply gap for the forecast period."""
        pipeline = compute_production_pipeline(plant_df, days_ahead=months * 30)
        demand = compute_demand_features(sales_df, lookback_days=months * 30)

        supply = pipeline["total_ready"]
        monthly_demand = demand.get("monthly_avg_quantity", 0)
        total_demand = int(monthly_demand * months)

        gap = supply - total_demand
        if gap > total_demand * 0.2:
            gap_status = "surplus"
        elif gap < -total_demand * 0.1:
            gap_status = "deficit"
        else:
            gap_status = "balanced"

        recommendations = []
        if gap_status == "surplus":
            recommendations.append(f"Consider promotions to move {gap} excess plants")
            recommendations.append("Explore new sales channels or bulk orders")
            recommendations.append("Reduce next planting batch by 15-20%")
        elif gap_status == "deficit":
            recommendations.append(f"Increase production capacity -- {abs(gap)} plant shortage expected")
            recommendations.append("Prioritize high-value varieties for available stock")
            recommendations.append("Start advance bookings to manage demand")
        else:
            recommendations.append("Supply and demand are well-balanced")
            recommendations.append("Maintain current production rate")

        # By variety analysis
        variety_demand = demand.get("variety_demand", {})
        by_variety = []
        for v_info in pipeline.get("by_variety", []):
            variety = v_info.get("variety_name", "Unknown")
            v_supply = int(v_info.get("count", 0))
            v_demand = int(variety_demand.get(variety, 0) * months / max(1, months))
            by_variety.append({
                "variety": variety,
                "supply": v_supply,
                "demand": v_demand,
                "gap": v_supply - v_demand,
                "status": "surplus" if v_supply > v_demand * 1.2 else "deficit" if v_supply < v_demand * 0.8 else "balanced",
            })

        return DemandSupplyAnalysis(
            period=f"Next {months} months",
            supply_forecast=supply,
            estimated_demand=total_demand,
            gap=gap,
            gap_status=gap_status,
            recommendations=recommendations,
            by_variety=by_variety,
        ).model_dump()

    def predict_quality(self, plant_df: pd.DataFrame) -> Dict:
        """Predict quality grade distribution."""
        quality = compute_quality_features(plant_df)

        improvement_potential = []
        grade_dist = quality.get("grade_distribution", {})
        total = quality.get("total_plants", 1)

        b_count = grade_dist.get("B", 0)
        c_count = grade_dist.get("C", 0)

        if b_count > 0:
            improvement_potential.append({
                "from_grade": "B",
                "to_grade": "A",
                "plant_count": b_count,
                "potential_revenue_increase": f"₹{int(b_count * config.AVG_PLANT_PRICE * 0.25):,}",
                "action": "Optimize growing conditions and fertilization schedule",
            })
        if c_count > 0:
            improvement_potential.append({
                "from_grade": "C",
                "to_grade": "B",
                "plant_count": c_count,
                "potential_revenue_increase": f"₹{int(c_count * config.AVG_PLANT_PRICE * 0.25):,}",
                "action": "Investigate root cause -- disease treatment or environmental fix",
            })

        return QualityDistribution(
            total_plants=total,
            distribution={
                grade: {
                    "count": count,
                    "percentage": f"{count/total*100:.1f}%",
                    "avg_revenue": f"₹{int(config.AVG_PLANT_PRICE * {'A+': 1.3, 'A': 1.0, 'B': 0.75, 'C': 0.5}.get(grade, 1)):,}",
                }
                for grade, count in grade_dist.items()
            },
            improvement_potential=improvement_potential,
        ).model_dump()
