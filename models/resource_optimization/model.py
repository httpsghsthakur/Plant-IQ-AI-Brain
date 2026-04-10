"""
PlantIQ AI Brain - Resource Optimization Model
Water optimization, fertilizer planning, inventory prediction, and energy optimization.
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
from models.resource_optimization.features import (
    compute_consumption_features,
    compute_water_usage_features,
    compute_fertilizer_needs,
)
from models.resource_optimization.schemas import (
    WaterOptimization, FertilizerPlan, InventoryForecast, ResourceSummary,
)


class ResourceOptimizationModel:
    """Resource Optimization AI model."""

    def __init__(self):
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "resource_optimization"

    def train(self, inventory_df: pd.DataFrame, sensor_df: pd.DataFrame) -> Dict:
        """Train resource optimization models (primarily rule/pattern-based)."""
        print("  📊 Training Resource Optimization Model...")

        # This model is primarily rule-based with learned consumption patterns
        item_patterns = {}
        items = inventory_df["item_name"].unique()
        for item in items:
            feats = compute_consumption_features(inventory_df, item, lookback_days=90)
            item_patterns[item] = feats

        self.item_patterns = item_patterns
        self.is_trained = True
        self._save_models()

        metrics = {
            "items_analyzed": len(items),
            "items_below_reorder": sum(1 for v in item_patterns.values() if v.get("below_reorder")),
            "avg_daily_consumption_total": round(sum(v["daily_avg"] for v in item_patterns.values()), 2),
        }
        print(f"  ✅ Resource Optimization Model trained: {len(items)} items analyzed")
        return metrics

    def _save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.item_patterns, self.model_path / "item_patterns.joblib")

    def load_models(self):
        try:
            self.item_patterns = joblib.load(self.model_path / "item_patterns.joblib")
            self.is_trained = True
            return True
        except FileNotFoundError:
            self.item_patterns = {}
            return False

    def optimize_water(self, sensor_df: pd.DataFrame, inventory_df: pd.DataFrame) -> List[Dict]:
        """Generate optimized water usage recommendations per zone."""
        results = []
        total_current = 0
        total_optimized = 0

        for zone in config.ZONES:
            zone_id = zone["id"]
            feats = compute_water_usage_features(sensor_df, zone_id)

            # Current estimated daily water usage (liters)
            current_daily = zone["capacity"] * 0.005 * 20  # Fixed 20min daily
            current_monthly = current_daily * 30

            # Optimized usage based on actual conditions
            if feats["avg_moisture"] > 55:
                # Reduce irrigation
                optimized_daily = current_daily * 0.5
                schedule_desc = "Irrigate every other day, 10 minutes"
            elif feats["avg_moisture"] > 45:
                # Normal
                optimized_daily = current_daily * 0.7
                schedule_desc = "Irrigate daily, 14 minutes"
            elif feats["avg_moisture"] > 35:
                # Increase slightly
                optimized_daily = current_daily * 0.85
                schedule_desc = "Irrigate daily, 17 minutes"
            else:
                # Critical - don't reduce
                optimized_daily = current_daily
                schedule_desc = "Irrigate daily, 20 minutes + supplementary"

            # Rain adjustment
            if feats["rainfall_total"] > 10:
                optimized_daily *= 0.6
                schedule_desc += " (reduced for recent rainfall)"

            optimized_monthly = optimized_daily * 30
            savings = max(0, (current_monthly - optimized_monthly) / current_monthly * 100)

            total_current += current_monthly
            total_optimized += optimized_monthly

            # Schedule items
            schedule = []
            if optimized_daily > 0:
                now = datetime.now()
                # Morning irrigation
                schedule.append({
                    "time": "06:00",
                    "duration_minutes": int(optimized_daily / (zone["capacity"] * 0.003)),
                    "method": "Drip irrigation",
                })
                # Evening if needed
                if feats["avg_temp"] > 30:
                    schedule.append({
                        "time": "17:00",
                        "duration_minutes": int(optimized_daily * 0.3 / (zone["capacity"] * 0.003)),
                        "method": "Misting",
                    })

            recommendations = []
            if savings > 20:
                recommendations.append(f"Reduce watering by {savings:.0f}% - soil is adequately moist")
            if feats["rainfall_total"] > 20:
                recommendations.append("Skip irrigation today due to recent rainfall")
            if feats["evaporation_estimate"] > 1:
                recommendations.append("High evaporation rate - consider mulching to retain moisture")
            if not recommendations:
                recommendations.append("Current watering schedule is near optimal")

            results.append(WaterOptimization(
                zone_id=zone_id,
                zone_name=zone["name"],
                current_usage_liters=round(current_monthly, 0),
                optimized_usage_liters=round(optimized_monthly, 0),
                savings_pct=round(savings, 1),
                schedule=schedule,
                recommendations=recommendations,
            ).model_dump())

        return results

    def get_fertilizer_plan(self, plant_df: pd.DataFrame) -> List[Dict]:
        """Generate fertilizer plan for all zones."""
        plans = []
        for zone in config.ZONES:
            zone_id = zone["id"]
            needs = compute_fertilizer_needs(plant_df, zone_id)

            for need in needs:
                # Estimate cost
                cost_per_kg = {"NPK 20-20-20": 55, "NPK 15-15-15": 45, "NPK 10-26-26": 50,
                               "NPK 10-10-10 + Micronutrients": 60, "NPK 10-10-10": 45}
                dose_grams = int(need["dosage_per_plant"].replace("g/plant", ""))
                total_kg = dose_grams * need["plant_count"] / 1000
                unit_cost = cost_per_kg.get(need["fertilizer_type"], 50)
                est_cost = total_kg * unit_cost

                plans.append(FertilizerPlan(
                    zone_id=zone_id,
                    variety=need["variety"],
                    stage=need["stage"],
                    fertilizer_type=need["fertilizer_type"],
                    dosage=f"{need['dosage_per_plant']} × {need['plant_count']} plants = {total_kg:.1f} kg",
                    frequency=need["frequency"],
                    estimated_cost=f"₹{int(est_cost):,}",
                    expected_impact=f"Maintain healthy growth for {need['plant_count']} {need['stage']} plants",
                ).model_dump())

        return plans

    def predict_inventory(self, inventory_df: pd.DataFrame) -> List[Dict]:
        """Predict inventory levels and generate reorder alerts."""
        items = inventory_df["item_name"].unique()
        forecasts = []

        for item in items:
            feats = compute_consumption_features(inventory_df, item)
            if feats["daily_avg"] == 0:
                continue

            # Days until stockout
            if feats["daily_avg"] > 0:
                days_left = int(feats["current_stock"] / feats["daily_avg"])
            else:
                days_left = 999

            # Reorder point (standard + safety stock)
            lead_time_days = 5
            safety_stock_days = 3
            reorder_point = feats["daily_avg"] * (lead_time_days + safety_stock_days)
            reorder_qty = feats["daily_avg"] * 30  # 30-day supply

            # Reorder date
            reorder_date = datetime.now() + timedelta(days=max(0, days_left - lead_time_days))

            # Urgency
            if days_left <= 3:
                urgency = "critical"
            elif days_left <= 7:
                urgency = "high"
            elif days_left <= 14:
                urgency = "medium"
            else:
                urgency = "low"

            # Get unit cost from inventory data
            item_data = inventory_df[inventory_df["item_name"] == item]
            unit_cost = item_data["unit_cost"].iloc[0] if "unit_cost" in item_data.columns else 50
            est_cost = reorder_qty * unit_cost

            forecasts.append(InventoryForecast(
                item_name=item,
                current_stock=feats["current_stock"],
                daily_usage_rate=feats["daily_avg"],
                days_until_stockout=days_left,
                reorder_date=reorder_date.strftime("%Y-%m-%d"),
                reorder_quantity=round(reorder_qty, 1),
                estimated_cost=f"₹{int(est_cost):,}",
                urgency=urgency,
            ).model_dump())

        return sorted(forecasts, key=lambda x: x["days_until_stockout"])

    def get_resource_summary(self, inventory_df: pd.DataFrame, sensor_df: pd.DataFrame,
                              plant_df: pd.DataFrame) -> Dict:
        """Get overall resource optimization summary."""
        # Water optimization summary
        water_opts = self.optimize_water(sensor_df, inventory_df)
        total_water_current = sum(w["current_usage_liters"] for w in water_opts)
        total_water_optimized = sum(w["optimized_usage_liters"] for w in water_opts)

        # Inventory alerts
        inv_forecasts = self.predict_inventory(inventory_df)
        critical_items = [f for f in inv_forecasts if f["urgency"] in ["critical", "high"]]

        # Cost estimates
        water_cost_savings = (total_water_current - total_water_optimized) * 0.05  # ₹0.05/liter
        monthly_savings = water_cost_savings

        # Top savings opportunities
        top_savings = []
        if total_water_current > total_water_optimized:
            top_savings.append({
                "area": "Water Usage",
                "current_cost": f"₹{int(total_water_current * 0.05):,}/month",
                "optimized_cost": f"₹{int(total_water_optimized * 0.05):,}/month",
                "savings": f"₹{int(water_cost_savings):,}/month",
                "action": "Implement smart irrigation schedule",
            })

        return ResourceSummary(
            total_monthly_cost=round(total_water_current * 0.05 + 200000, 0),  # Water + other ops
            optimized_monthly_cost=round(total_water_optimized * 0.05 + 180000, 0),
            savings_pct=round(monthly_savings / max(1, total_water_current * 0.05) * 100, 1),
            top_savings_opportunities=top_savings,
            inventory_alerts=[{
                "item": f["item_name"],
                "days_left": f["days_until_stockout"],
                "urgency": f["urgency"],
            } for f in critical_items[:5]],
        ).model_dump()
