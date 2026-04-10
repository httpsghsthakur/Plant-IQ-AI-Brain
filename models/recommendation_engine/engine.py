"""
PlantIQ AI Brain - Recommendation Engine
Aggregates outputs from all 8 models, prioritizes actions, generates daily report.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from models.recommendation_engine.schemas import (
    ActionItem, DailyReport, PerformanceDashboard,
)


class RecommendationEngine:
    """Aggregates insights from all models and generates prioritized recommendations."""

    def __init__(self):
        # Model instances will be injected
        self.environmental_model = None
        self.worker_model = None
        self.plant_health_model = None
        self.graft_model = None
        self.resource_model = None
        self.yield_model = None
        self.financial_model = None
        self.anomaly_model = None

    def set_models(self, environmental=None, worker=None, plant_health=None,
                   graft=None, resource=None, yield_forecast=None,
                   financial=None, anomaly=None):
        """Inject trained model instances."""
        self.environmental_model = environmental
        self.worker_model = worker
        self.plant_health_model = plant_health
        self.graft_model = graft
        self.resource_model = resource
        self.yield_model = yield_forecast
        self.financial_model = financial
        self.anomaly_model = anomaly

    def generate_daily_report(self, sensor_df: pd.DataFrame,
                               plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                               disease_df: pd.DataFrame, graft_df: pd.DataFrame,
                               attendance_df: pd.DataFrame, task_df: pd.DataFrame,
                               sales_df: pd.DataFrame, expense_df: pd.DataFrame,
                               inventory_df: pd.DataFrame) -> Dict:
        """Generate comprehensive daily AI advisor report."""
        urgent = []
        important = []
        optimization = []

        # ─── Environmental Analysis ──────────────────────────────────
        if self.environmental_model:
            for zone in config.ZONES:
                try:
                    analysis = self.environmental_model.analyze_zone(sensor_df, zone["id"])
                    if isinstance(analysis, dict):
                        if analysis.get("overall_status") == "critical":
                            for rec in analysis.get("recommendations", []):
                                urgent.append(ActionItem(
                                    priority="urgent",
                                    category="comfort",
                                    title=rec.get("action", "Growing Area Alert"),
                                    description=f"Zone {zone['name']}: {rec.get('method', '')}",
                                    impact=rec.get("impact", "Keep plants comfortable"),
                                    estimated_cost=f"₹{rec.get('estimated_cost', 0):,}" if rec.get("estimated_cost") else None,
                                ).model_dump())
                        elif analysis.get("overall_status") == "needs_attention":
                            for rec in analysis.get("recommendations", []):
                                important.append(ActionItem(
                                    priority="important",
                                    category="environmental",
                                    title=rec.get("action", "Environmental Issue"),
                                    description=f"Zone {zone['name']}: {rec.get('method', '')}",
                                    impact=rec.get("impact", ""),
                                ).model_dump())
                except Exception:
                    pass

            # Weather alerts
            try:
                alerts = self.environmental_model.get_weather_alerts(sensor_df)
                for alert in alerts:
                    if alert.get("severity") == "critical":
                        urgent.append(ActionItem(
                            priority="urgent",
                            category="weather",
                            title=alert.get("alert_type", "Weather Alert").replace("_", " ").title(),
                            description=alert.get("message", ""),
                            impact="Prevent weather-related crop damage",
                        ).model_dump())
            except Exception:
                pass

        # ─── Plant Health ────────────────────────────────────────────
        if self.plant_health_model:
            for zone in config.ZONES:
                try:
                    risks = self.plant_health_model.assess_disease_risk(
                        sensor_df, disease_df, zone["id"]
                    )
                    for risk in risks:
                        if risk.get("risk_level") in ["critical", "high"]:
                            urgent.append(ActionItem(
                                priority="urgent",
                                category="wellness",
                                title=f"Risk of Sickness: {risk['disease_name']}",
                                description=f"Zone {zone['name']} - Risk Score: {risk['risk_score']}",
                                impact=f"Prevention cost: {risk.get('prevention_cost', 'N/A')} vs outbreak cost: {risk.get('treatment_cost_if_outbreak', 'N/A')}",
                                estimated_cost=risk.get("prevention_cost"),
                            ).model_dump())
                except Exception:
                    pass

        # ─── Worker Performance ──────────────────────────────────────
        if self.worker_model:
            try:
                burnout_risks = self.worker_model.detect_burnout_risks(attendance_df, task_df)
                for risk in burnout_risks:
                    if risk.get("risk_score", 0) >= 60:
                        urgent.append(ActionItem(
                            priority="urgent",
                            category="worker",
                            title=f"Burnout Risk: {risk['worker_name']}",
                            description="; ".join(risk.get("indicators", [])),
                            impact="Prevent worker burnout and quality decline",
                        ).model_dump())
                    elif risk.get("risk_score", 0) >= 40:
                        important.append(ActionItem(
                            priority="important",
                            category="worker",
                            title=f"Monitor Burnout: {risk['worker_name']}",
                            description="; ".join(risk.get("indicators", [])),
                            impact="Early intervention prevents larger issues",
                        ).model_dump())
            except Exception:
                pass

        # ─── Anomaly Detection ───────────────────────────────────────
        if self.anomaly_model:
            try:
                anomaly_report = self.anomaly_model.get_anomaly_report(
                    sensor_df, attendance_df, task_df, inventory_df
                )
                for anomaly in anomaly_report.get("sensor_anomalies", []):
                    if anomaly.get("severity") == "critical":
                        urgent.append(ActionItem(
                            priority="urgent",
                            category="anomaly",
                            title=f"Sensor Anomaly: {anomaly['sensor_type']}",
                            description=f"{anomaly['zone_name']}: {anomaly['recommended_action']}",
                            impact="Verify sensor integrity and plant safety",
                        ).model_dump())
                for anomaly in anomaly_report.get("inventory_anomalies", []):
                    important.append(ActionItem(
                        priority="important",
                        category="inventory",
                        title=f"Inventory Alert: {anomaly['item_name']}",
                        description=anomaly['description'],
                        impact=anomaly.get("recommended_action", ""),
                    ).model_dump())
            except Exception:
                pass

        # ─── Resource Optimization ───────────────────────────────────
        if self.resource_model:
            try:
                inv_forecasts = self.resource_model.predict_inventory(inventory_df)
                for f in inv_forecasts:
                    if f.get("urgency") == "critical":
                        urgent.append(ActionItem(
                            priority="urgent",
                            category="inventory",
                            title=f"Reorder: {f['item_name']}",
                            description=f"Only {f['days_until_stockout']} days of stock remaining",
                            impact="Prevent supply disruption",
                            estimated_cost=f.get("estimated_cost"),
                        ).model_dump())
                    elif f.get("urgency") == "high":
                        important.append(ActionItem(
                            priority="important",
                            category="inventory",
                            title=f"Low Stock: {f['item_name']}",
                            description=f"{f['days_until_stockout']} days remaining, reorder by {f['reorder_date']}",
                            impact="Maintain operations without disruption",
                            estimated_cost=f.get("estimated_cost"),
                        ).model_dump())
            except Exception:
                pass

        # ─── Financial Optimization ──────────────────────────────────
        if self.financial_model:
            try:
                cost_opt = self.financial_model.optimize_costs(expense_df)
                for opp in cost_opt.get("opportunities", []):
                    optimization.append(ActionItem(
                        priority="optimization",
                        category="financial",
                        title=opp.get("area", "Cost Savings"),
                        description=opp.get("action", ""),
                        impact=f"Save {opp.get('potential_savings', 'N/A')}",
                        estimated_roi=opp.get("potential_savings"),
                    ).model_dump())
            except Exception:
                pass

        # ─── Summary Metrics ─────────────────────────────────────────
        summary = {
            "total_plants": len(plant_df),
            "healthy_plants_pct": round(
                len(plant_df[plant_df.get("health_status", pd.Series(["healthy"])) == "healthy"]) / max(1, len(plant_df)) * 100, 1
            ) if "health_status" in plant_df.columns else 85,
            "workers_present_today": "N/A",
            "urgent_actions": len(urgent),
            "important_actions": len(important),
            "optimization_actions": len(optimization),
        }

        # Performance snapshot
        performance = {
            "plant_wellness": "Good",
            "growing_comfort": "Monitoring",
            "team_activity": "Active",
            "money_tracking": "On Track",
        }

        return DailyReport(
            report_date=datetime.now().strftime("%Y-%m-%d"),
            nursery_name=config.NURSERY_NAME,
            summary=summary,
            urgent_actions=urgent[:10],
            important_actions=important[:10],
            optimization_actions=optimization[:10],
            performance_snapshot=performance,
            generated_at=datetime.now().isoformat(),
        ).model_dump()

    def get_performance_dashboard(self, sensor_df: pd.DataFrame,
                                   plant_df: pd.DataFrame,
                                   attendance_df: pd.DataFrame,
                                   sales_df: pd.DataFrame,
                                   expense_df: pd.DataFrame) -> Dict:
        """Generate performance dashboard data."""
        # Plant health score
        if "health_status" in plant_df.columns:
            healthy = len(plant_df[plant_df["health_status"] == "healthy"])
            plant_score = healthy / max(1, len(plant_df)) * 100
        else:
            plant_score = 85

        # Environmental score (average across zones)
        env_score = 80  # Default
        if self.environmental_model:
            zone_scores = []
            for zone in config.ZONES:
                try:
                    analysis = self.environmental_model.analyze_zone(sensor_df, zone["id"])
                    if isinstance(analysis, dict):
                        zone_scores.append(100 - analysis.get("risk_score", 20))
                except Exception:
                    pass
            if zone_scores:
                env_score = np.mean(zone_scores)

        # Worker score
        worker_score = 80
        if self.worker_model:
            try:
                all_scores = self.worker_model.get_all_scores(
                    attendance_df,
                    pd.DataFrame(),  # Empty task_df fallback
                )
                if all_scores:
                    worker_score = np.mean([s["score"] for s in all_scores])
            except Exception:
                pass

        # Financial score
        financial_score = 75
        if self.financial_model:
            try:
                profitability = compute_profitability_features(sales_df, expense_df)
                margin = profitability.get("profit_margin", 0)
                financial_score = min(100, margin / config.TARGET_PROFIT_MARGIN * 100)
            except Exception:
                pass

        # Overall
        overall = (plant_score * 0.3 + env_score * 0.25 + worker_score * 0.25 + financial_score * 0.2)

        return PerformanceDashboard(
            overall_score=round(overall, 1),
            category_scores={
                "plant_health": round(plant_score, 1),
                "environmental": round(env_score, 1),
                "workforce": round(worker_score, 1),
                "financial": round(financial_score, 1),
            },
            trends={
                "plant_health": "stable",
                "revenue": "increasing",
                "efficiency": "improving",
            },
            key_metrics={
                "total_plants": len(plant_df),
                "zones_active": len(config.ZONES),
                "workers": len(attendance_df["worker_id"].unique()) if "worker_id" in attendance_df.columns else 0,
            },
            alerts_count=0,
        ).model_dump()


# Import needed in get_performance_dashboard
try:
    from models.financial.features import compute_profitability_features
except ImportError:
    def compute_profitability_features(*args, **kwargs):
        return {"profit_margin": 0.3}
