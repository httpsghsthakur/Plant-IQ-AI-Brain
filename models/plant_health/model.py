"""
PlantIQ AI Brain - Plant Health Prediction Model
Growth prediction, disease risk scoring, stress detection, and mortality prediction.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from models.plant_health.features import (
    compute_growth_features,
    compute_disease_risk_features,
    compute_mortality_features,
    prepare_training_data,
)
from models.plant_health.schemas import (
    GrowthPrediction, DiseaseRisk, StressDetection, MortalityPrediction,
)


class PlantHealthModel:
    """Plant Health Prediction AI model."""

    def __init__(self):
        self.growth_predictor = None
        self.mortality_predictor = None
        self.scaler = StandardScaler()
        self.variety_encoder = LabelEncoder()
        self.stage_encoder = LabelEncoder()
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "plant_health"

    def train(self, plant_df: pd.DataFrame, growth_df: pd.DataFrame,
              disease_df: pd.DataFrame) -> Dict:
        """Train the plant health models."""
        print("  📊 Training Plant Health Prediction Model...")

        # Prepare training data
        train_data = prepare_training_data(plant_df, growth_df, disease_df)
        if len(train_data) < 20:
            print("  ⚠️  Insufficient data for plant health model")
            self.is_trained = True
            return {"status": "rule_based", "reason": "insufficient_data"}

        # Encode categorical features
        train_data["variety_encoded"] = self.variety_encoder.fit_transform(
            train_data["variety"].fillna("Unknown")
        )
        train_data["stage_encoded"] = self.stage_encoder.fit_transform(
            train_data["stage"].fillna("seedling")
        )

        # Feature columns for training
        numeric_cols = [
            "age_days", "height_deviation_pct", "growth_trend",
            "growth_acceleration", "avg_weekly_growth", "recent_weekly_growth",
            "growth_consistency", "health_score_latest", "health_score_trend",
            "stress_count", "disease_detected_count", "total_diseases",
            "severe_diseases", "untreated_diseases", "treatment_failures",
            "health_code", "is_grafted", "graft_factor",
            "variety_encoded", "stage_encoded",
        ]

        # Filter available columns
        available_cols = [c for c in numeric_cols if c in train_data.columns]
        X = train_data[available_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        # Train growth deviation predictor
        y_growth = train_data["height_deviation_pct"].fillna(0).values
        self.growth_predictor = GradientBoostingRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        )
        self.growth_predictor.fit(X_scaled, y_growth)
        growth_r2 = self.growth_predictor.score(X_scaled, y_growth)

        # Train mortality predictor (survival probability)
        y_survival = train_data["survival_probability"].values
        self.mortality_predictor = GradientBoostingRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=6,
            random_state=config.RANDOM_STATE,
        )
        self.mortality_predictor.fit(X_scaled, y_survival)
        mortality_r2 = self.mortality_predictor.score(X_scaled, y_survival)

        self.is_trained = True
        self._save_models()

        metrics = {
            "growth_predictor_r2": round(growth_r2, 4),
            "mortality_predictor_r2": round(mortality_r2, 4),
            "training_samples": len(X),
            "features_used": len(available_cols),
        }
        print(f"  ✅ Plant Health Model trained: growth R²={growth_r2:.3f}, mortality R²={mortality_r2:.3f}")
        return metrics

    def _save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        if self.growth_predictor:
            joblib.dump(self.growth_predictor, self.model_path / "growth_predictor.joblib")
        if self.mortality_predictor:
            joblib.dump(self.mortality_predictor, self.model_path / "mortality_predictor.joblib")
        joblib.dump(self.scaler, self.model_path / "scaler.joblib")
        joblib.dump(self.variety_encoder, self.model_path / "variety_encoder.joblib")
        joblib.dump(self.stage_encoder, self.model_path / "stage_encoder.joblib")

    def load_models(self):
        try:
            self.growth_predictor = joblib.load(self.model_path / "growth_predictor.joblib")
            self.mortality_predictor = joblib.load(self.model_path / "mortality_predictor.joblib")
            self.scaler = joblib.load(self.model_path / "scaler.joblib")
            self.variety_encoder = joblib.load(self.model_path / "variety_encoder.joblib")
            self.stage_encoder = joblib.load(self.model_path / "stage_encoder.joblib")
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def predict_growth(self, plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                       plant_id: str) -> Dict:
        """Predict growth status and deviations for a plant."""
        feats = compute_growth_features(plant_df, growth_df, plant_id)
        if not feats:
            return {"error": f"No data for plant {plant_id}"}

        plant = plant_df[plant_df["plant_id"] == plant_id].iloc[0]

        # Determine growth status
        deviation = feats["height_deviation_pct"]
        if deviation > 10:
            growth_status = "above_average"
        elif deviation > -10:
            growth_status = "normal"
        elif deviation > -25:
            growth_status = "below_average"
        else:
            growth_status = "stunted"

        # Determine likely causes for deviation
        causes = []
        recommendations = []

        if deviation < -15:
            if feats["stress_count"] > 3:
                causes.append("Repeated environmental stress")
                recommendations.append("Move to protected zone or improve shelter")
            if feats["disease_detected_count"] > 0:
                causes.append("Disease impact on growth")
                recommendations.append("Apply preventive fungicide treatment")
            if feats["growth_trend"] < -0.1:
                causes.append("Declining growth rate over time")
                recommendations.append("Check soil nutrients, consider foliar feed")
            if not causes:
                causes.append("Genetic variation or suboptimal conditions")
                recommendations.append("Review zone conditions and fertilization schedule")
        elif deviation > 15:
            causes.append("Excellent growing conditions")
            recommendations.append("Maintain current care regimen")
            if feats["health_score_latest"] > 90:
                recommendations.append("Consider as mother plant for grafting")

        if not recommendations:
            recommendations.append("Continue standard care protocol")

        recovery_time = None
        if growth_status in ["below_average", "stunted"]:
            weeks_needed = int(abs(deviation) / max(0.1, feats["avg_weekly_growth"]))
            recovery_time = f"{min(52, weeks_needed)} weeks with intervention"

        return GrowthPrediction(
            plant_id=plant_id,
            variety=plant.get("variety_name", "Unknown"),
            age_days=feats["age_days"],
            current_height=feats["current_height"],
            expected_height=feats["expected_height"],
            growth_status=growth_status,
            deviation_pct=round(deviation, 1),
            likely_causes=causes,
            recommendations=recommendations,
            predicted_recovery_time=recovery_time,
        ).model_dump()

    def assess_disease_risk(self, sensor_df: pd.DataFrame, disease_df: pd.DataFrame,
                             zone_id: str) -> List[Dict]:
        """Assess disease risk for a specific zone."""
        feats = compute_disease_risk_features(sensor_df, disease_df, zone_id)
        risks = []

        for disease in config.DISEASES:
            risk_score = 0
            factors = []

            # Temperature risk
            temp_min, temp_max = disease["risk_temp"]
            if temp_min <= feats["avg_temp"] <= temp_max:
                risk_score += 25
                factors.append(f"Temperature {feats['avg_temp']}°C in risk range ({temp_min}-{temp_max}°C)")

            # Humidity risk
            if feats["avg_humidity"] >= disease["risk_humidity"]:
                risk_score += 30
                factors.append(f"Humidity {feats['avg_humidity']}% exceeds risk threshold ({disease['risk_humidity']}%)")

            # High humidity duration
            if feats["high_humidity_hours"] > 48:
                risk_score += 15
                factors.append(f"{feats['high_humidity_hours']} hours of high humidity in last 30 days")

            # Rainfall
            if feats["rainfall_total"] > 100:
                risk_score += 10
                factors.append(f"Heavy rainfall: {feats['rainfall_total']:.0f}mm in 30 days")

            # Historical disease in zone
            if feats["disease_history_count"] > 5:
                risk_score += 15
                factors.append(f"Zone has history of {feats['disease_history_count']} disease cases")

            # Recurrence
            if feats["disease_recurrence_rate"] > 0.3:
                risk_score += 10
                factors.append(f"Treatment recurrence rate: {feats['disease_recurrence_rate']*100:.0f}%")

            risk_score = min(100, risk_score)

            if risk_score >= 20:
                risk_level = "critical" if risk_score >= 70 else "high" if risk_score >= 50 else "medium" if risk_score >= 30 else "low"
                actions = []
                if risk_score >= 50:
                    actions.append(f"Apply preventive {disease['name']} treatment immediately")
                    actions.append("Increase ventilation to reduce humidity")
                if risk_score >= 30:
                    actions.append("Monitor plants for early symptoms daily")
                    actions.append("Ensure proper drainage and air circulation")
                actions.append("Maintain optimal growing conditions")

                # Cost estimation
                prevention_cost = f"₹{int(risk_score * 15):,}"
                outbreak_cost = f"₹{int(risk_score * 150):,}"

                risks.append(DiseaseRisk(
                    zone_id=zone_id,
                    disease_name=disease["name"],
                    risk_score=risk_score,
                    risk_level=risk_level,
                    contributing_factors=factors,
                    recommended_actions=actions,
                    prevention_cost=prevention_cost,
                    treatment_cost_if_outbreak=outbreak_cost,
                ).model_dump())

        return sorted(risks, key=lambda x: x["risk_score"], reverse=True)

    def detect_stress(self, plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                      sensor_df: pd.DataFrame, plant_id: str) -> Dict:
        """Detect stress conditions for a specific plant."""
        plant_row = plant_df[plant_df["plant_id"] == plant_id]
        if plant_row.empty:
            return {"error": f"Plant {plant_id} not found"}
        plant = plant_row.iloc[0]

        feats = compute_growth_features(plant_df, growth_df, plant_id)
        zone_id = plant.get("zone_id", "")

        # Get current environmental data for plant's zone
        zone_data = sensor_df[sensor_df["zone_id"] == zone_id].copy()
        if not zone_data.empty:
            zone_data["timestamp"] = pd.to_datetime(zone_data["timestamp"])
            latest = zone_data.sort_values("timestamp").iloc[-1]
            current_temp = float(latest["temperature"])
            current_moisture = float(latest["soil_moisture"])
            current_humidity = float(latest["humidity"])
        else:
            current_temp = 23
            current_moisture = 50
            current_humidity = 65

        # Determine stress type and level
        stress_type = "none"
        stress_level = "none"
        indicators = []
        immediate_action = "No immediate action needed"
        long_term_fix = "Continue standard care"

        # Water stress
        if current_moisture < 30:
            stress_type = "water_stress"
            stress_level = "severe" if current_moisture < 20 else "moderate"
            indicators.append(f"Soil moisture critically low: {current_moisture}%")
            immediate_action = "Irrigate immediately for 30 minutes"
            long_term_fix = "Install drip irrigation, add mulch layer"
        elif current_moisture > 80:
            stress_type = "waterlog_stress"
            stress_level = "moderate"
            indicators.append(f"Soil moisture too high: {current_moisture}%")
            immediate_action = "Check drainage, stop irrigation"
            long_term_fix = "Improve drainage system, raised bed planting"

        # Heat stress
        if current_temp > 35:
            stress_type = "heat_stress"
            stress_level = "severe" if current_temp > 40 else "moderate"
            indicators.append(f"Temperature dangerously high: {current_temp}°C")
            immediate_action = "Deploy shade nets, activate misting"
            long_term_fix = "Permanent shade structure, heat-tolerant rootstock"
        elif current_temp < 5:
            stress_type = "cold_stress"
            stress_level = "severe" if current_temp < 0 else "moderate"
            indicators.append(f"Temperature critically low: {current_temp}°C")
            immediate_action = "Cover with frost blankets, activate heaters"
            long_term_fix = "Winter protection infrastructure"

        # Nutrient stress (from growth deviation)
        if feats.get("height_deviation_pct", 0) < -25 and stress_type == "none":
            stress_type = "nutrient_deficiency"
            stress_level = "moderate"
            indicators.append(f"Growth {feats['height_deviation_pct']:.0f}% below expected")
            immediate_action = "Apply balanced NPK fertilizer"
            long_term_fix = "Soil test and custom fertilization plan"

        # Growth decline indicator
        if feats.get("growth_trend", 0) < -0.2:
            indicators.append("Growth rate declining over time")
        if feats.get("health_score_trend", 0) < -5:
            indicators.append("Health score declining")
        if feats.get("stress_count", 0) > 5:
            indicators.append(f"Plant has experienced {feats['stress_count']} stress events")

        if not indicators:
            indicators.append("No stress indicators detected")

        return StressDetection(
            plant_id=plant_id,
            stress_level=stress_level,
            stress_type=stress_type,
            indicators=indicators,
            immediate_action=immediate_action,
            long_term_fix=long_term_fix,
        ).model_dump()

    def predict_mortality(self, plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                          disease_df: pd.DataFrame, plant_id: str) -> Dict:
        """Predict survival probability for a specific plant."""
        feats = compute_mortality_features(plant_df, growth_df, disease_df, plant_id)
        if not feats:
            return {"error": f"Plant {plant_id} not found"}

        # Rule-based survival estimation (works without trained model too)
        survival = 1.0
        reasons = []
        recommendations = []

        if feats["health_code"] == 2:
            survival -= 0.30
            reasons.append("Currently in critical health status")
            recommendations.append("Isolate plant and apply emergency treatment")
        elif feats["health_code"] == 1:
            survival -= 0.10
            reasons.append("Plant is at risk")
            recommendations.append("Increase monitoring frequency")

        if feats["severe_diseases"] > 0:
            survival -= 0.15 * feats["severe_diseases"]
            reasons.append(f"{feats['severe_diseases']} severe disease episodes")
            recommendations.append("Consult plant pathologist for treatment plan")

        if feats["treatment_failures"] > 0:
            survival -= 0.10 * feats["treatment_failures"]
            reasons.append(f"{feats['treatment_failures']} failed treatments")
            recommendations.append("Try alternative treatment approach")

        if feats["height_deviation_pct"] < -30:
            survival -= 0.10
            reasons.append(f"Severely stunted growth ({feats['height_deviation_pct']:.0f}% below expected)")
            recommendations.append("Investigate root health and soil conditions")

        if feats["graft_factor"] < 0:
            survival -= 0.15
            reasons.append("Graft failure detected")
            recommendations.append("Consider re-grafting or replacement")

        if feats["growth_trend"] < -0.3:
            survival -= 0.10
            reasons.append("Growth rate declining rapidly")

        if feats["stress_count"] > 5:
            survival -= 0.05
            reasons.append(f"Experienced {feats['stress_count']} stress events")

        survival = max(0.05, min(1.0, survival))

        if not reasons:
            reasons.append("Plant appears healthy with no risk factors")

        if not recommendations:
            recommendations.append("Continue standard care protocol")

        # Status classification
        if survival >= 0.85:
            status = "healthy"
            prognosis = "Excellent - plant is thriving"
        elif survival >= 0.65:
            status = "stable"
            prognosis = "Good - minor concerns, monitor regularly"
        elif survival >= 0.40:
            status = "at_risk"
            prognosis = "Concerning - intervention recommended"
        else:
            status = "critical"
            prognosis = "Critical - immediate action required"

        # AI recommendation
        if survival >= 0.85:
            ai_rec = "No intervention needed. This plant is performing well."
        elif survival >= 0.65:
            ai_rec = "Minor preventive action recommended. Optimize growing conditions."
        elif survival >= 0.40:
            ai_rec = f"Intervene within 7 days. Priority: {recommendations[0]}"
        else:
            ai_rec = f"URGENT: Immediate intervention required. {recommendations[0]}. Consider whether resources are better spent on healthier stock."

        return MortalityPrediction(
            plant_id=plant_id,
            survival_probability=round(survival * 100, 1),
            status=status,
            prognosis=prognosis,
            reasons=reasons,
            recommendations=recommendations,
            ai_recommendation=ai_rec,
        ).model_dump()

    def get_zone_health_summary(self, plant_df: pd.DataFrame, growth_df: pd.DataFrame,
                                 disease_df: pd.DataFrame, sensor_df: pd.DataFrame,
                                 zone_id: str) -> Dict:
        """Get overall health summary for a zone."""
        zone_plants = plant_df[plant_df["zone_id"] == zone_id]
        if zone_plants.empty:
            return {"error": f"No plants in zone {zone_id}"}

        total = len(zone_plants)
        healthy = len(zone_plants[zone_plants["health_status"] == "healthy"])
        at_risk = len(zone_plants[zone_plants["health_status"] == "at_risk"])
        critical = len(zone_plants[zone_plants["health_status"] == "critical"])

        # Growth analysis
        deviations = zone_plants["height_deviation_pct"].values
        avg_deviation = deviations.mean()
        stunted = int((deviations < -25).sum())

        # Disease risks
        disease_risks = self.assess_disease_risk(sensor_df, disease_df, zone_id)
        high_risks = [r for r in disease_risks if r["risk_level"] in ["high", "critical"]]

        # Zone health score
        health_score = (healthy / total * 100) - (critical * 5) - (at_risk * 2) - (stunted * 1)
        health_score = max(0, min(100, health_score))

        return {
            "zone_id": zone_id,
            "total_plants": total,
            "health_distribution": {
                "healthy": healthy,
                "at_risk": at_risk,
                "critical": critical,
            },
            "health_score": round(health_score, 1),
            "avg_growth_deviation": f"{avg_deviation:+.1f}%",
            "stunted_plants": stunted,
            "active_disease_risks": len(high_risks),
            "top_risks": [r["disease_name"] for r in high_risks[:3]],
            "recommendations": [
                f"Check {critical} critical plants immediately" if critical > 0 else "No critical plants",
                f"Monitor {at_risk} at-risk plants" if at_risk > 0 else "All plants stable",
                f"Address {len(high_risks)} high disease risks" if high_risks else "Low disease risk",
            ],
        }
