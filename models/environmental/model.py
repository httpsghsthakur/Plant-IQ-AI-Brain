"""
PlantIQ AI Brain - Environmental Optimization Model
Time-series analysis, condition optimization, and irrigation scheduling.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from models.environmental.features import (
    compute_zone_features,
    compute_daily_aggregates,
    get_irrigation_features,
)
from models.environmental.schemas import (
    EnvironmentalAnalysis, OptimalRange, Recommendation,
    IrrigationSchedule, IrrigationScheduleItem, WeatherAlert,
)


class EnvironmentalModel:
    """Environmental optimization AI model."""

    def __init__(self):
        self.temp_predictor = None
        self.moisture_predictor = None
        self.risk_classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "environmental"

    def train(self, sensor_df: pd.DataFrame) -> Dict:
        """Train the environmental optimization models."""
        print("  📊 Training Environmental Optimization Model...")

        # Compute daily aggregates for training
        daily = compute_daily_aggregates(sensor_df)
        daily = daily.dropna()

        if len(daily) < 10:
            print("  ⚠️  Insufficient data for environmental model training")
            self.is_trained = True  # Still mark as trained to use rule-based fallback
            return {"status": "rule_based", "reason": "insufficient_data"}

        # Prepare features for temperature prediction (next day)
        feature_cols = [
            "temp_mean", "temp_min", "temp_max", "temp_std",
            "humidity_mean", "moisture_mean", "light_mean",
            "rainfall_total", "wind_mean",
        ]

        # Create lag features
        for col in feature_cols:
            daily[f"{col}_lag1"] = daily.groupby("zone_id")[col].shift(1)
            daily[f"{col}_lag3"] = daily.groupby("zone_id")[col].shift(3)

        daily = daily.dropna()

        lag_cols = [c for c in daily.columns if "lag" in c]
        X = daily[feature_cols + lag_cols].values

        # Temperature prediction target
        y_temp = daily["temp_mean"].values

        # Moisture prediction target
        y_moisture = daily["moisture_mean"].values

        # Risk classification target
        y_risk = (daily["heat_stress_hours"] > 3).astype(int).values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train temperature predictor
        self.temp_predictor = GradientBoostingRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        )
        self.temp_predictor.fit(X_scaled, y_temp)
        temp_score = self.temp_predictor.score(X_scaled, y_temp)

        # Train moisture predictor
        self.moisture_predictor = GradientBoostingRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        )
        self.moisture_predictor.fit(X_scaled, y_moisture)
        moisture_score = self.moisture_predictor.score(X_scaled, y_moisture)

        # Train risk classifier
        self.risk_classifier = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        )
        self.risk_classifier.fit(X_scaled, y_risk)
        risk_score = self.risk_classifier.score(X_scaled, y_risk)

        self.is_trained = True
        self._save_models()

        metrics = {
            "temp_predictor_r2": round(temp_score, 4),
            "moisture_predictor_r2": round(moisture_score, 4),
            "risk_classifier_accuracy": round(risk_score, 4),
            "training_samples": len(X),
            "features_used": len(feature_cols + lag_cols),
        }
        print(f"  ✅ Environmental Model trained: temp R²={temp_score:.3f}, moisture R²={moisture_score:.3f}, risk acc={risk_score:.3f}")
        return metrics

    def _save_models(self):
        """Save trained models to disk."""
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.temp_predictor, self.model_path / "temp_predictor.joblib")
        joblib.dump(self.moisture_predictor, self.model_path / "moisture_predictor.joblib")
        joblib.dump(self.risk_classifier, self.model_path / "risk_classifier.joblib")
        joblib.dump(self.scaler, self.model_path / "scaler.joblib")

    def load_models(self):
        """Load trained models from disk."""
        try:
            self.temp_predictor = joblib.load(self.model_path / "temp_predictor.joblib")
            self.moisture_predictor = joblib.load(self.model_path / "moisture_predictor.joblib")
            self.risk_classifier = joblib.load(self.model_path / "risk_classifier.joblib")
            self.scaler = joblib.load(self.model_path / "scaler.joblib")
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def analyze_zone(self, sensor_df: pd.DataFrame, zone_id: str) -> Dict:
        """Analyze environmental conditions for a specific zone."""
        zone_data = sensor_df[sensor_df["zone_id"] == zone_id]
        if zone_data.empty:
            return {"error": f"No data for zone {zone_id}"}

        # Get latest readings
        zone_data = zone_data.copy()
        zone_data["timestamp"] = pd.to_datetime(zone_data["timestamp"])
        latest = zone_data.sort_values("timestamp").iloc[-1]

        zone_name = latest.get("zone_name", zone_id)
        current = {
            "temperature": float(latest["temperature"]),
            "humidity": float(latest["humidity"]),
            "soil_moisture": float(latest["soil_moisture"]),
            "light_intensity": float(latest.get("light_intensity", 0)),
            "soil_ph": float(latest.get("soil_ph", 6.8)),
        }

        # Evaluate each parameter against optimal ranges
        optimal_ranges = []
        recommendations = []
        risk_total = 0

        # Temperature check
        temp = current["temperature"]
        opt = config.OPTIMAL_CONDITIONS["temperature"]
        if temp < opt["critical_low"]:
            status = "critical"
            risk_total += 30
            recommendations.append(Recommendation(
                action="Protect plants from frost",
                method="Activate heaters, cover with frost blankets",
                urgency="critical",
                impact=f"Prevent frost damage at {temp}°C"
            ))
        elif temp < opt["min"]:
            status = "below"
            risk_total += 15
            recommendations.append(Recommendation(
                action="Increase temperature",
                method="Close greenhouse vents, activate heating if available",
                urgency="medium",
                impact="Improve growth rate by 10-15%"
            ))
        elif temp > opt["critical_high"]:
            status = "critical"
            risk_total += 30
            recommendations.append(Recommendation(
                action="Emergency cooling",
                method="Activate all ventilation, misting systems, deploy shade nets 100%",
                urgency="critical",
                impact=f"Prevent heat death at {temp}°C"
            ))
        elif temp > opt["max"]:
            status = "above"
            risk_total += 15
            recommendations.append(Recommendation(
                action="Reduce temperature",
                method="Open vents, activate misting system",
                urgency="high",
                impact="Prevent heat stress, improve growth by 15%"
            ))
        else:
            status = "optimal"

        optimal_ranges.append(OptimalRange(
            parameter="temperature",
            current_value=temp,
            optimal_min=opt["min"],
            optimal_max=opt["max"],
            status=status,
            deviation_pct=round(((temp - (opt["min"]+opt["max"])/2) / ((opt["max"]-opt["min"])/2)) * 100, 1)
        ))

        # Humidity check
        hum = current["humidity"]
        opt_h = config.OPTIMAL_CONDITIONS["humidity"]
        if hum < opt_h["min"]:
            h_status = "below"
            risk_total += 10
            recommendations.append(Recommendation(
                action="Increase humidity",
                method="Activate misting system, reduce ventilation",
                urgency="medium",
                impact="Prevent desiccation, improve leaf health"
            ))
        elif hum > opt_h["max"]:
            h_status = "above"
            risk_total += 15
            recommendations.append(Recommendation(
                action="Reduce humidity",
                method="Increase ventilation, activate exhaust fans",
                urgency="high",
                impact="Reduce disease risk (fungal infections) by 30%"
            ))
        else:
            h_status = "optimal"

        optimal_ranges.append(OptimalRange(
            parameter="humidity",
            current_value=hum,
            optimal_min=opt_h["min"],
            optimal_max=opt_h["max"],
            status=h_status,
            deviation_pct=round(((hum - (opt_h["min"]+opt_h["max"])/2) / ((opt_h["max"]-opt_h["min"])/2)) * 100, 1)
        ))

        # Soil moisture check
        moist = current["soil_moisture"]
        opt_m = config.OPTIMAL_CONDITIONS["soil_moisture"]
        if moist < opt_m["critical_low"]:
            m_status = "critical"
            risk_total += 25
            recommendations.append(Recommendation(
                action="Emergency irrigation",
                method="Activate drip irrigation immediately for 30 minutes",
                urgency="critical",
                impact=f"Prevent plant death, soil at {moist}%",
                estimated_cost=200
            ))
        elif moist < opt_m["min"]:
            m_status = "below"
            risk_total += 15
            recommendations.append(Recommendation(
                action="Increase soil moisture",
                method="Activate drip irrigation for 15 minutes",
                urgency="medium",
                impact="Prevent wilting, maintain optimal growth",
                estimated_cost=100
            ))
        elif moist > opt_m["critical_high"]:
            m_status = "critical"
            risk_total += 20
            recommendations.append(Recommendation(
                action="Reduce soil moisture",
                method="Check drainage, stop irrigation, improve aeration",
                urgency="high",
                impact="Prevent root rot, waterlogging"
            ))
        else:
            m_status = "optimal"

        optimal_ranges.append(OptimalRange(
            parameter="soil_moisture",
            current_value=moist,
            optimal_min=opt_m["min"],
            optimal_max=opt_m["max"],
            status=m_status,
            deviation_pct=round(((moist - (opt_m["min"]+opt_m["max"])/2) / ((opt_m["max"]-opt_m["min"])/2)) * 100, 1)
        ))

        # Overall status
        risk_score = min(100, risk_total)
        if risk_score >= 50:
            overall = "critical"
        elif risk_score >= 20:
            overall = "needs_attention"
        else:
            overall = "optimal"

        return EnvironmentalAnalysis(
            zone_id=zone_id,
            zone_name=zone_name,
            timestamp=datetime.now().isoformat(),
            current_conditions=current,
            optimal_ranges=optimal_ranges,
            recommendations=recommendations,
            overall_status=overall,
            risk_score=risk_score,
        ).model_dump()

    def get_irrigation_schedule(self, sensor_df: pd.DataFrame) -> Dict:
        """Generate optimal irrigation schedule for all zones."""
        schedule_items = []
        total_water = 0
        fixed_water = 0

        for zone in config.ZONES:
            zone_id = zone["id"]
            feats = get_irrigation_features(sensor_df, zone_id)

            moisture = feats["current_moisture"]
            trend = feats["moisture_trend"]
            rain = feats["total_rainfall_24h"]
            evap = feats["evaporation_estimate"]

            # Calculate water need
            target_moisture = 50
            deficit = max(0, target_moisture - moisture)
            predicted_loss = evap * 4  # Next 4 hours estimated loss

            # Skip if recent rain
            if rain > 5:
                continue

            if moisture < 42 or (moisture < 50 and trend < -0.5):
                # Calculate duration based on deficit
                duration = int(min(45, max(10, deficit * 2)))
                water_amount = duration * zone["capacity"] * 0.003  # liters

                urgency = "high" if moisture < 35 else "medium"
                reason = f"Soil moisture at {moisture:.0f}% (trend: {trend:+.1f}%/hr)"

                # Best time for irrigation
                now = datetime.now()
                if now.hour < 6:
                    next_time = now.replace(hour=6, minute=0)
                elif now.hour < 17:
                    next_time = now.replace(hour=17, minute=0)
                else:
                    next_time = (now + timedelta(days=1)).replace(hour=6, minute=0)

                schedule_items.append(IrrigationScheduleItem(
                    zone_id=zone_id,
                    zone_name=zone["name"],
                    next_irrigation=next_time.strftime("%Y-%m-%d %H:%M"),
                    duration_minutes=duration,
                    water_amount_liters=round(water_amount, 0),
                    reason=reason,
                    urgency=urgency,
                ))
                total_water += water_amount

            # Fixed schedule comparison
            fixed_water += zone["capacity"] * 0.005 * 20  # 20 min daily

        savings_pct = round((1 - total_water / max(1, fixed_water)) * 100, 1) if fixed_water > 0 else 0

        return IrrigationSchedule(
            schedule=schedule_items,
            total_water_liters=round(total_water, 0),
            water_savings_pct=max(0, savings_pct),
            cost_savings_monthly=round(savings_pct * 450, 0),  # Estimate
        ).model_dump()

    def get_weather_alerts(self, sensor_df: pd.DataFrame) -> List[Dict]:
        """Generate weather-based proactive alerts from sensor trends."""
        alerts = []
        df = sensor_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        for zone in config.ZONES:
            zone_id = zone["id"]
            zone_data = df[df["zone_id"] == zone_id].sort_values("timestamp").tail(48)

            if zone_data.empty:
                continue

            latest_temp = zone_data["temperature"].iloc[-1]
            temp_trend = zone_data["temperature"].diff().mean()
            latest_humidity = zone_data["humidity"].iloc[-1]
            rainfall_24h = zone_data["rainfall_mm"].sum()

            # Frost alert
            if latest_temp < 8 and temp_trend < 0:
                predicted_min = latest_temp + temp_trend * 6
                if predicted_min < 3:
                    alerts.append(WeatherAlert(
                        alert_type="frost_warning",
                        severity="critical",
                        message=f"Frost warning: Temperature dropping to {predicted_min:.0f}°C tonight. Cover young plants!",
                        affected_zones=[zone_id],
                        recommended_actions=[
                            "Cover all young seedlings with frost blankets",
                            "Activate greenhouse heaters",
                            "Move sensitive plants indoors if possible",
                        ],
                        valid_until=(datetime.now() + timedelta(hours=12)).isoformat(),
                    ).model_dump())

            # Heatwave alert
            if latest_temp > 33 and temp_trend > 0:
                alerts.append(WeatherAlert(
                    alert_type="heatwave",
                    severity="high",
                    message=f"Heatwave alert: Temperature at {latest_temp:.0f}°C and rising. Increase watering!",
                    affected_zones=[zone_id],
                    recommended_actions=[
                        "Increase watering frequency for next 3 days",
                        "Deploy shade nets at 50-70%",
                        "Activate misting systems during peak hours",
                    ],
                    valid_until=(datetime.now() + timedelta(hours=24)).isoformat(),
                ).model_dump())

            # Heavy rain alert
            if rainfall_24h > 20:
                alerts.append(WeatherAlert(
                    alert_type="heavy_rain",
                    severity="medium",
                    message=f"Heavy rain: {rainfall_24h:.0f}mm in 24 hours. Check drainage!",
                    affected_zones=[zone_id],
                    recommended_actions=[
                        "Close greenhouse vents",
                        "Check drainage systems",
                        "Hold off on irrigation",
                        "Monitor for waterlogging",
                    ],
                    valid_until=(datetime.now() + timedelta(hours=6)).isoformat(),
                ).model_dump())

        return alerts
