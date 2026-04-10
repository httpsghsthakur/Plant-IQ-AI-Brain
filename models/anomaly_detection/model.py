"""
PlantIQ AI Brain - Anomaly Detection Model
Isolation Forest for sensor anomalies, statistical anomaly detection for worker/inventory.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from datetime import datetime
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from models.anomaly_detection.features import (
    compute_sensor_baselines,
    detect_sensor_spikes,
    detect_stuck_sensors,
    compute_worker_baseline,
)
from models.anomaly_detection.schemas import (
    SensorAnomaly, WorkerAnomaly, InventoryAnomaly, AnomalyReport,
)


class AnomalyDetectionModel:
    """Anomaly Detection AI model."""

    def __init__(self):
        self.isolation_forest = None
        self.sensor_baselines = {}
        self.worker_baselines = {}
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "anomaly_detection"

    def train(self, sensor_df: pd.DataFrame, attendance_df: pd.DataFrame,
              task_df: pd.DataFrame, inventory_df: pd.DataFrame) -> Dict:
        """Train anomaly detection models."""
        print("  📊 Training Anomaly Detection Model...")

        # Compute baselines
        self.sensor_baselines = compute_sensor_baselines(sensor_df)
        self.worker_baselines = compute_worker_baseline(attendance_df, task_df)

        # Train Isolation Forest on sensor data
        sensor_cols = ["temperature", "humidity", "soil_moisture"]
        available_cols = [c for c in sensor_cols if c in sensor_df.columns]

        if len(available_cols) >= 2:
            X = sensor_df[available_cols].dropna().values
            if len(X) > 100:
                self.isolation_forest = IsolationForest(
                    n_estimators=100,
                    contamination=0.05,
                    random_state=config.RANDOM_STATE,
                )
                self.isolation_forest.fit(X)

        self.is_trained = True
        self._save_models()

        metrics = {
            "zones_monitored": len(self.sensor_baselines),
            "workers_baselined": len(self.worker_baselines),
            "isolation_forest_trained": self.isolation_forest is not None,
        }
        print(f"  ✅ Anomaly Detection Model trained: {len(self.sensor_baselines)} zones, {len(self.worker_baselines)} workers")
        return metrics

    def _save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.sensor_baselines, self.model_path / "sensor_baselines.joblib")
        joblib.dump(self.worker_baselines, self.model_path / "worker_baselines.joblib")
        if self.isolation_forest:
            joblib.dump(self.isolation_forest, self.model_path / "isolation_forest.joblib")

    def load_models(self):
        try:
            self.sensor_baselines = joblib.load(self.model_path / "sensor_baselines.joblib")
            self.worker_baselines = joblib.load(self.model_path / "worker_baselines.joblib")
            try:
                self.isolation_forest = joblib.load(self.model_path / "isolation_forest.joblib")
            except FileNotFoundError:
                pass
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def detect_sensor_anomalies(self, sensor_df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in sensor readings."""
        anomalies = []

        # Statistical spike detection
        spikes = detect_sensor_spikes(sensor_df, self.sensor_baselines)
        for spike in spikes:
            zone_name = ""
            for z in config.ZONES:
                if z["id"] == spike["zone_id"]:
                    zone_name = z["name"]
                    break

            severity = "critical" if spike["z_score"] > 5 else "high" if spike["z_score"] > 4 else "medium"

            sensor_labels = {
                "temperature": "Temperature",
                "humidity": "Humidity",
                "soil_moisture": "Soil Moisture",
                "light_intensity": "Light Intensity",
                "soil_ph": "Soil pH",
            }

            action = "Check sensor calibration"
            if spike["sensor_type"] == "temperature" and spike["value"] > 40:
                action = "URGENT: Verify reading. If real, activate emergency cooling"
            elif spike["sensor_type"] == "soil_moisture" and spike["value"] < 15:
                action = "URGENT: Verify reading. If real, irrigate immediately"
            elif spike["sensor_type"] == "soil_ph" and (spike["value"] < 5 or spike["value"] > 9):
                action = "Verify sensor. If pH confirmed, perform soil treatment"

            anomalies.append(SensorAnomaly(
                sensor_type=sensor_labels.get(spike["sensor_type"], spike["sensor_type"]),
                zone_id=spike["zone_id"],
                zone_name=zone_name,
                anomaly_type="spike",
                severity=severity,
                current_value=spike["value"],
                expected_range={
                    "mean": spike["expected_mean"],
                    "low": round(spike["expected_mean"] - 2 * spike["expected_std"], 1),
                    "high": round(spike["expected_mean"] + 2 * spike["expected_std"], 1),
                },
                detection_method=f"Statistical (z-score: {spike['z_score']})",
                recommended_action=action,
                timestamp=spike.get("timestamp", datetime.now().isoformat()),
            ).model_dump())

        # Stuck sensor detection
        stuck = detect_stuck_sensors(sensor_df)
        for s in stuck:
            zone_name = ""
            for z in config.ZONES:
                if z["id"] == s["zone_id"]:
                    zone_name = z["name"]
                    break

            anomalies.append(SensorAnomaly(
                sensor_type=s["sensor_type"],
                zone_id=s["zone_id"],
                zone_name=zone_name,
                anomaly_type="stuck",
                severity="high",
                current_value=s["stuck_value"],
                expected_range={"note": f"Same value for {s['duration_hours']} hours"},
                detection_method="Variance analysis (std ≈ 0)",
                recommended_action=f"Check {s['sensor_type']} sensor in zone {s['zone_id']} — may be malfunctioning",
                timestamp=datetime.now().isoformat(),
            ).model_dump())

        return anomalies

    def detect_worker_anomalies(self, attendance_df: pd.DataFrame,
                                 task_df: pd.DataFrame) -> List[Dict]:
        """Detect unusual worker activity patterns."""
        anomalies = []
        att = attendance_df.copy()
        att["date"] = pd.to_datetime(att["date"])

        # Check last 7 days
        cutoff = att["date"].max() - pd.Timedelta(days=7)
        recent_att = att[att["date"] >= cutoff]

        for wid, baseline in self.worker_baselines.items():
            worker_recent = recent_att[recent_att["worker_id"] == wid]
            if worker_recent.empty:
                continue

            worker_name = worker_recent["worker_name"].iloc[0]
            present = worker_recent[worker_recent["status"] == "present"]

            # Unusual absence pattern
            recent_absence_rate = 1 - len(present) / max(1, len(worker_recent))
            if recent_absence_rate > baseline["absence_rate"] * 2 and recent_absence_rate > 0.3:
                anomalies.append(WorkerAnomaly(
                    worker_id=wid,
                    worker_name=worker_name,
                    anomaly_type="unusual_absence",
                    description=f"Absence rate {recent_absence_rate*100:.0f}% vs baseline {baseline['absence_rate']*100:.0f}%",
                    severity="medium",
                    recommended_action="Check with worker — may indicate personal issues or dissatisfaction",
                ).model_dump())

            # Unusual hours (working much more or less)
            if len(present) > 0:
                recent_hours = present["work_hours"].mean()
                if baseline["std_hours"] > 0:
                    z = abs(recent_hours - baseline["avg_hours"]) / baseline["std_hours"]
                    if z > 2.5:
                        direction = "more" if recent_hours > baseline["avg_hours"] else "fewer"
                        anomalies.append(WorkerAnomaly(
                            worker_id=wid,
                            worker_name=worker_name,
                            anomaly_type="unusual_hours",
                            description=f"Working {direction} hours than usual ({recent_hours:.1f} vs avg {baseline['avg_hours']:.1f})",
                            severity="low" if direction == "more" else "medium",
                            recommended_action=f"Review workload assignment for {worker_name}",
                        ).model_dump())

        return anomalies

    def detect_inventory_anomalies(self, inventory_df: pd.DataFrame) -> List[Dict]:
        """Detect unusual inventory consumption patterns."""
        anomalies = []
        df = inventory_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        items = df["item_name"].unique()
        for item in items:
            item_data = df[df["item_name"] == item].sort_values("date")
            if len(item_data) < 14:
                continue

            recent = item_data.tail(7)
            historical = item_data.iloc[:-7]

            # Unusual consumption
            recent_avg = recent["consumed"].mean()
            hist_avg = historical["consumed"].mean()
            hist_std = historical["consumed"].std()

            if hist_std > 0 and recent_avg > hist_avg + 2.5 * hist_std:
                anomalies.append(InventoryAnomaly(
                    item_name=item,
                    anomaly_type="consumption_spike",
                    description=f"Recent daily usage {recent_avg:.1f} vs historical avg {hist_avg:.1f}",
                    severity="medium",
                    recommended_action=f"Investigate unusual {item} consumption — possible waste or theft",
                ).model_dump())

            # Stock drop anomaly
            if len(recent) >= 2:
                stock_change = recent.iloc[0]["closing_stock"] - recent.iloc[-1]["closing_stock"]
                expected_change = recent_avg * len(recent)
                if stock_change > expected_change * 2:
                    anomalies.append(InventoryAnomaly(
                        item_name=item,
                        anomaly_type="unexpected_stock_drop",
                        description=f"Stock dropped by {stock_change:.0f} units (expected ~{expected_change:.0f})",
                        severity="high",
                        recommended_action=f"Audit {item} inventory — check for recording errors or misuse",
                    ).model_dump())

        return anomalies

    def get_anomaly_report(self, sensor_df: pd.DataFrame, attendance_df: pd.DataFrame,
                            task_df: pd.DataFrame, inventory_df: pd.DataFrame) -> Dict:
        """Generate comprehensive anomaly report."""
        sensor_anomalies = self.detect_sensor_anomalies(sensor_df)
        worker_anomalies = self.detect_worker_anomalies(attendance_df, task_df)
        inventory_anomalies = self.detect_inventory_anomalies(inventory_df)

        all_anomalies = sensor_anomalies + worker_anomalies + inventory_anomalies
        critical = sum(1 for a in all_anomalies if a.get("severity") == "critical")

        return AnomalyReport(
            total_anomalies=len(all_anomalies),
            critical_count=critical,
            sensor_anomalies=sensor_anomalies,
            worker_anomalies=worker_anomalies,
            inventory_anomalies=inventory_anomalies,
            generated_at=datetime.now().isoformat(),
        ).model_dump()
