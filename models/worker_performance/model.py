"""
PlantIQ AI Brain - Worker Performance Analytics Model
Scoring, productivity analysis, burnout detection, and absenteeism prediction.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from models.worker_performance.features import (
    compute_attendance_features,
    compute_productivity_features,
    compute_worker_score,
    compute_burnout_features,
)


class WorkerPerformanceModel:
    """Worker Performance Analytics AI model."""

    def __init__(self):
        self.burnout_classifier = None
        self.absence_predictor = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "worker_performance"

    def train(self, attendance_df: pd.DataFrame, task_df: pd.DataFrame) -> Dict:
        """Train worker performance models."""
        print("  [*] Training Worker Performance Model...")

        # Prepare training data for burnout classifier
        worker_ids = attendance_df["worker_id"].unique()
        burnout_features = []
        burnout_labels = []

        for wid in worker_ids:
            feats = compute_burnout_features(attendance_df, task_df, wid)
            burnout_features.append([
                feats["avg_weekly_hours"],
                feats["max_weekly_hours"],
                feats["weeks_over_50hrs"],
                feats["overtime_total"],
                feats["quality_trend"],
                feats["task_count_recent"],
            ])
            # Label: burnout risk if avg weekly hours > 48 AND quality declining
            is_burnout = 1 if (feats["avg_weekly_hours"] > 48 and feats["quality_trend"] < -3) else 0
            burnout_labels.append(is_burnout)

        X_burnout = np.array(burnout_features)
        y_burnout = np.array(burnout_labels)

        if len(X_burnout) > 5:
            X_scaled = self.scaler.fit_transform(X_burnout)
            self.burnout_classifier = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=config.RANDOM_STATE,
                class_weight="balanced"
            )
            self.burnout_classifier.fit(X_scaled, y_burnout)
            burnout_accuracy = self.burnout_classifier.score(X_scaled, y_burnout)
        else:
            burnout_accuracy = 0

        # Prepare absence prediction model
        absence_features = []
        absence_labels = []

        att_df = attendance_df.copy()
        att_df["date"] = pd.to_datetime(att_df["date"])
        att_df["day_of_week"] = att_df["date"].dt.dayofweek
        att_df["month"] = att_df["date"].dt.month

        for wid in worker_ids:
            worker_att = att_df[att_df["worker_id"] == wid]
            for _, row in worker_att.iterrows():
                absence_features.append([
                    row["day_of_week"],
                    row["month"],
                    len(worker_att[worker_att["status"] == "absent"]) / max(1, len(worker_att)),
                ])
                absence_labels.append(1 if row["status"] == "absent" else 0)

        X_absence = np.array(absence_features)
        y_absence = np.array(absence_labels)

        if len(X_absence) > 20:
            self.absence_predictor = GradientBoostingClassifier(
                n_estimators=50, max_depth=4, random_state=config.RANDOM_STATE,
            )
            self.absence_predictor.fit(X_absence, y_absence)
            absence_accuracy = self.absence_predictor.score(X_absence, y_absence)
        else:
            absence_accuracy = 0

        self.is_trained = True
        self._save_models()

        metrics = {
            "burnout_classifier_accuracy": round(burnout_accuracy, 4),
            "absence_predictor_accuracy": round(absence_accuracy, 4),
            "workers_analyzed": len(worker_ids),
        }
        print(f"  [OK] Worker Performance Model trained: burnout acc={burnout_accuracy:.3f}, absence acc={absence_accuracy:.3f}")
        return metrics

    def _save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        if self.burnout_classifier:
            joblib.dump(self.burnout_classifier, self.model_path / "burnout_classifier.joblib")
        if self.absence_predictor:
            joblib.dump(self.absence_predictor, self.model_path / "absence_predictor.joblib")
        joblib.dump(self.scaler, self.model_path / "scaler.joblib")

    def load_models(self):
        try:
            self.burnout_classifier = joblib.load(self.model_path / "burnout_classifier.joblib")
            self.absence_predictor = joblib.load(self.model_path / "absence_predictor.joblib")
            self.scaler = joblib.load(self.model_path / "scaler.joblib")
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def get_worker_scorecard(self, attendance_df: pd.DataFrame, task_df: pd.DataFrame,
                              worker_id: str, period_days: int = 30) -> Dict:
        """Generate comprehensive performance scorecard for a worker."""
        att_feats = compute_attendance_features(attendance_df, worker_id, period_days)
        prod_feats = compute_productivity_features(task_df, worker_id, period_days)
        scores = compute_worker_score(att_feats, prod_feats)

        # Get worker name
        worker_rows = attendance_df[attendance_df["worker_id"] == worker_id]
        worker_name = worker_rows["worker_name"].iloc[0] if len(worker_rows) > 0 else worker_id

        # Rank among all workers
        all_scores = self.get_all_scores(attendance_df, task_df, period_days)
        sorted_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)
        rank = next((i+1 for i, s in enumerate(sorted_scores) if s["worker_id"] == worker_id), 0)

        # Generate recommendations
        recommendations = []
        if scores["attendance_score"] < 70:
            recommendations.append("Improve attendance regularity - target 95% attendance rate")
        if scores["productivity_score"] < 70:
            recommendations.append("Focus on task completion speed - consider skill training")
        if scores["quality_score"] > 85 and scores["productivity_score"] > 80:
            recommendations.append(f"Consider for senior specialist role - strong performer")
            recommendations.append(f"Reward with performance bonus: ₹{int(scores['composite_score'] * 30)}")
        if prod_feats["error_rate"] > 0.1:
            recommendations.append("Reduce error rate through additional training")
        if att_feats["overtime_hours"] > 20:
            recommendations.append("Monitor overtime levels - risk of burnout")

        return {
            "worker_id": worker_id,
            "worker_name": worker_name,
            "period": f"Last {period_days} days",
            "performance_score": scores["composite_score"],
            "breakdown": {
                "attendance": {
                    "score": scores["attendance_score"],
                    "days_present": att_feats["days_present"],
                    "days_total": att_feats["total_working_days"],
                    "on_time_rate": f"{att_feats['on_time_rate']*100:.0f}%",
                    "late_arrivals": att_feats["late_arrivals"],
                },
                "productivity": {
                    "score": scores["productivity_score"],
                    "tasks_completed": prod_feats["tasks_completed"],
                    "tasks_assigned": prod_feats["tasks_assigned"],
                    "completion_rate": f"{prod_feats['completion_rate']*100:.0f}%",
                    "avg_completion_time": f"{prod_feats['time_efficiency']*100:.0f}% of estimated",
                },
                "quality": {
                    "score": scores["quality_score"],
                    "error_rate": f"{prod_feats['error_rate']*100:.1f}%",
                    "rework_needed": prod_feats["rework_count"],
                },
                "initiative": {
                    "score": scores["initiative_score"],
                },
            },
            "rank": f"{rank}{'st' if rank==1 else 'nd' if rank==2 else 'rd' if rank==3 else 'th'} out of {len(sorted_scores)} workers",
            "trend": "^ Performing well" if scores["composite_score"] > 75 else "-> Stable" if scores["composite_score"] > 60 else "v Needs improvement",
            "recommendations": recommendations,
        }

    def get_all_scores(self, attendance_df: pd.DataFrame, task_df: pd.DataFrame,
                        period_days: int = 30) -> List[Dict]:
        """Get performance scores for all workers."""
        worker_ids = attendance_df["worker_id"].unique()
        scores = []
        for wid in worker_ids:
            att_feats = compute_attendance_features(attendance_df, wid, period_days)
            prod_feats = compute_productivity_features(task_df, wid, period_days)
            worker_scores = compute_worker_score(att_feats, prod_feats)
            worker_rows = attendance_df[attendance_df["worker_id"] == wid]
            name = worker_rows["worker_name"].iloc[0] if len(worker_rows) > 0 else wid
            scores.append({
                "worker_id": wid,
                "worker_name": name,
                "score": worker_scores["composite_score"],
            })
        return scores

    def detect_burnout_risks(self, attendance_df: pd.DataFrame,
                               task_df: pd.DataFrame) -> List[Dict]:
        """Detect workers at risk of burnout."""
        worker_ids = attendance_df["worker_id"].unique()
        alerts = []

        for wid in worker_ids:
            feats = compute_burnout_features(attendance_df, task_df, wid)
            worker_rows = attendance_df[attendance_df["worker_id"] == wid]
            name = worker_rows["worker_name"].iloc[0] if len(worker_rows) > 0 else wid

            # Rule-based burnout detection (works even without trained model)
            risk_score = 0
            indicators = []

            if feats["avg_weekly_hours"] > 50:
                risk_score += 30
                indicators.append(f"Worked {feats['avg_weekly_hours']:.0f}+ hours/week average")
            if feats["weeks_over_50hrs"] >= 2:
                risk_score += 25
                indicators.append(f"Exceeded 50 hrs/week for {feats['weeks_over_50hrs']} consecutive weeks")
            if feats["quality_trend"] < -5:
                risk_score += 25
                indicators.append(f"Quality score dropped by {abs(feats['quality_trend']):.0f} points")
            if feats["overtime_total"] > 30:
                risk_score += 20
                indicators.append(f"Total overtime: {feats['overtime_total']:.0f} hours in 4 weeks")

            if risk_score >= 40:
                risk_level = "High" if risk_score >= 60 else "Medium"
                recommendations = []
                if feats["avg_weekly_hours"] > 50:
                    recommendations.append("Reduce workload by 20% next week")
                if feats["weeks_over_50hrs"] >= 2:
                    recommendations.append("Schedule 2 days off this week")
                recommendations.append("Check-in conversation with manager")
                if risk_score >= 70:
                    recommendations.append("Consider temporary reassignment to lighter duties")

                alerts.append({
                    "worker_id": wid,
                    "worker_name": name,
                    "burnout_risk": f"{risk_level} (Score: {risk_score}/100)",
                    "risk_score": risk_score,
                    "indicators": indicators,
                    "recommendations": recommendations,
                })

        return sorted(alerts, key=lambda x: x["risk_score"], reverse=True)

    def predict_absenteeism(self, attendance_df: pd.DataFrame) -> List[Dict]:
        """Predict workers likely to be absent in coming days."""
        worker_ids = attendance_df["worker_id"].unique()
        predictions = []
        att_df = attendance_df.copy()
        att_df["date"] = pd.to_datetime(att_df["date"])

        for wid in worker_ids:
            worker_att = att_df[att_df["worker_id"] == wid]
            name = worker_att["worker_name"].iloc[0] if len(worker_att) > 0 else wid

            # Analyze patterns
            absent_records = worker_att[worker_att["status"] == "absent"]
            if len(absent_records) == 0:
                continue

            absent_records = absent_records.copy()
            absent_records["day_of_week"] = absent_records["date"].dt.dayofweek
            day_counts = absent_records["day_of_week"].value_counts()

            total_records = len(worker_att)
            absence_rate = len(absent_records) / max(1, total_records)

            # Find most likely absent day
            if len(day_counts) > 0:
                most_likely_day = day_counts.index[0]
                day_probability = day_counts.iloc[0] / max(1, len(worker_att[worker_att["date"].dt.dayofweek == most_likely_day]))

                if day_probability > 0.3:  # More than 30% absence rate on this day
                    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                    predictions.append({
                        "worker_id": wid,
                        "worker_name": name,
                        "absence_probability": round(day_probability * 100, 0),
                        "most_likely_day": day_names[most_likely_day],
                        "overall_absence_rate": f"{absence_rate*100:.1f}%",
                        "reason": f"Pattern: Absent on {int(day_counts.iloc[0])}/{len(worker_att[worker_att['date'].dt.dayofweek == most_likely_day])} previous {day_names[most_likely_day]}s",
                        "recommended_action": "Contact worker to confirm availability, arrange backup",
                    })

        return sorted(predictions, key=lambda x: x["absence_probability"], reverse=True)

    def get_training_needs(self, task_df: pd.DataFrame, graft_df: pd.DataFrame = None) -> List[Dict]:
        """Identify worker training needs from error patterns."""
        worker_ids = task_df["worker_id"].unique()
        training_needs = []

        # Analyze task performance by type
        for task_type in task_df["task_type"].unique():
            type_tasks = task_df[task_df["task_type"] == task_type]
            type_completed = type_tasks[type_tasks["status"] == "completed"]
            if len(type_completed) < 10:
                continue

            avg_quality = type_completed["quality_score"].mean()

            # Find workers below average
            weak_workers = []
            for wid in worker_ids:
                worker_tasks = type_completed[type_completed["worker_id"] == wid]
                if len(worker_tasks) < 3:
                    continue
                worker_quality = worker_tasks["quality_score"].mean()
                if worker_quality < avg_quality - 10:
                    name = worker_tasks["worker_name"].iloc[0]
                    weak_workers.append(name)

            if weak_workers:
                training_needs.append({
                    "skill": task_type,
                    "workers": weak_workers[:5],
                    "reason": f"Quality score {int(avg_quality - 10)}+ below team average",
                    "training_type": "Hands-on workshop",
                    "duration": "1-2 days",
                    "expected_improvement": f"Increase quality from below avg to team standard",
                    "estimated_roi": f"₹{len(weak_workers) * 10000:,}/year from reduced errors",
                })

        # Graft-specific training if data available
        if graft_df is not None and len(graft_df) > 0:
            overall_success = graft_df["success"].mean()
            for wid in worker_ids:
                worker_grafts = graft_df[graft_df["worker_id"] == wid]
                if len(worker_grafts) < 5:
                    continue
                worker_success = worker_grafts["success"].mean()
                if worker_success < overall_success - 0.15:
                    name = worker_grafts["worker_name"].iloc[0]
                    training_needs.append({
                        "skill": "Advanced grafting techniques",
                        "workers": [name],
                        "reason": f"Graft success rate {worker_success*100:.0f}% vs team avg {overall_success*100:.0f}%",
                        "training_type": "Hands-on workshop with expert",
                        "duration": "2 days",
                        "expected_improvement": f"Increase success rate from {worker_success*100:.0f}% to {overall_success*100:.0f}%",
                        "estimated_roi": f"₹{int((overall_success - worker_success) * 100 * 2500):,}/year",
                    })

        return training_needs

    def get_workload_balance(self, task_df: pd.DataFrame, period_days: int = 7) -> Dict:
        """Analyze and recommend workload balancing."""
        df = task_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        cutoff = df["date"].max() - pd.Timedelta(days=period_days)
        df = df[df["date"] >= cutoff]

        worker_loads = df.groupby(["worker_id", "worker_name"]).size().reset_index(name="task_count")
        avg_load = worker_loads["task_count"].mean()

        overloaded = worker_loads[worker_loads["task_count"] > avg_load * 1.3]
        underutilized = worker_loads[worker_loads["task_count"] < avg_load * 0.7]

        redistributions = []
        for _, over in overloaded.iterrows():
            for _, under in underutilized.iterrows():
                excess = int(over["task_count"] - avg_load)
                redistributions.append({
                    "move_tasks": min(excess, 3),
                    "move_from": f"{over['worker_name']} ({int(over['task_count'])} tasks)",
                    "move_to": f"{under['worker_name']} ({int(under['task_count'])} tasks)",
                    "reason": "Balance workload across team",
                })
                break

        return {
            "current_workload": {
                "overloaded": [f"{r['worker_name']} - {int(r['task_count'])} tasks" for _, r in overloaded.iterrows()],
                "underutilized": [f"{r['worker_name']} - {int(r['task_count'])} tasks" for _, r in underutilized.iterrows()],
                "average_tasks": int(avg_load),
            },
            "recommended_redistribution": redistributions[:5],
            "expected_outcome": f"Reduce overtime by {len(overloaded)*2} hours, improve completion rate by 15%",
        }
