"""
PlantIQ AI Brain - Graft Success Prediction Model
XGBoost success predictor, worker-method optimization, batch assignment optimizer.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from models.graft_prediction.features import (
    compute_graft_features,
    compute_worker_graft_stats,
    compute_method_stats,
    prepare_graft_training_data,
)
from models.graft_prediction.schemas import (
    GraftPrediction, WorkerMethodStats, BatchOptimization, GraftMonitoring,
)


class GraftPredictionModel:
    """Graft Success Prediction AI model."""

    def __init__(self):
        self.success_predictor = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = config.MODELS_DIR / "graft_prediction"

    def train(self, graft_df: pd.DataFrame) -> Dict:
        """Train the graft success prediction model."""
        print("  [*] Training Graft Success Prediction Model...")

        X, y = prepare_graft_training_data(graft_df)
        if len(X) < 50:
            print("  [WARN]  Insufficient graft data for training")
            self.is_trained = True
            return {"status": "rule_based", "reason": "insufficient_data"}

        X_scaled = self.scaler.fit_transform(X)

        self.success_predictor = GradientBoostingClassifier(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            random_state=config.RANDOM_STATE,
        )
        self.success_predictor.fit(X_scaled, y)
        accuracy = self.success_predictor.score(X_scaled, y)

        # Feature importance
        feature_names = [
            "rootstock_age", "scion_freshness", "temperature", "humidity",
            "callus_formation", "cambium_alignment", "morning", "afternoon",
            "age_optimal", "scion_fresh", "temp_optimal", "humidity_optimal",
            "worker_success_rate", "worker_total_grafts", "worker_skill_trend",
        ]
        importances = self.success_predictor.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]

        self.is_trained = True
        self._save_models()

        metrics = {
            "accuracy": round(accuracy, 4),
            "training_samples": len(X),
            "positive_rate": round(y.mean(), 4),
            "top_features": {name: round(float(imp), 4) for name, imp in top_features},
        }
        print(f"  [OK] Graft Prediction Model trained: accuracy={accuracy:.3f}, samples={len(X)}")
        return metrics

    def _save_models(self):
        self.model_path.mkdir(parents=True, exist_ok=True)
        if self.success_predictor:
            joblib.dump(self.success_predictor, self.model_path / "success_predictor.joblib")
        joblib.dump(self.scaler, self.model_path / "scaler.joblib")

    def load_models(self):
        try:
            self.success_predictor = joblib.load(self.model_path / "success_predictor.joblib")
            self.scaler = joblib.load(self.model_path / "scaler.joblib")
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

    def predict_success(self, graft_df: pd.DataFrame, worker_id: str, method: str,
                        rootstock_variety: str, scion_variety: str,
                        rootstock_age_days: int = 180, scion_freshness_days: int = 1,
                        temperature: float = 23, humidity: float = 65) -> Dict:
        """Predict success probability for a planned graft."""
        worker_stats = compute_worker_graft_stats(graft_df, worker_id, method)

        # Get worker name
        worker_rows = graft_df[graft_df["worker_id"] == worker_id]
        worker_name = worker_rows["worker_name"].iloc[0] if len(worker_rows) > 0 else worker_id

        # Build feature vector
        feature_vector = [
            rootstock_age_days, scion_freshness_days, temperature, humidity,
            75, 80,  # Default callus/cambium for pre-graft prediction
            1 if True else 0,  # morning default
            0,  # afternoon
            1 if 150 <= rootstock_age_days <= 240 else 0,
            1 if scion_freshness_days <= 2 else 0,
            1 if 18 <= temperature <= 28 else 0,
            1 if 50 <= humidity <= 75 else 0,
            worker_stats["success_rate"],
            worker_stats["total_grafts"],
            worker_stats["skill_trend"],
        ]

        # Predict with model or rule-based
        if self.success_predictor:
            X = self.scaler.transform([feature_vector])
            success_prob = self.success_predictor.predict_proba(X)[0][1]
        else:
            # Rule-based fallback
            success_prob = (
                worker_stats["success_rate"] * 0.35
                + (1.0 if 150 <= rootstock_age_days <= 240 else 0.8) * 0.20
                + (1.0 - scion_freshness_days * 0.03) * 0.15
                + (1.0 if 18 <= temperature <= 28 else 0.8) * 0.15
                + (1.0 if 50 <= humidity <= 75 else 0.85) * 0.15
            )

        # Identify risk factors
        risk_factors = []
        recommendations = []
        optimal_conditions = {}

        if rootstock_age_days < 150:
            risk_factors.append(f"Rootstock too young ({rootstock_age_days} days, optimal: 150-240)")
            recommendations.append("Wait until rootstock reaches 150+ days")
        elif rootstock_age_days > 240:
            risk_factors.append(f"Rootstock older than optimal ({rootstock_age_days} days)")

        if scion_freshness_days > 3:
            risk_factors.append(f"Scion not fresh ({scion_freshness_days} days old)")
            recommendations.append("Use scion material within 2 days of collection")

        if temperature < 18 or temperature > 28:
            risk_factors.append(f"Temperature {temperature}°C outside optimal range (18-28°C)")
            recommendations.append("Schedule grafting for optimal temperature window")

        if humidity < 50 or humidity > 75:
            risk_factors.append(f"Humidity {humidity}% outside optimal range (50-75%)")

        if worker_stats["total_grafts"] < 10:
            risk_factors.append(f"Worker has limited experience with {method} ({worker_stats['total_grafts']} grafts)")
            recommendations.append("Pair with experienced grafter for mentoring")

        if worker_stats["success_rate"] < 0.7:
            risk_factors.append(f"Worker's {method} success rate is below average ({worker_stats['success_rate']*100:.0f}%)")
            recommendations.append(f"Consider training workshop for {method} technique")

        if not risk_factors:
            risk_factors.append("No significant risk factors identified")
        if not recommendations:
            recommendations.append("Proceed with grafting under current conditions")

        optimal_conditions = {
            "temperature": "18-28°C",
            "humidity": "50-75%",
            "time_of_day": "Morning (6-10 AM)",
            "rootstock_age": "150-240 days",
            "scion_freshness": "Same day or within 2 days",
        }

        return GraftPrediction(
            worker_id=worker_id,
            worker_name=worker_name,
            method=method,
            rootstock_variety=rootstock_variety,
            scion_variety=scion_variety,
            success_probability=round(success_prob * 100, 1),
            risk_factors=risk_factors,
            recommendations=recommendations,
            optimal_conditions=optimal_conditions,
        ).model_dump()

    def get_worker_method_analysis(self, graft_df: pd.DataFrame) -> List[Dict]:
        """Analyze each worker's performance per grafting method."""
        results = []
        worker_ids = graft_df["worker_id"].unique()

        for wid in worker_ids:
            worker_rows = graft_df[graft_df["worker_id"] == wid]
            worker_name = worker_rows["worker_name"].iloc[0]

            for method in config.GRAFT_METHODS:
                stats = compute_worker_graft_stats(graft_df, wid, method)
                if stats["total_grafts"] < 3:
                    continue

                # Generate recommendation
                if stats["success_rate"] >= 0.85:
                    rec = f"Expert level - assign priority {method} tasks"
                elif stats["success_rate"] >= 0.70:
                    rec = "Good - continue current practice"
                elif stats["success_rate"] >= 0.55:
                    rec = f"Needs improvement - pair with expert for {method}"
                else:
                    rec = f"Avoid assigning {method} until additional training"

                results.append(WorkerMethodStats(
                    worker_id=wid,
                    worker_name=worker_name,
                    method=method,
                    total_grafts=stats["total_grafts"],
                    success_rate=round(stats["success_rate"] * 100, 1),
                    avg_callus_formation=stats["avg_callus"],
                    recommendation=rec,
                ).model_dump())

        return sorted(results, key=lambda x: x["success_rate"], reverse=True)

    def optimize_batch_assignment(self, graft_df: pd.DataFrame,
                                   grafts_needed: int = 20,
                                   method: str = None) -> Dict:
        """Optimize worker-method assignments for a batch of grafts."""
        worker_ids = graft_df["worker_id"].unique()
        methods = [method] if method else config.GRAFT_METHODS

        # Build worker-method success matrix
        worker_method_scores = {}
        for wid in worker_ids:
            worker_rows = graft_df[graft_df["worker_id"] == wid]
            worker_name = worker_rows["worker_name"].iloc[0]
            worker_method_scores[wid] = {"name": worker_name, "methods": {}}
            for m in methods:
                stats = compute_worker_graft_stats(graft_df, wid, m)
                worker_method_scores[wid]["methods"][m] = stats["success_rate"]

        # Greedy assignment: assign each graft to best available worker-method pair
        assignments = []
        worker_capacity = {wid: 0 for wid in worker_ids}
        max_per_worker = max(3, grafts_needed // len(worker_ids) + 2)

        for i in range(grafts_needed):
            best_score = -1
            best_worker = None
            best_method = None

            for wid in worker_ids:
                if worker_capacity[wid] >= max_per_worker:
                    continue
                for m in methods:
                    score = worker_method_scores[wid]["methods"].get(m, 0)
                    if score > best_score:
                        best_score = score
                        best_worker = wid
                        best_method = m

            if best_worker:
                worker_capacity[best_worker] += 1
                assignments.append({
                    "graft_number": i + 1,
                    "worker_id": best_worker,
                    "worker_name": worker_method_scores[best_worker]["name"],
                    "method": best_method,
                    "expected_success": f"{best_score * 100:.0f}%",
                })

        # Calculate metrics
        avg_success = np.mean([worker_method_scores[a["worker_id"]]["methods"][a["method"]]
                              for a in assignments]) if assignments else 0

        # Random baseline
        random_success = np.mean([
            np.mean(list(worker_method_scores[wid]["methods"].values()))
            for wid in worker_ids
        ])

        improvement = avg_success - random_success

        return BatchOptimization(
            assignments=assignments,
            expected_success_rate=round(avg_success * 100, 1),
            improvement_over_random=f"+{improvement * 100:.1f}% vs random assignment",
            total_grafts=len(assignments),
            estimated_cost_savings=f"₹{int(improvement * len(assignments) * 2500):,} from reduced failures",
        ).model_dump()

    def monitor_graft(self, graft_df: pd.DataFrame, graft_id: str) -> Dict:
        """Monitor a specific graft's progress and predict outcome."""
        graft = graft_df[graft_df["graft_id"] == graft_id]
        if graft.empty:
            return {"error": f"Graft {graft_id} not found"}
        graft = graft.iloc[0]

        callus = float(graft.get("callus_formation_pct", 50))
        cambium = float(graft.get("cambium_alignment_pct", 50))
        days = 14  # Default monitoring period

        # Predict outcome from monitoring data
        if callus >= 80 and cambium >= 85:
            outcome = "likely_success"
            confidence = 90
            action = "Continue standard monitoring. Union forming well."
        elif callus >= 60 and cambium >= 70:
            outcome = "probable_success"
            confidence = 70
            action = "Monitor closely. Maintain humidity around union."
        elif callus >= 40:
            outcome = "uncertain"
            confidence = 50
            action = "Apply wound sealant. Check for infection. Re-wrap if needed."
        else:
            outcome = "likely_failure"
            confidence = 75
            action = "Consider removing and re-grafting. Check for necrosis."

        return GraftMonitoring(
            graft_id=graft_id,
            days_post_graft=days,
            callus_formation_pct=callus,
            cambium_alignment_pct=cambium,
            predicted_outcome=outcome,
            confidence=confidence,
            action_needed=action,
        ).model_dump()
