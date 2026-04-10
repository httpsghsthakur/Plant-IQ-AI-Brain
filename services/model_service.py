"""
PlantIQ AI Brain - Model Service
Loads and manages all trained AI models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from models.environmental.model import EnvironmentalModel
from models.worker_performance.model import WorkerPerformanceModel
from models.plant_health.model import PlantHealthModel
from models.graft_prediction.model import GraftPredictionModel
from models.resource_optimization.model import ResourceOptimizationModel
from models.yield_forecasting.model import YieldForecastingModel
from models.financial.model import FinancialModel
from models.anomaly_detection.model import AnomalyDetectionModel
from models.recommendation_engine.engine import RecommendationEngine


class ModelService:
    """Centralized model loading and management."""

    def __init__(self):
        self.environmental = EnvironmentalModel()
        self.worker = WorkerPerformanceModel()
        self.plant_health = PlantHealthModel()
        self.graft = GraftPredictionModel()
        self.resource = ResourceOptimizationModel()
        self.yield_forecast = YieldForecastingModel()
        self.financial = FinancialModel()
        self.anomaly = AnomalyDetectionModel()
        self.recommendation = RecommendationEngine()
        self._loaded = False

    def load_all(self) -> dict:
        """Load all trained models from disk."""
        results = {}
        results["environmental"] = self.environmental.load_models()
        results["worker_performance"] = self.worker.load_models()
        results["plant_health"] = self.plant_health.load_models()
        results["graft_prediction"] = self.graft.load_models()
        results["resource_optimization"] = self.resource.load_models()
        results["yield_forecasting"] = self.yield_forecast.load_models()
        results["financial"] = self.financial.load_models()
        results["anomaly_detection"] = self.anomaly.load_models()

        # Wire up recommendation engine
        self.recommendation.set_models(
            environmental=self.environmental,
            worker=self.worker,
            plant_health=self.plant_health,
            graft=self.graft,
            resource=self.resource,
            yield_forecast=self.yield_forecast,
            financial=self.financial,
            anomaly=self.anomaly,
        )

        self._loaded = True
        loaded_count = sum(1 for v in results.values() if v)
        print(f"  📦 Models loaded: {loaded_count}/8 from disk")
        return results

    def get_status(self) -> dict:
        """Get status of all models."""
        return {
            "models": {
                "environmental": {"trained": self.environmental.is_trained},
                "worker_performance": {"trained": self.worker.is_trained},
                "plant_health": {"trained": self.plant_health.is_trained},
                "graft_prediction": {"trained": self.graft.is_trained},
                "resource_optimization": {"trained": self.resource.is_trained},
                "yield_forecasting": {"trained": self.yield_forecast.is_trained},
                "financial": {"trained": self.financial.is_trained},
                "anomaly_detection": {"trained": self.anomaly.is_trained},
                "recommendation_engine": {"status": "active"},
            },
            "all_loaded": self._loaded,
        }


# Singleton instance
model_service = ModelService()
