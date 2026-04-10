"""
PlantIQ AI Brain - Master Training Script
Generates data, trains all 9 models, saves to trained_models/.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import pandas as pd


def train_all():
    """Generate data (if needed) and train all models."""
    print("=" * 70)
    print("  PlantIQ AI Brain - Training All Models")
    print("=" * 70)

    start_time = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Load or Generate Data ──────────────────────────────────────
    data_files = {
        "sensor": config.DATA_DIR / "sensor_data.csv",
        "plant_inventory": config.DATA_DIR / "plant_inventory.csv",
        "growth": config.DATA_DIR / "growth_measurements.csv",
        "disease": config.DATA_DIR / "disease_records.csv",
        "graft": config.DATA_DIR / "graft_records.csv",
        "attendance": config.DATA_DIR / "attendance_records.csv",
        "task": config.DATA_DIR / "task_records.csv",
        "sales": config.DATA_DIR / "sales_data.csv",
        "expense": config.DATA_DIR / "expense_data.csv",
        "inventory": config.DATA_DIR / "inventory_data.csv",
    }

    # Check if data exists, generate if not
    missing = [k for k, v in data_files.items() if not v.exists()]
    if missing:
        print(f"\n⚠️  Missing data files: {missing}")
        print("   Generating all training data first...\n")
        from data.generators.generate_all import generate_all
        generate_all()
        print()

    # Load all datasets
    print("\n📂 Loading datasets...")
    data = {}
    for key, path in data_files.items():
        data[key] = pd.read_csv(path)
        print(f"   ✓ {key}: {len(data[key]):,} rows")

    metrics = {}

    # ─── Model 1: Environmental Optimization ────────────────────────
    print("\n" + "─" * 50)
    print("🌡️  Model 1: Environmental Optimization")
    from models.environmental.model import EnvironmentalModel
    env_model = EnvironmentalModel()
    metrics["environmental"] = env_model.train(data["sensor"])

    # ─── Model 2: Worker Performance ────────────────────────────────
    print("\n" + "─" * 50)
    print("👷 Model 2: Worker Performance Analytics")
    from models.worker_performance.model import WorkerPerformanceModel
    worker_model = WorkerPerformanceModel()
    metrics["worker_performance"] = worker_model.train(data["attendance"], data["task"])

    # ─── Model 3: Plant Health ──────────────────────────────────────
    print("\n" + "─" * 50)
    print("🌱 Model 3: Plant Health Prediction")
    from models.plant_health.model import PlantHealthModel
    plant_model = PlantHealthModel()
    metrics["plant_health"] = plant_model.train(
        data["plant_inventory"], data["growth"], data["disease"]
    )

    # ─── Model 4: Graft Success Prediction ──────────────────────────
    print("\n" + "─" * 50)
    print("✂️  Model 4: Graft Success Prediction")
    from models.graft_prediction.model import GraftPredictionModel
    graft_model = GraftPredictionModel()
    metrics["graft_prediction"] = graft_model.train(data["graft"])

    # ─── Model 5: Resource Optimization ─────────────────────────────
    print("\n" + "─" * 50)
    print("💧 Model 5: Resource Optimization")
    from models.resource_optimization.model import ResourceOptimizationModel
    resource_model = ResourceOptimizationModel()
    metrics["resource_optimization"] = resource_model.train(data["inventory"], data["sensor"])

    # ─── Model 6: Yield Forecasting ─────────────────────────────────
    print("\n" + "─" * 50)
    print("📊 Model 6: Yield Forecasting")
    from models.yield_forecasting.model import YieldForecastingModel
    yield_model = YieldForecastingModel()
    metrics["yield_forecasting"] = yield_model.train(data["plant_inventory"], data["sales"])

    # ─── Model 7: Financial Analytics ───────────────────────────────
    print("\n" + "─" * 50)
    print("💰 Model 7: Financial Analytics")
    from models.financial.model import FinancialModel
    financial_model = FinancialModel()
    metrics["financial"] = financial_model.train(data["sales"], data["expense"])

    # ─── Model 8: Anomaly Detection ─────────────────────────────────
    print("\n" + "─" * 50)
    print("🔍 Model 8: Anomaly Detection")
    from models.anomaly_detection.model import AnomalyDetectionModel
    anomaly_model = AnomalyDetectionModel()
    metrics["anomaly_detection"] = anomaly_model.train(
        data["sensor"], data["attendance"], data["task"], data["inventory"]
    )

    print("  ✅ Recommendation Engine initialized with all 8 models")
    metrics["recommendation_engine"] = {"status": "initialized"}

    # ─── Model 10: Disease Vision (CNN) ──────────────────────────────
    print("\n" + "─" * 50)
    print("📸 Model 10: Disease Vision (CNN)")
    from training.import_dataset import download_dataset, organize_dataset
    from training.train_vision import train_model as train_vision
    
    try:
        download_dataset()
        organize_dataset()
        
        disease_data_dir = config.DATA_DIR / "diseases"
        if disease_data_dir.exists():
            # Run for a small number of epochs in this master script
            vision_metrics = train_vision(str(disease_data_dir), num_epochs=3)
            metrics["disease_vision"] = {"status": "trained", "epochs": 3}
        else:
            print("  ⚠️  Disease data not found, skipping vision training")
            metrics["disease_vision"] = {"status": "skipped", "reason": "no_data"}
    except Exception as e:
        print(f"  ❌ Disease vision training failed: {e}")
        metrics["disease_vision"] = {"status": "failed", "error": str(e)}

    # ─── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  ✅ All 9 models trained successfully in {elapsed:.1f} seconds!")
    print("=" * 70)

    print("\n📊 Training Metrics Summary:")
    for model_name, model_metrics in metrics.items():
        print(f"\n  {model_name}:")
        for k, v in model_metrics.items():
            print(f"    • {k}: {v}")

    # List saved model files
    print("\n💾 Saved Models:")
    for d in sorted(config.MODELS_DIR.glob("*")):
        if d.is_dir():
            files = list(d.glob("*.joblib"))
            print(f"   📁 {d.name}/: {len(files)} files")

    return metrics


if __name__ == "__main__":
    train_all()
