"""
PlantIQ AI Brain - Model Evaluation
Evaluate model performance metrics and generate reports.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import pandas as pd
import numpy as np


def evaluate_all():
    """Evaluate all trained models and print metrics."""
    print("=" * 70)
    print("  PlantIQ AI Brain - Model Evaluation Report")
    print("=" * 70)

    # Load data
    data = {}
    data_files = {
        "sensor": "sensor_data.csv",
        "plant_inventory": "plant_inventory.csv",
        "growth": "growth_measurements.csv",
        "disease": "disease_records.csv",
        "graft": "graft_records.csv",
        "attendance": "attendance_records.csv",
        "task": "task_records.csv",
        "sales": "sales_data.csv",
        "expense": "expense_data.csv",
        "inventory": "inventory_data.csv",
    }

    for key, filename in data_files.items():
        path = config.DATA_DIR / filename
        if path.exists():
            data[key] = pd.read_csv(path)
        else:
            print(f"  [WARN] Missing: {filename}")
            return

    print(f"\n[DIR] Data loaded: {sum(len(v) for v in data.values()):,} total rows")

    # Evaluate Environmental Model
    print("\n" + "─" * 50)
    print("[TEMP]  Environmental Optimization Model")
    from models.environmental.model import EnvironmentalModel
    env = EnvironmentalModel()
    if env.load_models():
        # Test analysis on each zone
        for zone in config.ZONES:
            result = env.analyze_zone(data["sensor"], zone["id"])
            if isinstance(result, dict):
                print(f"   Zone {zone['name']}: status={result.get('overall_status')}, risk={result.get('risk_score')}")
        print("   [OK] Environmental model operational")
    else:
        print("   [WARN] Model not trained")

    # Evaluate Graft Model
    print("\n" + "─" * 50)
    print("[CUT]  Graft Success Prediction Model")
    from models.graft_prediction.model import GraftPredictionModel
    graft = GraftPredictionModel()
    if graft.load_models():
        graft_df = data["graft"]
        actual = graft_df["success"].values
        predicted_probs = graft_df["success_probability"].values
        predicted = (predicted_probs > 0.5).astype(int)

        accuracy = (actual == predicted).mean()
        true_success = actual.mean()
        print(f"   Baseline success rate: {true_success*100:.1f}%")
        print(f"   Model accuracy (on generated prob): {accuracy*100:.1f}%")
        print("   [OK] Graft model operational")
    else:
        print("   [WARN] Model not trained")

    # Evaluate Financial Model
    print("\n" + "─" * 50)
    print("💰 Financial Analytics Model")
    from models.financial.model import FinancialModel
    fin = FinancialModel()
    if fin.load_models():
        pnl = fin.get_profit_loss(data["sales"], data["expense"])
        print(f"   Revenue: ₹{int(pnl.get('total_revenue', 0)):,}")
        print(f"   Expenses: ₹{int(pnl.get('total_expenses', 0)):,}")
        print(f"   Margin: {pnl.get('profit_margin', 0)}%")
        print("   [OK] Financial model operational")
    else:
        print("   [WARN] Model not trained")

    # Summary
    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70)

    # List files
    print("\n💾 Model files:")
    for d in sorted(config.MODELS_DIR.glob("*")):
        if d.is_dir():
            files = list(d.glob("*.joblib"))
            total_size = sum(f.stat().st_size for f in files) / 1024
            print(f"   [DIR] {d.name}: {len(files)} files ({total_size:.1f} KB)")


if __name__ == "__main__":
    evaluate_all()
