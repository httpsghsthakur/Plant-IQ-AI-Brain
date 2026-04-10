"""
PlantIQ AI Brain - Master Data Generator
Runs all data generators to produce the complete training dataset.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Ensure output directory exists
config.DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_all():
    """Run all data generators."""
    print("=" * 70)
    print("  PlantIQ AI Brain - Generating Synthetic Training Data")
    print("=" * 70)
    print(f"\n📁 Output directory: {config.DATA_DIR}")
    print(f"📅 Generating {config.HISTORY_DAYS} days of data")
    print(f"🌱 Plants: {config.NUM_PLANTS} | 👷 Workers: {config.NUM_WORKERS} | 🏗️ Zones: {config.NUM_ZONES}")
    print("-" * 70)

    start_time = time.time()

    # 1. Sensor Data
    print("\n🌡️  Generating IoT sensor data...")
    from data.generators.sensor_data_generator import generate_sensor_data
    generate_sensor_data()

    # 2. Worker Data
    print("\n👷 Generating worker data...")
    from data.generators.worker_data_generator import (
        generate_worker_profiles,
        generate_attendance_records,
        generate_task_records
    )
    generate_worker_profiles()
    generate_attendance_records()
    generate_task_records()

    # 3. Plant Data
    print("\n🌱 Generating plant data...")
    from data.generators.plant_data_generator import (
        generate_plant_inventory,
        generate_growth_measurements,
        generate_disease_records
    )
    generate_plant_inventory()
    generate_growth_measurements()
    generate_disease_records()

    # 4. Graft Data
    print("\n✂️ Generating graft data...")
    from data.generators.graft_data_generator import generate_graft_records
    generate_graft_records()

    # 5. Inventory Data
    print("\n📦 Generating inventory data...")
    from data.generators.inventory_data_generator import generate_inventory_data
    generate_inventory_data()

    # 6. Financial Data
    print("\n💰 Generating financial data...")
    from data.generators.financial_data_generator import generate_sales_data, generate_expense_data
    generate_sales_data()
    generate_expense_data()

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  ✅ All data generated successfully in {elapsed:.1f} seconds!")
    print("=" * 70)

    # Print summary of generated files
    print("\n📊 Generated files:")
    for f in sorted(config.DATA_DIR.glob("*.csv")):
        import pandas as pd
        df = pd.read_csv(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"   • {f.name}: {len(df):,} rows, {len(df.columns)} columns ({size_mb:.1f} MB)")


if __name__ == "__main__":
    generate_all()
