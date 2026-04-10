"""
PlantIQ AI Brain - IoT Sensor Data Generator
Generates realistic sensor readings with daily patterns, seasonal variation, and noise.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def generate_sensor_data(days: int = None, output_path: str = None) -> pd.DataFrame:
    """Generate realistic IoT sensor data across all zones."""
    days = days or config.HISTORY_DAYS
    output_path = output_path or str(config.DATA_DIR / "sensor_data.csv")

    np.random.seed(config.RANDOM_STATE)
    records = []

    start_date = datetime(2025, 4, 10)

    for zone in config.ZONES:
        zone_id = zone["id"]
        zone_type = zone["type"]

        # Zone-specific base conditions
        if zone_type == "greenhouse":
            base_temp = 24
            temp_variation = 4
            base_humidity = 65
            humidity_variation = 10
        elif zone_type == "open_field":
            base_temp = 22
            temp_variation = 10
            base_humidity = 55
            humidity_variation = 20
        elif zone_type == "nursery_bed":
            base_temp = 23
            temp_variation = 6
            base_humidity = 60
            humidity_variation = 15
        else:  # hardening
            base_temp = 21
            temp_variation = 8
            base_humidity = 50
            humidity_variation = 18

        for day in range(days):
            current_date = start_date + timedelta(days=day)
            day_of_year = current_date.timetuple().tm_yday

            # Seasonal adjustment (Kashmir climate)
            seasonal_temp = 8 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
            seasonal_humidity = -10 * np.sin(2 * np.pi * (day_of_year - 180) / 365)

            # Generate 24 hourly readings per day
            for hour in range(24):
                # Diurnal temperature pattern
                diurnal_temp = -4 * np.cos(2 * np.pi * (hour - 14) / 24)

                temperature = (
                    base_temp + seasonal_temp + diurnal_temp
                    + np.random.normal(0, 1.5)
                )

                # Humidity inversely related to temperature
                humidity = (
                    base_humidity + seasonal_humidity
                    - 0.8 * diurnal_temp
                    + np.random.normal(0, 3)
                )
                humidity = np.clip(humidity, 20, 98)

                # Soil moisture - decreases during day, increases after irrigation
                soil_moisture = (
                    50 + 5 * np.sin(2 * np.pi * (hour - 6) / 24)
                    + seasonal_humidity * 0.3
                    + np.random.normal(0, 4)
                )
                soil_moisture = np.clip(soil_moisture, 15, 85)

                # Light intensity - bell curve during day
                if 6 <= hour <= 18:
                    light = (
                        40000 * np.sin(np.pi * (hour - 6) / 12)
                        * (0.7 if zone_type == "greenhouse" else 1.0)
                        + np.random.normal(0, 3000)
                    )
                    # Cloud cover effect (random days)
                    if np.random.random() < 0.3:
                        light *= np.random.uniform(0.3, 0.7)
                else:
                    light = np.random.uniform(0, 50)
                light = max(0, light)

                # Soil pH - relatively stable with slight variation
                soil_ph = 6.8 + np.random.normal(0, 0.2) + 0.1 * np.sin(2 * np.pi * day / 30)
                soil_ph = np.clip(soil_ph, 5.5, 8.5)

                # Rainfall (mm/hour) - more in monsoon (Jul-Sep)
                rain_probability = 0.05
                if 6 <= (current_date.month) <= 9:  # monsoon
                    rain_probability = 0.25
                rainfall = 0
                if np.random.random() < rain_probability:
                    rainfall = np.random.exponential(3)

                # Wind speed
                wind_speed = max(0, np.random.gamma(2, 3) + 2 * np.sin(2 * np.pi * hour / 24))

                records.append({
                    "timestamp": current_date.replace(hour=hour),
                    "zone_id": zone_id,
                    "zone_name": zone["name"],
                    "zone_type": zone_type,
                    "temperature": round(temperature, 1),
                    "humidity": round(humidity, 1),
                    "soil_moisture": round(soil_moisture, 1),
                    "light_intensity": round(light, 0),
                    "soil_ph": round(soil_ph, 2),
                    "rainfall_mm": round(rainfall, 1),
                    "wind_speed_kmh": round(wind_speed, 1),
                })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"✅ Sensor data generated: {len(df)} records → {output_path}")
    return df


if __name__ == "__main__":
    generate_sensor_data()
