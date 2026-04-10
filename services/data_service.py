"""
PlantIQ AI Brain - Data Service
Fetch and transform Supabase data into Pandas DataFrames for ML models.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from services.supabase_client import get_supabase


class DataService:
    """Service to fetch real-time data from Supabase for a specific nursery."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self.CACHE_TTL_MINUTES = 5

    def _should_refresh(self, nursery_id: str) -> bool:
        if nursery_id not in self._cache_timestamp:
            return True
        elapsed = datetime.now() - self._cache_timestamp[nursery_id]
        return elapsed > timedelta(minutes=self.CACHE_TTL_MINUTES)

    def load_nursery_data(self, nursery_id: str, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch all operational data from Supabase for a specific nursery and convert to DataFrames."""
        if not force_refresh and not self._should_refresh(nursery_id):
            return self._cache[nursery_id]

        supabase = get_supabase()
        db_data = {}

        try:
            # 1. Fetch Plants (joined with varieties implicitly or we fetch separately and merge)
            plants_res = supabase.table("plants").select("*, plant_varieties(name), zones(name)").eq("nursery_id", nursery_id).execute()
            plants_raw = plants_res.data
            
            # 2. Fetch IoT Readings
            iot_res = supabase.table("iot_readings").select("*, iot_sensors(zone_id)").eq("nursery_id", nursery_id).order("timestamp", desc=True).limit(5000).execute()
            iot_raw = iot_res.data

            # 3. Fetch Diseases
            disease_res = supabase.table("disease_detections").select("*, plants(zone_id)").execute() # Requires complex join/filter but we get what we can
            # Filter manually to only this nursery's plants if needed
            plant_ids = [p["id"] for p in plants_raw]
            disease_raw = [d for d in disease_res.data if d.get("plant_id") in plant_ids]

            # 4. Fetch Grafts
            graft_res = supabase.table("plant_grafts").select("*").eq("nursery_id", nursery_id).execute()
            graft_raw = graft_res.data

            # 5. Fetch Attendance & Workers
            att_res = supabase.table("attendance").select("*, workers(full_name)").eq("nursery_id", nursery_id).execute()
            att_raw = att_res.data

            # 6. Fetch Tasks
            task_res = supabase.table("tasks").select("*").eq("nursery_id", nursery_id).execute()
            task_raw = task_res.data

            # 7. Fetch Orders (Sales)
            sales_res = supabase.table("orders").select("*, order_items(variety_name, quantity, unit_price)").eq("nursery_id", nursery_id).execute()
            sales_raw = sales_res.data

            # 8. Fetch Inventory
            inv_res = supabase.table("inventory").select("*").eq("nursery_id", nursery_id).execute()
            inv_raw = inv_res.data

            # ─── DATA TRANSFORMATIONS ────────────────────────────────────
            
            # Plant Inventory -> expected: plant_id, variety_name, zone_id, stage, health_status, age_days, current_height_cm, height_deviation_pct, is_grafted, graft_success
            plant_df = pd.DataFrame()
            if plants_raw:
                p_df = pd.DataFrame(plants_raw)
                p_df["plant_id"] = p_df["id"]
                p_df["variety_name"] = p_df["plant_varieties"].apply(lambda x: x.get("name") if isinstance(x, dict) else "Unknown")
                p_df["zone_id"] = p_df["zone_id"]
                p_df["stage"] = p_df["stage"]
                p_df["health_status"] = p_df["current_health"].fillna("healthy")
                
                # Approximate age if not present
                p_df["planted_date"] = pd.to_datetime(p_df["planted_date"])
                p_df["age_days"] = p_df.get("age_days", (datetime.now() - p_df["planted_date"]).dt.days)
                p_df["current_height_cm"] = p_df["current_height_cm"].fillna(0)
                p_df["height_deviation_pct"] = 0  # Add generic default if not computed
                p_df["is_grafted"] = p_df["parent_plant_id"].notna()
                p_df["graft_success"] = True # Approximation
                plant_df = p_df

            # Growth / Plant Events
            # For simplicity, we create a mock growth dataframe based on plants
            growth_df = pd.DataFrame()
            if not plant_df.empty:
                growth_records = []
                for _, row in plant_df.iterrows():
                    growth_records.append({
                        "plant_id": row["plant_id"],
                        "measurement_date": datetime.now().isoformat(),
                        "weekly_growth_cm": row["growth_rate"] if "growth_rate" in row else 2.0,
                        "health_score": 90 if row["health_status"] == "healthy" else 50,
                        "stress_indicators": "none",
                        "disease_detected": pd.NA
                    })
                growth_df = pd.DataFrame(growth_records)

            # IoT -> Sensor DataFrame: timestamp, zone_id, temperature, humidity, soil_moisture, light_intensity, soil_ph, rainfall_mm
            sensor_df = pd.DataFrame()
            if iot_raw:
                s_df = pd.DataFrame(iot_raw)
                s_df["timestamp"] = pd.to_datetime(s_df["timestamp"])
                s_df["zone_id"] = s_df["iot_sensors"].apply(lambda x: x.get("zone_id") if isinstance(x, dict) else "")
                
                # Pivot type/value into columns
                pivot = s_df.pivot_table(index=["timestamp", "zone_id"], columns="type", values="value").reset_index()
                # Rename columns if needed
                if "temperature" not in pivot.columns: pivot["temperature"] = 25.0
                if "humidity" not in pivot.columns: pivot["humidity"] = 60.0
                if "soil_moisture" not in pivot.columns: pivot["soil_moisture"] = 50.0
                if "light_intensity" not in pivot.columns: pivot["light_intensity"] = 20000.0
                if "ph" in pivot.columns: pivot["soil_ph"] = pivot["ph"]
                else: pivot["soil_ph"] = 6.8
                pivot["rainfall_mm"] = 0.0 # Approximation
                
                sensor_df = pivot

            # Diseases
            disease_df = pd.DataFrame()
            if disease_raw:
                d_df = pd.DataFrame(disease_raw)
                d_df["plant_id"] = d_df["plant_id"]
                d_df["severity"] = d_df["severity"]
                d_df["treatment_applied"] = d_df["treatment_notes"].notna()
                d_df["treatment_success"] = d_df["status"] == "cured"
                d_df["zone_id"] = d_df["plants"].apply(lambda x: x.get("zone_id") if isinstance(x, dict) else "")
                disease_df = d_df

            # Grafting
            graft_df = pd.DataFrame()
            if graft_raw:
                g_df = pd.DataFrame(graft_raw)
                g_df["graft_id"] = g_df["id"]
                g_df["worker_id"] = g_df.get("worker_id", "Unknown") # Mapped differently in your action schema, using approximation
                g_df["worker_name"] = "Worker"
                g_df["method"] = g_df["graft_method"]
                g_df["success"] = g_df["union_status"] == "successful"
                g_df["callus_formation_pct"] = 80 if "successful" else 30
                g_df["cambium_alignment_pct"] = 80 if "successful" else 30
                graft_df = g_df

            # Attendance
            att_df = pd.DataFrame()
            if att_raw:
                a_df = pd.DataFrame(att_raw)
                a_df["worker_id"] = a_df["worker_id"]
                a_df["worker_name"] = a_df["workers"].apply(lambda x: x.get("full_name") if isinstance(x, dict) else "Unknown")
                a_df["date"] = pd.to_datetime(a_df["date"])
                a_df["work_hours"] = a_df["work_hours"].fillna(0).astype(float)
                a_df["overtime_hours"] = a_df["work_hours"].apply(lambda x: max(0, x - 8))
                a_df["status"] = a_df["status"]
                att_df = a_df

            # Tasks
            task_df = pd.DataFrame()
            if task_raw:
                t_df = pd.DataFrame(task_raw)
                t_df["task_id"] = t_df["id"]
                t_df["worker_id"] = t_df["assigned_to"]
                t_df["status"] = t_df["status"]
                t_df["quality_score"] = 90 # Approximation since quality_score isn't explicitly in schema
                t_df["completed_at"] = pd.to_datetime(t_df["completed_at"])
                task_df = t_df

            # Sales (Orders)
            sales_df = pd.DataFrame()
            if sales_raw:
                s_records = []
                for order in sales_raw:
                    items = order.get("order_items", [])
                    for item in items:
                        s_records.append({
                            "order_id": order["id"],
                            "date": order["created_at"],
                            "variety_name": item.get("variety_name"),
                            "quantity": item.get("quantity", 0),
                            "unit_price": float(item.get("unit_price") or 0),
                            "total_amount": float(item.get("unit_price") or 0) * int(item.get("quantity") or 0),
                            "amount_paid": order.get("total_amount") if order.get("payment_status") == "paid" else 0
                        })
                sales_df = pd.DataFrame(s_records)

            # Inventory
            inventory_df = pd.DataFrame()
            if inv_raw:
                i_df = pd.DataFrame(inv_raw)
                i_df["item_name"] = i_df["name"]
                i_df["closing_stock"] = i_df["quantity"].astype(float)
                i_df["date"] = pd.to_datetime(i_df["last_updated"])
                i_df["consumed"] = 5.0 # Daily usage proxy
                i_df["below_reorder"] = i_df["closing_stock"] <= i_df["min_threshold"].astype(float)
                i_df["unit_cost"] = 50.0  # Proxy if no cost in schema
                inventory_df = i_df

            # Expense
            # Assuming expenses don't exist dedicatedly in schema yet -> Empty df to prevent breaking
            expense_df = pd.DataFrame(columns=["date", "category", "amount", "description"])

            db_data = {
                "plant_inventory": plant_df,
                "growth": growth_df,
                "sensor": sensor_df,
                "disease": disease_df,
                "graft": graft_df,
                "attendance": att_df,
                "task": task_df,
                "sales": sales_df,
                "inventory": inventory_df,
                "expense": expense_df,
            }

            self._cache[nursery_id] = db_data
            self._cache_timestamp[nursery_id] = datetime.now()
            print(f"[V] Fetched live data from Supabase for nursery: {nursery_id}")

            return db_data

        except Exception as e:
            print(f"[!] Error fetching from Supabase for nursery {nursery_id}: {e}")
            raise Exception(f"Database error: {e}")


# Singleton instance
data_service = DataService()
