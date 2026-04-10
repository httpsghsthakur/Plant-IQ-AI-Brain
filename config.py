"""
PlantIQ AI Brain - Global Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "generated"
MODELS_DIR = BASE_DIR / "trained_models"
GENERATORS_DIR = BASE_DIR / "data" / "generators"

# ─── Supabase Configuration ──────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# ─── Nursery Configuration ──────────────────────────────────────────────
NURSERY_NAME = "Green Valley Walnut Nursery"
NURSERY_LOCATION = "Kashmir, India"
CURRENCY = "₹"

# ─── Data Generation Parameters ─────────────────────────────────────────
NUM_WORKERS = 15
NUM_PLANTS = 5000
NUM_ZONES = 6
NUM_SENSORS = 20
HISTORY_DAYS = 365  # 1 year of data

ZONES = [
    {"id": "ZONE-A", "name": "Greenhouse A", "type": "greenhouse", "capacity": 1000},
    {"id": "ZONE-B", "name": "Greenhouse B", "type": "greenhouse", "capacity": 1000},
    {"id": "ZONE-C", "name": "Open Field C", "type": "open_field", "capacity": 1200},
    {"id": "ZONE-D", "name": "Open Field D", "type": "open_field", "capacity": 1000},
    {"id": "ZONE-E", "name": "Nursery Beds E", "type": "nursery_bed", "capacity": 500},
    {"id": "ZONE-F", "name": "Hardening Zone F", "type": "hardening", "capacity": 300},
]

PLANT_VARIETIES = [
    {"name": "Kashmiri Walnut", "code": "KW", "price_range": (2500, 3500), "growth_rate": 0.3},
    {"name": "Chandler", "code": "CH", "price_range": (2000, 3000), "growth_rate": 0.35},
    {"name": "Franquette", "code": "FR", "price_range": (1800, 2800), "growth_rate": 0.28},
    {"name": "Local Variety", "code": "LV", "price_range": (1200, 2000), "growth_rate": 0.25},
    {"name": "Pusa Kanchan", "code": "PK", "price_range": (2200, 3200), "growth_rate": 0.32},
]

GRAFT_METHODS = ["whip_and_tongue", "bud_grafting", "cleft_grafting", "bark_grafting", "side_veneer"]

# ─── Optimal Environmental Ranges ───────────────────────────────────────
OPTIMAL_CONDITIONS = {
    "temperature": {"min": 18, "max": 28, "critical_low": 5, "critical_high": 38},
    "humidity": {"min": 55, "max": 75, "critical_low": 30, "critical_high": 90},
    "soil_moisture": {"min": 40, "max": 60, "critical_low": 25, "critical_high": 80},
    "light_intensity": {"min": 15000, "max": 45000, "unit": "lux"},
    "soil_ph": {"min": 6.0, "max": 7.5, "optimal": 6.8},
}

# ─── Worker Scoring Weights ─────────────────────────────────────────────
WORKER_SCORE_WEIGHTS = {
    "attendance": 0.25,
    "productivity": 0.30,
    "quality": 0.30,
    "initiative": 0.15,
}

# ─── Disease Risk Factors ───────────────────────────────────────────────
DISEASES = [
    {"name": "Walnut Blight", "risk_temp": (18, 24), "risk_humidity": 70, "severity": "high"},
    {"name": "Crown Rot", "risk_temp": (15, 25), "risk_humidity": 80, "severity": "high"},
    {"name": "Anthracnose", "risk_temp": (20, 28), "risk_humidity": 75, "severity": "medium"},
    {"name": "Powdery Mildew", "risk_temp": (15, 30), "risk_humidity": 60, "severity": "medium"},
    {"name": "Root Rot", "risk_temp": (10, 30), "risk_humidity": 85, "severity": "high"},
]

# ─── Financial Defaults ─────────────────────────────────────────────────
MONTHLY_LABOR_COST = 125000  # ₹
MONTHLY_MATERIAL_COST = 85000
MONTHLY_UTILITY_COST = 18000
AVG_PLANT_PRICE = 2500
TARGET_PROFIT_MARGIN = 0.35

# ─── Model Training Parameters ──────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 8

# ─── API Configuration ──────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("PORT", 8000))
API_VERSION = "v1"
API_TITLE = "PlantIQ AI Brain"
API_DESCRIPTION = "Intelligent Nursery Management System - AI/ML Backend"
