"""
PlantIQ AI Brain - FastAPI Entry Point
Intelligent Nursery Management System - AI/ML Backend
"""
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
import config
from api.middleware import setup_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plantiq")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # ─── Startup ────────────────────────────────────────────────────
    logger.info("🌱 PlantIQ AI Brain starting up...")

    # Data loading now happens dynamically per nursery via DataService caching
    logger.info("📂 DataService configured for dynamic Supabase fetch")

    # Load trained models
    from services.model_service import model_service
    load_results = model_service.load_all()
    loaded = sum(1 for v in load_results.values() if v)
    if loaded > 0:
        logger.info(f"🤖 Models loaded: {loaded}/8 trained models ready")
    else:
        logger.warning("⚠️  No trained models found. Run 'python training/train_all_models.py' first.")

    logger.info(f"✅ PlantIQ AI Brain ready at http://{config.API_HOST}:{config.API_PORT}")
    logger.info(f"📖 API docs at http://localhost:{config.API_PORT}/docs")

    yield

    # ─── Shutdown ───────────────────────────────────────────────────
    logger.info("🛑 PlantIQ AI Brain shutting down...")


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Setup middleware
setup_middleware(app)

# Register routes
from api.routes.health import router as health_router
from api.routes.dashboard import router as dashboard_router
from api.routes.environmental import router as env_router
from api.routes.worker import router as worker_router
from api.routes.plant_health import router as plant_router
from api.routes.graft import router as graft_router
from api.routes.resources import router as resource_router
from api.routes.yield_forecast import router as yield_router
from api.routes.financial import router as financial_router
from api.routes.anomaly import router as anomaly_router
from api.routes.recommendations import router as rec_router
from api.routes.disease_vision import router as vision_router
from api.routes.chat import router as chat_router

app.include_router(health_router)
app.include_router(dashboard_router)
app.include_router(env_router)
app.include_router(worker_router)
app.include_router(plant_router)
app.include_router(graft_router)
app.include_router(resource_router)
app.include_router(yield_router)
app.include_router(financial_router)
app.include_router(anomaly_router)
app.include_router(rec_router)
app.include_router(vision_router)
app.include_router(chat_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API overview."""
    return {
        "service": config.API_TITLE,
        "description": config.API_DESCRIPTION,
        "version": config.API_VERSION,
        "nursery": config.NURSERY_NAME,
        "location": config.NURSERY_LOCATION,
        "docs": f"http://localhost:{config.API_PORT}/docs",
        "endpoints": {
            "health": "/api/health",
            "models_status": "/api/ai/models/status",
            "daily_report": "/api/ai/reports/daily",
            "dashboard": "/api/ai/dashboard",
            "environmental": "/api/ai/analyze/environment",
            "irrigation": "/api/ai/optimize/irrigation",
            "worker_performance": "/api/ai/analyze/workers",
            "plant_health": "/api/ai/analyze/plant/{plant_id}",
            "disease_risk": "/api/ai/predict/disease-risk/{zone_id}",
            "graft_prediction": "/api/ai/predict/graft-success",
            "resource_optimization": "/api/ai/optimize/water",
            "yield_forecast": "/api/ai/forecast/production",
            "financial_analysis": "/api/ai/analyze/financials",
            "anomaly_detection": "/api/ai/anomalies/report",
            "pricing": "/api/ai/recommend/pricing",
            "disease_vision_cnn": "/api/ai/vision/disease",
            "chat_advisor": "/api/ai/chat",
            "worker_burnout_v2": "/api/ai/predict/worker-burnout",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )
