"""PlantIQ AI Brain - API Route: Health Check"""
from fastapi import APIRouter
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

router = APIRouter(tags=["Health"])


@router.get("/api/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "service": config.API_TITLE,
        "version": config.API_VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/api/ai/models/status")
async def model_status():
    """Check training status of all AI models."""
    from services.model_service import model_service
    return model_service.get_status()
