"""
PlantIQ AI Brain - API Middleware
CORS, logging, error handling.
"""
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

logger = logging.getLogger("plantiq")


def setup_middleware(app):
    """Configure all middleware for the FastAPI app."""

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.3f}s)")
        return response

    # Global error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": str(request.url.path),
            },
        )
