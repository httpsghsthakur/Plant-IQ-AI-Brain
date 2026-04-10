"""
PlantIQ AI Brain - API Route: Advisor Chat
"""
from fastapi import APIRouter, Query, Body, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from services.chat_service import chat_service

router = APIRouter(prefix="/api/ai", tags=["Advisor Chat"])

class ChatRequest(BaseModel):
    query: str
    nursery_id: Optional[str] = None

@router.post("/chat")
async def chat_with_advisor(
    request: ChatRequest,
    nursery_id: Optional[str] = Query(None, description="Fallback nursery_id from URL params"),
) -> Dict[str, Any]:
    """
    Expert Advisor Chatbot.
    Ask questions about plant health, water, workers, or system status in plain language.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    resolved_nursery_id = request.nursery_id or nursery_id or "default"
    
    response = chat_service.process_query(request.query, resolved_nursery_id)
    return {
        "status": "success",
        "query": request.query,
        "nursery_id": resolved_nursery_id,
        "response": response
    }

