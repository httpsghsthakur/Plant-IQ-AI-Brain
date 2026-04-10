from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from typing import Dict, Any, Optional
from models.disease_vision.inference import vision_service

router = APIRouter(prefix="/api/ai/vision", tags=["Computer Vision"])

@router.post("/disease")
async def analyze_plant_disease_image(
    file: UploadFile = File(...),
    nursery_id: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """
    Accepts an Image Upload (PNG/JPG) of a plant leaf and uses a PyTorch Convolutional Neural Network (CNN)
    to classify whether the leaf is Healthy or has a disease (e.g. Walnut Blight, Crown Rot).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read the raw byte stream from the uploaded file
        file_bytes = await file.read()
        
        # Pass to the PyTorch inference engine
        result = vision_service.predict(file_bytes)
        
        return {
            "status": "success",
            "message": "Image analyzed successfully using CNN.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CNN Processing Error: {str(e)}")
