import io
import os
from pathlib import Path
import config

try:
    import torch
    from torchvision import transforms
    from .cnn_model import PlantDiseaseCNN
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PyTorch not fully available locally (DLL Error): {e}. CNN running in MOCK mode.")
    TORCH_AVAILABLE = False
    
from PIL import Image

# Match the 5 classes output by our CNN
DISEASE_CLASSES = [
    "Healthy",
    "Walnut Blight",
    "Crown Rot",
    "Anthracnose",
    "Powdery Mildew"
]

DISEASE_TREATMENTS = {
    "Healthy": "No treatment required. Maintain current environmental monitoring and soil nutrition balance.",
    "Walnut Blight": "Apply copper-based fungicides immediately. Prune backend infected branches and ensure low humidity in the canopy via spacing.",
    "Crown Rot": "Urgent: Reduce irrigation. Improve soil drainage and avoid pooling water around the plant base. Consider localized fungicide drench.",
    "Anthracnose": "Remove and destroy infected foliage. Improve air circulation and apply organic neem oil or potassium bicarbonate.",
    "Powdery Mildew": "Apply systematic fungicide or a mixture of baking soda and water. Increase sunlight exposure and maintain dry leaves."
}

class DiseaseVisionService:
    def __init__(self):
        self.model_loaded = False
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cpu')  # Force CPU to minimize deployment size
            self.model = PlantDiseaseCNN(num_classes=len(DISEASE_CLASSES))
            # Image transformations: Resize to 224x224 and normalize
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.device = None
            self.model = None
            self.transform = None
        
    def load_model(self):
        if self.model_loaded or not TORCH_AVAILABLE:
            return True
        
        weights_path = config.MODELS_DIR / "disease_vision" / "cnn_weights.pth"
        if not weights_path.exists():
            print(f"⚠️ CNN weights not found at {weights_path}. Model will use random initialization.")
        else:
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"✅ CNN Disease Model loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to load CNN weights.")
                
        self.model.eval()
        self.model_loaded = True
        return True

    def process_image(self, file_bytes: bytes):
        """Preprocess the raw image bytes into a PyTorch Tensor."""
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            return tensor
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")

    def predict(self, file_bytes: bytes):
        """Runs the CNN forward pass and returns human-readable predictions."""
        if not TORCH_AVAILABLE:
            # Return high-confidence mock data if running locally without C++ binaries
            return {
                "primary_diagnosis": "Walnut Blight",
                "confidence_score": 0.88,
                "treatment_recommendation": DISEASE_TREATMENTS["Walnut Blight"],
                "all_probabilities": [
                    {"disease": "Walnut Blight", "probability": 0.88},
                    {"disease": "Healthy", "probability": 0.05},
                    {"disease": "Crown Rot", "probability": 0.04},
                    {"disease": "Anthracnose", "probability": 0.02},
                    {"disease": "Powdery Mildew", "probability": 0.01}
                ],
                "note": "MOCK MODE (PyTorch DLL missing locally. Works instantly in Cloud Docker)."
            }

        self.load_model()
        
        # 1. Preprocess
        input_tensor = self.process_image(file_bytes)
        
        # 2. Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # 3. Format results as a sorted list of dictionaries
        results = [
            {"disease": DISEASE_CLASSES[i], "probability": float(probabilities[i])}
            for i in range(len(DISEASE_CLASSES))
        ]
        
        results.sort(key=lambda x: x["probability"], reverse=True)
        primary_diagnosis = results[0]
        
        return {
            "primary_diagnosis": primary_diagnosis["disease"],
            "confidence_score": primary_diagnosis["probability"],
            "treatment_recommendation": DISEASE_TREATMENTS.get(primary_diagnosis["disease"], "No specific treatment available."),
            "all_probabilities": results
        }

# Singleton Service Instance
vision_service = DiseaseVisionService()
