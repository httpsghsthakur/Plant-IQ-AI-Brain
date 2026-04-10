import torch
import sys
from pathlib import Path
import os

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the model
from models.disease_vision.cnn_model import PlantDiseaseCNN
import config

def generate_stub_weights():
    """
    Instantiates the CNN and saves the randomly initialized weights to disk.
    This allows the FastAPI backend to boot and run inference (outputting random predictions)
    until you replace this file with a truly trained .pth file from Google Colab.
    """
    print("Initializing dummy PyTorch CNN...")
    model = PlantDiseaseCNN(num_classes=5)
    
    # Create directory if it doesn't exist
    save_dir = config.MODELS_DIR / "disease_vision"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = save_dir / "cnn_weights.pth"
    print(f"Saving structural weights to: {save_path}")
    
    # Save the state dict
    torch.save(model.state_dict(), save_path)
    print("[OK] Successfully generated dummy CNN weights.")

if __name__ == "__main__":
    generate_stub_weights()
