"""
PlantIQ AI Brain - Lite Vision Data Generator
Generates synthetic placeholder images for the disease vision model.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import random
import sys

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

OUTPUT_DIR = config.DATA_DIR / "diseases"

def create_placeholder_image(path, label_name):
    """Creates a synthetic image with a specific label."""
    # Create a 224x224 RGB image
    img = Image.new('RGB', (224, 224), color=(
        random.randint(0, 100), 
        random.randint(50, 200), 
        random.randint(0, 100)
    ))
    
    draw = ImageDraw.Draw(img)
    
    # Draw some random 'leaf' shapes (ovals)
    for _ in range(5):
        x0, y0 = random.randint(20, 100), random.randint(20, 100)
        x1, y1 = x0 + random.randint(50, 100), y0 + random.randint(50, 100)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse([x0, y0, x1, y1], fill=color, outline=(0, 0, 0))

    # Add text (optional, but helps for visual inspection)
    # Using default font
    draw.text((10, 180), f"Class: {label_name}", fill=(255, 255, 255))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def generate_lite_vision_data():
    """Generates a small synthetic dataset for development/testing."""
    print("🎨 Generating synthetic plant disease images...")
    
    classes = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Grape___Black_rot",
        "Tomato___Bacterial_spot",
        "Tomato___healthy"
    ]
    
    splits = {"train": 50, "val": 10}
    
    total_images = 0
    for split, count in splits.items():
        for cls in classes:
            target_dir = OUTPUT_DIR / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                file_name = f"synth_{i}.jpg"
                create_placeholder_image(target_dir / file_name, cls)
                total_images += 1
                
    print(f"✅ Generated {total_images} synthetic images in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_lite_vision_data()
