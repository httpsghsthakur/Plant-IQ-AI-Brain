"""
PlantIQ AI Brain - Dataset Importer
Downloads and organizes the open-source PlantVillage dataset for training.
"""
import os
import sys
import zipfile
import shutil
import httpx
from pathlib import Path
import subprocess

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

DATASET_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"
EXTRACT_PATH = config.DATA_DIR / "plantvillage_extracted"
FINAL_DATA_DIR = config.DATA_DIR / "diseases"

def download_dataset():
    """
    Skipping external download as requested.
    Checks for local plantvillage_raw.zip in data/ or triggers synthetic generation.
    """
    zip_path = Path(__file__).parent.parent / "data" / "plantvillage_raw.zip"
    
    if zip_path.exists():
        print(f"📦 Found local dataset at {zip_path}. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("✅ Extraction complete.")
    else:
        print("⚠️  Local dataset ZIP not found. Switching to 'Lite' (synthetic) image generation...")
        from data.generators.vision_lite_generator import generate_lite_vision_data
        generate_lite_vision_data()

def organize_dataset():
    """Organizes images into train/val folders."""
    if not os.path.exists(EXTRACT_PATH):
        # If EXTRACT_PATH doesn't exist, we assume Lite generation already filled FINAL_DATA_DIR
        if FINAL_DATA_DIR.exists():
            print(f"✅ FINAL_DATA_DIR already exists at {FINAL_DATA_DIR}. Skipping organization.")
            return
        else:
            print("❌ Error: No source images found and Lite generation failed.")
            return

    print(f"Organizing images from {EXTRACT_PATH}...")
    color_dir = None
    for root, dirs, files in os.walk(EXTRACT_PATH):
        if "color" in dirs:
            color_dir = Path(root) / "color"
            break
            
    if not color_dir:
        print("Error: Could not find 'color' directory in the extracted dataset.")
        return

    print(f"Found source images at: {color_dir}")
    
    # Organize into train/val (80/20 split)
    categories = [d for d in color_dir.iterdir() if d.is_dir()]
    print(f"Processing {len(categories)} plant disease categories...")

    for category in categories:
        cat_name = category.name
        # Sanitize name
        cat_name = cat_name.replace("___", "_").replace(",", "").replace(" ", "_")
        
        train_dst = FINAL_DATA_DIR / "train" / cat_name
        val_dst = FINAL_DATA_DIR / "val" / cat_name
        
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)
        
        images = list(category.glob("*.*"))
        split_idx = int(len(images) * 0.8)
        
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Copy files (limiting to 200 per class to keep local training manageable)
        for img in train_imgs[:200]:
            shutil.copy(img, train_dst / img.name)
        for img in val_imgs[:50]:
            shutil.copy(img, val_dst / img.name)
            
        print(f"   - {cat_name}: {len(train_imgs[:200])} train, {len(val_imgs[:50])} val")

    print(f"Dataset organized in {FINAL_DATA_DIR}")

if __name__ == "__main__":
    try:
        if not os.path.exists(EXTRACT_PATH):
            download_dataset()
        organize_dataset()
    except Exception as e:
        print(f"Failed to import dataset: {e}")
