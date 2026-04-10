"""
PlantIQ AI Brain - Vision Training Pipeline
Usage: python training/train_vision.py --data_dir ./data/diseases --epochs 25
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pathlib import Path
import os
import sys
import argparse
import time

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.disease_vision.cnn_model import PlantDiseaseCNN
import config

def train_model(data_dir, num_epochs=25, batch_size=32, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting training on device: {device}")

    # 1. Data Augmentation & Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load Datasets
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory {data_dir} not found.")
        print("Please ensure you have subfolders named after the disease classes.")
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"📊 Classes detected: {class_names}")

    # 3. Model Initialization
    model = PlantDiseaseCNN(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Fine-tune the whole model or just the head?
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Training Loop
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'🏆 Best val Acc: {best_acc:4f}')

    # 5. Save the weights
    save_dir = config.MODELS_DIR / "disease_vision"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / "cnn_weights.pth"
    
    torch.save(best_model_wts, save_path)
    print(f"✅ Trained model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlantIQ Vision Trainer")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing train/val folders")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_model(args.data_dir, args.epochs, args.batch_size, args.lr)
