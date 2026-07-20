"""
fine_tune_domain_adaptation.py
==============================
Few-Shot Domain Adaptation using REAL noisy spectrograms.
Uses the actual vibration data with added noise (from generate_noisy_spectrograms.py)
instead of AI-generated placeholder images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def fine_tune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flush_print(f"Fine-tuning on: {device}")

    # Load the best trained model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    
    weights_path = os.path.join(BASE_DIR, '..', 'models', 'gear_bearing_resnet50_weights.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    flush_print(f"Loaded base model from: {weights_path}")

    # Freeze early layers — only train layer4 + FC
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flush_print(f"Trainable parameters: {trainable:,}")

    # Data augmentation for noisy samples
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the REAL noisy spectrograms (organized in class subfolders)
    noisy_dir = os.path.join(BASE_DIR, '..', 'data', 'real_world_samples')
    
    # Check for subfolder structure (class_name/images)
    subdirs = [d for d in os.listdir(noisy_dir) 
               if os.path.isdir(os.path.join(noisy_dir, d))]
    
    if len(subdirs) >= 6:
        flush_print(f"Loading noisy dataset from: {noisy_dir}")
        noisy_dataset = datasets.ImageFolder(noisy_dir, train_transform)
        flush_print(f"  Classes: {noisy_dataset.classes}")
        flush_print(f"  Samples: {len(noisy_dataset)}")
        
        noisy_loader = DataLoader(noisy_dataset, batch_size=8, shuffle=True, num_workers=0)
    else:
        flush_print("[ERROR] No class subfolders found. Run generate_noisy_spectrograms.py first.")
        return

    # Training
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005
    )
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    model.train()
    flush_print(f"\nStarting domain adaptation ({epochs} epochs)...")
    flush_print()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in noisy_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        flush_print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f} ({correct}/{total})")

    # Save the domain-adapted model
    models_dir = os.path.join(BASE_DIR, '..', 'models')
    save_path = os.path.join(models_dir, 'gear_bearing_resnet50_domain_adapted.pth')
    torch.save(model.state_dict(), save_path)
    flush_print(f"\nSaved adapted model to {save_path}")

    # Root copy
    root_path = os.path.join(BASE_DIR, '..', 'gear_bearing_resnet50_domain_adapted.pth')
    torch.save(model.state_dict(), root_path)
    
    flush_print("Domain adaptation complete!")


if __name__ == '__main__':
    fine_tune()
