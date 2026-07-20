"""
train_model.py
==============
Trains a ResNet-50 classifier on the spectrogram images using Transfer Learning.
Optimized for CPU training: freezes all layers except FC, uses efficient batching.

Usage:
    python train_model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import os
import sys
import time
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_WORKERS = 0


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flush_print(f"Training device: {device}")
    flush_print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        flush_print(f"GPU: {torch.cuda.get_device_name(0)}")
    flush_print()

    # ── Data Transforms ──────────────────────────────────────────────────
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # ── Data Loading ─────────────────────────────────────────────────────
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ['train', 'validation']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                       shuffle=(x == 'train'), num_workers=NUM_WORKERS)
        for x in ['train', 'validation']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes

    flush_print(f"Classes: {class_names}")
    flush_print(f"Training samples:   {dataset_sizes['train']}")
    flush_print(f"Validation samples: {dataset_sizes['validation']}")
    flush_print()

    # ── Model Architecture ───────────────────────────────────────────────
    flush_print("Loading ResNet-50 with ImageNet pre-trained weights...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # PHASE 1: Freeze all layers, only train new FC layer (feature extraction)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    flush_print(f"Total parameters: {total:,}")
    flush_print(f"Trainable parameters (FC only): {trainable:,}")
    flush_print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # ── Training Loop ────────────────────────────────────────────────────
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    start_time = time.time()

    flush_print("=" * 60)
    flush_print(f"PHASE 1: Feature Extraction (FC only) — {NUM_EPOCHS} epochs")
    flush_print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        flush_print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        flush_print("-" * 40)

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            batch_count = 0

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
                batch_count += 1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            elapsed_epoch = time.time() - epoch_start
            flush_print(f"  {phase:10s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  ({elapsed_epoch:.0f}s)")

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_path = os.path.join(MODEL_DIR, 'gear_bearing_resnet50_weights.pth')
                torch.save(model.state_dict(), best_path)
                flush_print(f"  >> Saved best model (acc: {best_acc:.4f})")

    # ── PHASE 2: Fine-tune layer4 + FC ───────────────────────────────────
    flush_print()
    flush_print("=" * 60)
    flush_print("PHASE 2: Fine-tuning (layer4 + FC) — 5 epochs")
    flush_print("=" * 60)

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flush_print(f"Trainable parameters (layer4+FC): {trainable:,}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    for epoch in range(5):
        epoch_start = time.time()
        flush_print(f"\nEpoch {epoch+1}/5")
        flush_print("-" * 40)

        for phase in ['train', 'validation']:
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            elapsed_epoch = time.time() - epoch_start
            flush_print(f"  {phase:10s}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}  ({elapsed_epoch:.0f}s)")

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_path = os.path.join(MODEL_DIR, 'gear_bearing_resnet50_weights.pth')
                torch.save(model.state_dict(), best_path)
                flush_print(f"  >> Saved best model (acc: {best_acc:.4f})")

    elapsed_total = time.time() - start_time
    flush_print(f"\n{'='*60}")
    flush_print(f"Training complete in {elapsed_total/60:.1f} minutes")
    flush_print(f"Best Validation Accuracy: {best_acc:.4f}")
    flush_print(f"{'='*60}")

    # Save final tuned model
    tuned_path = os.path.join(MODEL_DIR, 'gear_bearing_resnet50_tuned.pth')
    torch.save(model.state_dict(), tuned_path)
    flush_print(f"Saved tuned model: {tuned_path}")

    # Save history
    history_path = os.path.join(MODEL_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    flush_print(f"Saved training history: {history_path}")

    # ── Confusion Matrix ─────────────────────────────────────────────────
    flush_print("\nGenerating confusion matrix...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['validation']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_preds)

    flush_print("\nConfusion Matrix:")
    header = f"{'':15s}" + "".join([f"{c[:10]:>12s}" for c in class_names])
    flush_print(header)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i][:15]:15s}" + "".join([f"{v:12d}" for v in row])
        flush_print(row_str)

    flush_print("\nClassification Report:")
    flush_print(classification_report(all_labels, all_preds,
                                       target_names=class_names, digits=4))


if __name__ == '__main__':
    main()
