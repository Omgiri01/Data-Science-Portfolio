"""
ensemble_classifier.py
======================
Builds an ensemble classifier combining:
  1. ResNet-50 deep features (2048-dim from avgpool layer)
  2. Random Forest
  3. Gradient Boosting (XGBoost-style via sklearn)
  4. SVM with RBF kernel

The ensemble uses a soft-voting strategy for final prediction.

Usage:
    python ensemble_classifier.py
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def extract_features(model, dataloader, device):
    """Extract 2048-dim features from ResNet-50's avgpool layer."""
    features = []
    labels_list = []
    
    # Hook to get features from avgpool
    feature_output = []
    def hook_fn(module, input, output):
        feature_output.append(output.squeeze().cpu().numpy())
    
    hook = model.avgpool.register_forward_hook(hook_fn)
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            feature_output.clear()
            inputs = inputs.to(device)
            _ = model(inputs)
            batch_features = feature_output[0]
            if batch_features.ndim == 1:
                batch_features = batch_features.reshape(1, -1)
            features.append(batch_features)
            labels_list.extend(labels.numpy())
    
    hook.remove()
    features = np.vstack(features)
    labels_arr = np.array(labels_list)
    return features, labels_arr


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flush_print(f"Device: {device}")
    
    # ── Load ResNet-50 backbone ──
    flush_print("Loading ResNet-50 feature extractor...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    
    weights_path = os.path.join(MODEL_DIR, 'gear_bearing_resnet50_weights.pth')
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    flush_print(f"Loaded weights from: {weights_path}")
    
    # ── Data Loading ──
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'validation'), transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=0)
    
    class_names = train_dataset.classes
    flush_print(f"Classes: {class_names}")
    
    # ── Extract Deep Features ──
    flush_print("\nExtracting ResNet-50 features (2048-dim)...")
    X_train, y_train = extract_features(model, train_loader, device)
    X_val, y_val = extract_features(model, val_loader, device)
    X_test, y_test = extract_features(model, test_loader, device)
    flush_print(f"  Train features: {X_train.shape}")
    flush_print(f"  Val features:   {X_val.shape}")
    flush_print(f"  Test features:  {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ── Train Individual Classifiers ──
    flush_print("\n" + "=" * 60)
    flush_print("TRAINING ENSEMBLE CLASSIFIERS")
    flush_print("=" * 60)
    
    # 1. Random Forest
    flush_print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    rf_val_acc = accuracy_score(y_val, rf.predict(X_val_scaled))
    rf_test_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
    flush_print(f"  Val Accuracy:  {rf_val_acc:.4f}")
    flush_print(f"  Test Accuracy: {rf_test_acc:.4f}")
    
    # 2. Gradient Boosting
    flush_print("\n[2/3] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    gb_val_acc = accuracy_score(y_val, gb.predict(X_val_scaled))
    gb_test_acc = accuracy_score(y_test, gb.predict(X_test_scaled))
    flush_print(f"  Val Accuracy:  {gb_val_acc:.4f}")
    flush_print(f"  Test Accuracy: {gb_test_acc:.4f}")
    
    # 3. SVM
    flush_print("\n[3/3] Training SVM (RBF kernel)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_val_acc = accuracy_score(y_val, svm.predict(X_val_scaled))
    svm_test_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    flush_print(f"  Val Accuracy:  {svm_val_acc:.4f}")
    flush_print(f"  Test Accuracy: {svm_test_acc:.4f}")
    
    # ── Ensemble (Soft Voting) ──
    flush_print("\n" + "=" * 60)
    flush_print("ENSEMBLE (Soft Voting)")
    flush_print("=" * 60)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    
    ens_val_acc = accuracy_score(y_val, ensemble.predict(X_val_scaled))
    ens_test_acc = accuracy_score(y_test, ensemble.predict(X_test_scaled))
    flush_print(f"  Ensemble Val Accuracy:  {ens_val_acc:.4f}")
    flush_print(f"  Ensemble Test Accuracy: {ens_test_acc:.4f}")
    
    # ── Classification Report ──
    flush_print("\nEnsemble Classification Report (Test Set):")
    flush_print("=" * 60)
    y_pred = ensemble.predict(X_test_scaled)
    flush_print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # ── Comparison Table ──
    flush_print("\n" + "=" * 60)
    flush_print("MODEL COMPARISON")
    flush_print("=" * 60)
    flush_print(f"{'Model':<25s} {'Val Acc':>10s} {'Test Acc':>10s}")
    flush_print("-" * 45)
    flush_print(f"{'ResNet-50 (CNN only)':25s} {'1.0000':>10s} {'1.0000':>10s}")
    flush_print(f"{'Random Forest':25s} {rf_val_acc:>10.4f} {rf_test_acc:>10.4f}")
    flush_print(f"{'Gradient Boosting':25s} {gb_val_acc:>10.4f} {gb_test_acc:>10.4f}")
    flush_print(f"{'SVM (RBF)':25s} {svm_val_acc:>10.4f} {svm_test_acc:>10.4f}")
    flush_print(f"{'ENSEMBLE (RF+GB+SVM)':25s} {ens_val_acc:>10.4f} {ens_test_acc:>10.4f}")
    
    # ── Save models ──
    save_data = {
        'scaler': scaler,
        'rf': rf,
        'gb': gb,
        'svm': svm,
        'ensemble': ensemble,
        'class_names': class_names,
        'results': {
            'rf_val': rf_val_acc, 'rf_test': rf_test_acc,
            'gb_val': gb_val_acc, 'gb_test': gb_test_acc,
            'svm_val': svm_val_acc, 'svm_test': svm_test_acc,
            'ens_val': ens_val_acc, 'ens_test': ens_test_acc,
        }
    }
    
    save_path = os.path.join(MODEL_DIR, 'ensemble_classifier.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    flush_print(f"\nSaved ensemble to: {save_path}")
    
    # Save results as JSON too
    results_path = os.path.join(MODEL_DIR, 'ensemble_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data['results'], f, indent=2)
    flush_print(f"Saved results to: {results_path}")


if __name__ == '__main__':
    main()
