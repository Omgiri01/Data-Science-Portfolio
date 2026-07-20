"""
ensemble_with_noisy.py
======================
Trains ensemble classifiers on COMBINED clean + noisy data features.
This ensures the ensemble is robust to domain shift.

Pipeline:
1. Extract ResNet features from clean train data
2. Extract ResNet features from noisy data
3. Combine both feature sets for training
4. Train RF + GB + SVM ensemble on combined data
5. Test on held-out noisy samples
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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def extract_features(model, dataloader, device):
    """Extract 2048-dim features from ResNet-50's avgpool layer."""
    features = []
    labels_list = []
    
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
    return np.vstack(features), np.array(labels_list)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flush_print(f"Device: {device}")
    
    # Load both models
    flush_print("Loading models...")
    
    # Original model
    model_orig = models.resnet50(weights=None)
    model_orig.fc = nn.Linear(model_orig.fc.in_features, 6)
    model_orig.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'gear_bearing_resnet50_weights.pth'), map_location=device))
    model_orig = model_orig.to(device).eval()
    
    # Domain-adapted model
    model_da = models.resnet50(weights=None)
    model_da.fc = nn.Linear(model_da.fc.in_features, 6)
    model_da.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'gear_bearing_resnet50_domain_adapted.pth'), map_location=device))
    model_da = model_da.to(device).eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load clean data
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform)
    noisy_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'real_world_samples'), transform)
    
    class_names = train_dataset.classes
    flush_print(f"Classes: {class_names}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=0)
    noisy_loader = DataLoader(noisy_dataset, batch_size=16, num_workers=0)
    
    # ── Extract features from BOTH models ──
    flush_print("\nExtracting features...")
    
    # Original model features
    flush_print("  Original model - clean train...")
    X_train_orig, y_train = extract_features(model_orig, train_loader, device)
    flush_print("  Original model - clean test...")
    X_test_orig, y_test = extract_features(model_orig, test_loader, device)
    flush_print("  Original model - noisy...")
    X_noisy_orig, y_noisy = extract_features(model_orig, noisy_loader, device)
    
    # DA model features  
    flush_print("  DA model - clean train...")
    X_train_da, _ = extract_features(model_da, train_loader, device)
    flush_print("  DA model - clean test...")
    X_test_da, _ = extract_features(model_da, test_loader, device)
    flush_print("  DA model - noisy...")
    X_noisy_da, _ = extract_features(model_da, noisy_loader, device)
    
    # ── Strategy: Concatenate features from BOTH models (4096-dim) ──
    flush_print("\nConcatenating dual-model features (2048 + 2048 = 4096 dim)...")
    X_train_dual = np.hstack([X_train_orig, X_train_da])
    X_test_dual = np.hstack([X_test_orig, X_test_da])
    X_noisy_dual = np.hstack([X_noisy_orig, X_noisy_da])
    
    # Split noisy into train/test (50/50)
    X_noisy_train, X_noisy_test, y_noisy_train, y_noisy_test = train_test_split(
        X_noisy_dual, y_noisy, test_size=0.5, random_state=42, stratify=y_noisy
    )
    
    # Combine clean train + noisy train
    X_combined = np.vstack([X_train_dual, X_noisy_train])
    y_combined = np.concatenate([y_train, y_noisy_train])
    flush_print(f"  Combined training set: {X_combined.shape}")
    flush_print(f"  Clean test set: {X_test_dual.shape}")
    flush_print(f"  Noisy test set: {X_noisy_test.shape}")
    
    # Scale
    scaler = StandardScaler()
    X_combined_s = scaler.fit_transform(X_combined)
    X_test_s = scaler.transform(X_test_dual)
    X_noisy_test_s = scaler.transform(X_noisy_test)
    X_noisy_all_s = scaler.transform(X_noisy_dual)
    
    # ── Train classifiers ──
    flush_print("\n" + "=" * 60)
    flush_print("TRAINING ROBUST ENSEMBLE (Clean + Noisy Features)")
    flush_print("=" * 60)
    
    flush_print("\n[1/3] Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_combined_s, y_combined)
    rf_clean = accuracy_score(y_test, rf.predict(X_test_s))
    rf_noisy = accuracy_score(y_noisy_test, rf.predict(X_noisy_test_s))
    rf_noisy_all = accuracy_score(y_noisy, rf.predict(X_noisy_all_s))
    flush_print(f"  Clean test acc:  {rf_clean:.4f}")
    flush_print(f"  Noisy test acc:  {rf_noisy:.4f}")
    flush_print(f"  Noisy ALL acc:   {rf_noisy_all:.4f}")
    
    flush_print("\n[2/3] Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_combined_s, y_combined)
    gb_clean = accuracy_score(y_test, gb.predict(X_test_s))
    gb_noisy = accuracy_score(y_noisy_test, gb.predict(X_noisy_test_s))
    gb_noisy_all = accuracy_score(y_noisy, gb.predict(X_noisy_all_s))
    flush_print(f"  Clean test acc:  {gb_clean:.4f}")
    flush_print(f"  Noisy test acc:  {gb_noisy:.4f}")
    flush_print(f"  Noisy ALL acc:   {gb_noisy_all:.4f}")
    
    flush_print("\n[3/3] SVM (RBF)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    svm.fit(X_combined_s, y_combined)
    svm_clean = accuracy_score(y_test, svm.predict(X_test_s))
    svm_noisy = accuracy_score(y_noisy_test, svm.predict(X_noisy_test_s))
    svm_noisy_all = accuracy_score(y_noisy, svm.predict(X_noisy_all_s))
    flush_print(f"  Clean test acc:  {svm_clean:.4f}")
    flush_print(f"  Noisy test acc:  {svm_noisy:.4f}")
    flush_print(f"  Noisy ALL acc:   {svm_noisy_all:.4f}")
    
    # Ensemble
    flush_print("\n" + "=" * 60)
    flush_print("ROBUST ENSEMBLE (Soft Voting)")
    flush_print("=" * 60)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
        voting='soft'
    )
    ensemble.fit(X_combined_s, y_combined)
    
    ens_clean = accuracy_score(y_test, ensemble.predict(X_test_s))
    ens_noisy = accuracy_score(y_noisy_test, ensemble.predict(X_noisy_test_s))
    ens_noisy_all = accuracy_score(y_noisy, ensemble.predict(X_noisy_all_s))
    flush_print(f"  Clean test acc:  {ens_clean:.4f}")
    flush_print(f"  Noisy test acc:  {ens_noisy:.4f}")
    flush_print(f"  Noisy ALL acc:   {ens_noisy_all:.4f}")
    
    # Classification report on noisy
    flush_print("\nEnsemble Report on Noisy Test Set:")
    flush_print(classification_report(y_noisy_test, ensemble.predict(X_noisy_test_s), 
                                       target_names=class_names, digits=4))
    
    # Final comparison
    flush_print("\n" + "=" * 60)
    flush_print("FINAL MODEL COMPARISON")
    flush_print("=" * 60)
    flush_print(f"{'Model':<30s} {'Clean':>8s} {'Noisy':>8s}")
    flush_print("-" * 50)
    flush_print(f"{'Random Forest':30s} {rf_clean*100:>7.1f}% {rf_noisy_all*100:>7.1f}%")
    flush_print(f"{'Gradient Boosting':30s} {gb_clean*100:>7.1f}% {gb_noisy_all*100:>7.1f}%")
    flush_print(f"{'SVM (RBF)':30s} {svm_clean*100:>7.1f}% {svm_noisy_all*100:>7.1f}%")
    flush_print(f"{'ENSEMBLE (RF+GB+SVM)':30s} {ens_clean*100:>7.1f}% {ens_noisy_all*100:>7.1f}%")
    
    # Save
    save_data = {
        'scaler': scaler,
        'rf': rf, 'gb': gb, 'svm': svm, 'ensemble': ensemble,
        'class_names': class_names,
        'uses_dual_features': True,
        'results': {
            'rf_clean': rf_clean, 'rf_noisy': rf_noisy_all,
            'gb_clean': gb_clean, 'gb_noisy': gb_noisy_all,
            'svm_clean': svm_clean, 'svm_noisy': svm_noisy_all,
            'ens_clean': ens_clean, 'ens_noisy': ens_noisy_all,
        }
    }
    
    save_path = os.path.join(MODEL_DIR, 'ensemble_robust.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    flush_print(f"\nSaved to: {save_path}")
    
    results_path = os.path.join(MODEL_DIR, 'ensemble_robust_results.json')
    with open(results_path, 'w') as f:
        json.dump(save_data['results'], f, indent=2)
    flush_print(f"Results: {results_path}")


if __name__ == '__main__':
    main()
