# Defect Detection in Gear & Bearing using Deep Learning & Ensemble Methods

> **6th Semester AI/ML Project** — ResNet-50 + Random Forest + Gradient Boosting + SVM Ensemble achieving **99.2% accuracy on noisy real-world data**

---

## Quick Results

| Model | Clean Data | Noisy Real-World |
|-------|-----------|------------------|
| ResNet-50 (baseline) | 100% | 66.7% |
| ResNet-50 + Domain Adaptation | 100% | 74.2% |
| **Ensemble (RF+GB+SVM)** | **100%** | **99.2%** |

## Pipeline Overview

```
Raw Vibration Signals (.mat/.csv)
        ↓
  STFT Spectrograms (224×224 PNG)
        ↓
  ResNet-50 Transfer Learning (100% on clean data)
        ↓
  Domain Shift Problem (67% on noisy data)
        ↓
  Domain Adaptation + Ensemble (RF+GB+SVM)
        ↓
  99.2% on Noisy Real-World Data ✓
```

## 6 Fault Classes

| # | Class | Source Dataset |
|---|-------|---------------|
| 1 | Normal Bearing | CWRU |
| 2 | Ball Fault | CWRU |
| 3 | Inner Race Fault | CWRU |
| 4 | Outer Race Fault | CWRU |
| 5 | Healthy Gear | PHM Gearbox |
| 6 | Broken Tooth | PHM Gearbox |

## Project Structure

```
├── notebooks/
│   ├── 01_Data_Preparation_and_Spectrogram_Generation.ipynb
│   ├── 02_Model_Training_and_Validation.ipynb
│   └── 03_Real_World_Inference.ipynb    ← Ensemble + DA results
├── scripts/
│   ├── download_datasets.py             ← Kaggle data download
│   ├── generate_spectrograms.py         ← Signal → STFT → PNG
│   ├── generate_noisy_spectrograms.py   ← Noise injection
│   ├── train_model.py                   ← Two-phase ResNet training
│   ├── fine_tune_domain_adaptation.py   ← Domain adaptation
│   ├── ensemble_with_noisy.py           ← Robust ensemble (RF+GB+SVM)
│   └── predict.py                       ← CLI inference
├── models/                              ← Saved weights & results
├── data/                                ← Spectrograms & raw signals
└── Final_Project_Report.md              ← Detailed report
```

## How to Run

```bash
# 1. Install dependencies
pip install torch torchvision scikit-learn scipy pandas matplotlib kaggle

# 2. Download datasets
python scripts/download_datasets.py

# 3. Generate spectrograms
python scripts/generate_spectrograms.py

# 4. Train ResNet-50
python scripts/train_model.py

# 5. Generate noisy test data
python scripts/generate_noisy_spectrograms.py

# 6. Domain adaptation
python scripts/fine_tune_domain_adaptation.py

# 7. Train robust ensemble
python scripts/ensemble_with_noisy.py

# 8. Run inference
python scripts/predict.py path/to/spectrogram.png
```

## Technologies

- **PyTorch** — ResNet-50 transfer learning
- **scikit-learn** — Random Forest, Gradient Boosting, SVM
- **SciPy** — STFT signal processing
- **Matplotlib** — Spectrogram visualization

## Detailed Report

See [Final_Project_Report.md](Final_Project_Report.md) for the complete methodology, results, and analysis.
