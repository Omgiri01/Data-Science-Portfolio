# ⚙️ Remaining Useful Life (RUL) Prediction with Deep CNN (Predictive Maintenance)

This project implements a Deep Convolutional Neural Network (DCNN) to estimate the Remaining Useful Life (RUL) of turbofan jet engines. Trained on preprocessed NASA C-MAPSS dataset tracks, the model applies a piecewise-linear RUL degradation threshold to predict equipment failure before it occurs.

---

## 📈 Degradation Modeling & Data Cleaning

In real-world telemetry, engines operate in a stable, normal state for an initial period ($R_{early}$ cycles) before micro-cracks and thermal stresses cause measurable degradation.
- **Piecewise Linear RUL**: A threshold is set (e.g. 125 cycles) where RUL is assumed constant during initial cycles, followed by linear decay until failure.
- **Feature Selection**: Sensors exhibiting non-i.i.d or zero-variance profiles (e.g. constant pressure outputs) are removed.
- **Target Ingestion**: Integrates `Train_data.csv` (preprocessed NASA FD001) and `test_engine_21.csv` for out-of-sample evaluation.

---

## 🧠 Neural Network Architecture

The DCNN is optimized for spatial-temporal sequence extraction:
- **Feature Extraction**: 4 convolutional hidden layers with varying filter counts (kernel size = 3) to capture high-frequency sensor anomalies.
- **Dimensionality Reduction**: `GlobalMaxPooling1D` layer to isolate maximum activation features.
- **Regularization**: `Dropout` layers stacked at regular intervals to prevent overfitting.
- **Output Activation**: Linear activation outputting a single continuous RUL projection value.

---

## 📂 Project Structure

- **`TURBOFAN Engine Remaining Useful Life Prediction with DCNN_main.ipynb`**: The main notebook containing data cleaning, data profiling, model construction, training, and testing loops.
- **`C-MAPSS_data_cleaner.py`**: A python script helper extracting and scaling sensor parameters.
- **`Train_data.csv`**: Cleansed training set containing sensor logs.
- **`test_engine_21.csv`**: Target test set logs for engine #21.

---

## 🚀 Usage Guide

1. **Install Dependencies**:
   ```bash
   pip install tensorflow pandas numpy matplotlib scikit-learn
   ```
2. **Execute Ingest & Training**:
   Run the notebook `TURBOFAN Engine Remaining Useful Life Prediction with DCNN_main.ipynb` to execute the preprocessing steps, train the neural network, and view predicted vs actual RUL curves.
