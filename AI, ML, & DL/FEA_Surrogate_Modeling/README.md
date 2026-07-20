# Deep Learning Surrogate Modeling for Finite Element Analysis (FEA)

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Artificial Intelligence, Machine Learning & Deep Learning  

---

## 📌 Executive Summary

High-fidelity Finite Element Analysis (FEA) simulations (e.g., structural stress, hull deformation, displacement fields) are computationally expensive and can take hours to converge for complex geometries. This project develops a **Deep Neural Network (DNN) Surrogate Model** capable of emulating non-linear FEA structural solvers. By learning the non-linear mapping between spatial design/load parameters and stress/displacement responses, the surrogate model reduces evaluation latency from **minutes to milliseconds** with high structural accuracy.

---

## 🛠️ Architecture & Technical Methodology

```
+--------------------------------+      +-----------------------------------+
|  Spatial Load & Coordinate Data | ---> | Fully Connected DNN Architecture  |
|      (Nodes, Elements, Forces) |      | (Batch Normalization, GELU/ReLU)  |
+--------------------------------+      +-----------------------------------+
                                                          |
                                                          v
                                        +-----------------------------------+
                                        | Real-Time FEA Field Prediction    |
                                        | (Stress, Displacement, Strain)    |
                                        +-----------------------------------+
```

1. **Dataset Generation & Feature Pipeline (`utils.py`, `lp.py`)**:
   - Parses spatial node coordinates, element connection topologies, and load vectors.
   - Normalizes input parameters using standard z-score scaling.
2. **Surrogate Model Architecture (`student_model.py`, `nn_train.py`)**:
   - Deep Multi-Layer Perceptron (MLP) built in PyTorch with dense skip connections.
   - Batch normalization and dropout layers to prevent overfitting on localized stress concentrators.
3. **Training & Loss Formulation**:
   - Loss Function: Combined Mean Squared Error ($\text{MSE}$) and Relative L2 Norm:
     $$\mathcal{L} = \alpha \|\mathbf{y}_{true} - \mathbf{y}_{pred}\|_2^2 + \beta \frac{\|\mathbf{y}_{true} - \mathbf{y}_{pred}\|_2}{\|\mathbf{y}_{true}\|_2}$$
   - Adam Optimizer with exponential learning rate decay.
4. **Prediction & Testing (`nn_prediction.py`, `random_tester.py`)**:
   - Evaluates unseen load cases and compares prediction contours against FEA ground truth.

---

## 📁 Repository Structure

```
FEA_Surrogate_Modeling/
├── models/                   # Saved PyTorch/TensorFlow trained weights
├── data_and_othercodes/      # FEA node datasets and pre-processing scripts
├── DNN_architecture.png      # Network architecture visualization
├── hull_surrogate.png        # FEA stress contour comparison plot
├── nnn1_learning.png         # Loss convergence curve
├── student_model.py          # Neural network model definitions
├── nn_train.py               # Model training script & validation pipeline
├── nn_prediction.py          # Inference script for new geometry load cases
├── random_tester.py          # Stochastic evaluation & metric reporting
├── lp.py                     # Load parsing utility functions
├── range_of_interest         # Spatial region bounds file
└── utils.py                  # Tensor transformations & plotting helpers
```

---

## 🚀 Getting Started

### 1. Prerequisites
Install required Python libraries:
```bash
pip install torch numpy scipy pandas matplotlib seaborn
```

### 2. Model Training
To train the surrogate neural network from raw FEA simulation datasets:
```bash
python nn_train.py
```

### 3. Inference & Prediction
To predict structural fields on test load conditions:
```bash
python nn_prediction.py
```

### 4. Random Sampling Verification
Run stochastic sampling and benchmark prediction latency against numerical solvers:
```bash
python random_tester.py
```

---

## 📈 Key Results & Performance

- **Speedup**: Over **1,000x acceleration** compared to classical iterative FEA solvers.
- **Accuracy**: $< 2.5\%$ Relative L2 error across node displacement fields.
- **Applications**: Real-time structural optimization, digital twin applications, and interactive design space exploration.
