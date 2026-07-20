# 🌊 Data-Driven CFD Surrogate Modeling Using Deep Learning (ConvLSTM)

This project implements a spatiotemporal deep learning surrogate designed to accelerate Computational Fluid Dynamics (CFD) simulations for supersonic flow fields. By training a **Convolutional LSTM (ConvLSTM)** network with ResNet-style skip connections on OpenFOAM dataset grids, this model predicts fluid velocities, pressures, and temperatures dynamically in real-time, bypassing slow numerical solvers.

---

## 📐 Problem Formulation & Case Setup

Supercritical and supersonic flows are characterized by strong shocks and discontinuities. In this project, **supersonic flow over a forward-facing step** is investigated using [OpenFOAM](https://openfoam.org/) for numerical pre-processing and data generation.

The geometry consists of a Mach 3 flow entering a rectangular channel with a structural step, inducing intersecting shock reflections.

---

## ⚙️ Data Generation & Pipeline

To train the surrogate model, we automatically generate high-velocity simulations by varying step dimensions ($0.1 < x < 2.9$) and height parameters ($0.1 < y < 0.4$):
1. Parametric mesh script (`gen_blockMeshDict.py`) writes mesh and cell matrices.
2. The solver executing `sonicFoam` handles supersonic compression.
3. Post-processor `foamToVTK` converts grid databases into `.vtk` fields.
4. Key physical arrays are extracted: $x$-direction velocity ($u$), $y$-direction velocity ($v$), pressure ($p$), and temperature ($T$).

---

## 🧠 Model Architecture

The neural network utilizes **ConvLSTM2D** cells to manage spatiotemporal correlations. The spatial features are downsampled using time-distributed convolutional networks:

- **Spatial Encoder**: 3x `TimeDistributed(Conv2D)` layers extracting transient boundary descriptors.
- **Spatiotemporal Recurrent Unit**: `ConvLSTM2D` layers resolving spatial-temporal equations.
- **ResNet Skip Connections**: Short-circuit channels added to propagate features across blocks and ensure fast training convergence.

---

## 📂 Project Structure

- **`codes/`**: Includes OpenFOAM mesh scripts, solver macros, and model execution notebooks.
- **`assets/`**: Visual schematics and model architecture diagrams.
- **`requirements.txt`**: Project dependency definitions.

---

## 🚀 Usage Guide

1. **Pre-requisites**:
   ```bash
   pip install tensorflow numpy matplotlib scipy pandas
   ```

2. **Inference / Training**:
   Run the training notebook or python modules inside the `codes/` directory to run the ConvLSTM model:
   ```bash
   python codes/train_model.py
   ```
