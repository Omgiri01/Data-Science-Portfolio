# 🌊 Physics-Informed Neural Network (PINN) for Incompressible Laminar Flows

This project implements a Physics-Informed Neural Network (PINN) designed to solve 2D Navier-Stokes equations for incompressible fluid flows past a cylinder. By integrating physical laws directly into the loss function of deep neural networks, this model predicts fluid velocities and pressure distributions from sparse boundary constraints without relying on grid discretization.

---

## 📐 Mathematical Formulation

Traditional deep neural networks act as pure data-driven estimators. In this project, the networks are constrained by the **conservation of mass (continuity)** and **conservation of momentum (Navier-Stokes)** equations:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

The neural network outputs $[u, v, p]$ (velocity fields and pressure) given space-time inputs $[x, y, t]$. The residual of these PDEs is minimized alongside standard boundary conditions to train the network.

---

## 📂 Project Structure

- **`PINN_steady/`**: Model configuration, training, and evaluation scripts for steady-state laminar flow past a cylinder.
- **`PINN_unsteady/`**: Model configurations handling transient (time-varying) fluid flows.
- **`FluentReferenceMu002/`**: High-fidelity reference solutions exported from ANSYS Fluent for validation.

---

## 🚀 Execution & Usage

1. **Pre-requisites**:
   ```bash
   pip install tensorflow numpy matplotlib scipy
   ```

2. **Run Steady-State Flow Training**:
   ```bash
   cd PINN_steady
   python train.py
   ```

3. **Run Transient (Unsteady) Flow Training**:
   ```bash
   cd PINN_unsteady
   python train.py
   ```

---

## 📈 Results Comparison

The models are validated against Ansys Fluent reference simulations. The neural network predictions for velocity magnitude and pressure contours demonstrate high alignment with traditional finite volume methods while allowing mesh-free inference.
