# Autonomous Vehicle Simulation & Control Systems

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Artificial Intelligence, Machine Learning & Deep Learning  

---

## 📌 Executive Summary

This project implements an end-to-end autonomous driving software pipeline operating inside the **CARLA Simulator** environment. It combines deep learning semantic segmentation for real-time lane boundary detection, PID and Model Predictive Control (MPC) steering/throttle controllers, and geometric camera calibration to join perception and control modules into a closed-loop self-driving vehicle stack.

---

## 🛠️ System Architecture & Subsystems

```
+-----------------------------------+
|  CARLA Camera Sensors & Images    |
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  1. Camera Calibration Module     |
|     (Intrinsic/Extrinsic Matrix)  |
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  2. Deep Learning Lane Detection  |
|     (Semantic Segmentation Model) |
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  3. Vehicle Control & Trajectory  |
|     (PID / MPC Steering Controller)|
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  CARLA Steering & Throttle Actuation|
+-----------------------------------+
```

### Module Breakdown:
1. **Camera Calibration**: Calculates intrinsic parameters ($\mathbf{K}$) and extrinsic matrices ($\mathbf{R}, \mathbf{t}$) to transform 2D pixel coordinates to 3D vehicle coordinate frames.
2. **Deep Learning Lane Detection**: Ingests front-facing camera feeds and predicts lane boundaries using convolutional neural network segmentation architectures.
3. **Vehicle Dynamics & Control**: Computes smooth steering angle and throttle commands based on lane polynomial fitting and target velocity curves.

---

## 📁 Repository Structure

```
Autonomous_Vehicle_Simulation/
├── book/                     # Technical documentation & chapter guides
├── code/                     # Python implementation scripts & exercise modules
│   └── solutions/            # Completed lane detection & control solvers
├── data/                     # Calibration target images & synthetic datasets
├── CONTRIBUTORS.md           # Author & maintainer details
├── LICENSE                   # Open-source license terms
└── README.md                 # Project README
```

---

## 🚀 Getting Started

### 1. Prerequisites
Install required dependencies:
```bash
pip install numpy scipy opencv-python torch torchvision matplotlib
```

Ensure CARLA Simulator (v0.9.11+) is installed and running on your system.

### 2. Running Lane Detection Pipeline
```bash
python code/solutions/lane_detection/lane_detector.py
```

### 3. Running Closed-Loop Control Loop
```bash
python code/solutions/control/steering_control.py
```

---

## 📈 Key Accomplishments

- Integrated deep neural segmentation with real-time CARLA simulator telemetry.
- Developed robust camera projection matrices for bird's-eye view lane extraction.
- Implemented smooth PID trajectory tracking under dynamic driving conditions.
