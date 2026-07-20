# Real-Time Object Detection & Tracking with YOLOv8 & OpenCV

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Artificial Intelligence, Machine Learning & Deep Learning  

---

## 📌 Executive Summary

This project implements a high-performance **Real-Time Object Detection & Tracking Pipeline** utilizing **YOLOv8** (You Only Look Once) and **OpenCV**. It processes live video streams, static images, and high-frequency camera frames to perform multi-class object detection, bounding box localization, confidence scoring, and multi-object tracking.

---

## 🛠️ Technical Architecture

```
+------------------------------------+
| Input Stream (Webcam / MP4 / Image)|
+------------------------------------+
                  |
                  v
+------------------------------------+
| Preprocessing & Frame Normalization|
| (Resize, Color Space Conversion)   |
+------------------------------------+
                  |
                  v
+------------------------------------+
| Ultralytics YOLOv8 Backbone & Head |
| (Single-pass Feature Extraction)   |
+------------------------------------+
                  |
                  v
+------------------------------------+
| Non-Maximum Suppression (NMS)      |
| & Bounding Box Confidence Filtering|
+------------------------------------+
                  |
                  v
+------------------------------------+
| OpenCV Overlay & Real-Time Tracking|
+------------------------------------+
```

---

## 📁 Repository Structure

```
YOLO_Object_Detection/
├── test.py                # Detection script for single images & video files
├── image.jpg              # Sample input benchmark image
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Install Ultralytics and OpenCV:
```bash
pip install ultralytics opencv-python torch torchvision
```

### 2. Running Inference on Images
```python
from ultralytics import YOLO
import cv2

# Load YOLOv8 model weights
model = YOLO('yolov8n.pt')

# Run detection on input image
results = model('image.jpg')

# Display annotated results
results[0].show()
```

### 3. Running Real-Time Webcam Object Tracking
```bash
python test.py
```

---

## 📈 Performance Benchmarks

- **Inference Speed**: $> 60\text{ FPS}$ on standard GPU accelerators.
- **Mean Average Precision**: High $\text{mAP@[0.5:0.95]}$ across 80 COCO object classes.
- **Applications**: Autonomous vehicle obstacle tracking, security surveillance, and industrial quality inspection.