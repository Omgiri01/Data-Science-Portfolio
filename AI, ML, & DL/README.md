# 🧠 Deep Learning, Computer Vision & Physics-Informed AI Showcase

Welcome to my Deep Learning, Computer Vision, and Engineering AI portfolio section. This workspace contains advanced implementations focusing on **Physics-Informed Neural Networks (PINNs), structural stress surrogate modeling, neural autograd mechanics, real-time object tracking, and custom NLP transformers**.

---

## 📂 Project Architectures

### 1. [FEA_Surrogate_Modeling](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/FEA_Surrogate_Modeling)
- **Objective**: Develop a high-speed Multilayered Deep Neural Network (DNN) with skip connections to replace slow, computationally expensive Finite Element Analysis (FEA) structural simulations.
- **Key Features**: Auto-generating stress-strain profiles for complex pressure vessels and 3D subsea geometries, delivering 15x–700x speed-ups over direct numerical FEM solvers.
- **Tech Stack**: Python, TensorFlow, Keras, NumPy, Matplotlib.

### 2. [PINN_Laminar_Flow](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/PINN_Laminar_Flow)
- **Objective**: Implement a Physics-Informed Neural Network (PINN) to resolve Navier-Stokes equations for incompressible fluid flow past a cylinder.
- **Key Features**: Leverages physical loss constraints (integrating fluid conservation laws directly into neural training) to accurately predict velocity and pressure profiles even with sparse sensor inputs.
- **Tech Stack**: Python, PyTorch, TensorFlow, Physical boundary equations.

### 3. [Autonomous_Vehicle_Simulation](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Autonomous_Vehicle_Simulation)
- **Objective**: Construct lane-keeping systems, traffic light classifiers, and obstacle detectors inside virtual environments.
- **Key Features**: Ingests high-frequency camera views from CARLA simulator, applies bird's-eye perspective transformations, and outputs steering angle controls.
- **Tech Stack**: Python, OpenCV, CARLA Simulator API, ROS/ROS2 nodes.

### 4. [Deepfake_Detection](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Deepfake_Detection)
- **Objective**: Build binary classifiers to intercept video manipulations and face-swapping deepfakes.
- **Key Features**: Automates video frame extraction, runs MTCNN face cropping pipelines, and trains an EfficientNet backbone to scan for visual artifacts.
- **Tech Stack**: Python, Keras, TensorFlow, MTCNN, OpenCV.

### 5. [Generative_Art_GANs](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Generative_Art_GANs)
- **Objective**: Deploy Deep Convolutional Generative Adversarial Networks (DCGAN) to synthesize realistic visual artwork.
- **Key Features**: Alternating mini-max adversarial training between Generator and Discriminator neural networks to create MNIST/CelebA-like portraits.
- **Tech Stack**: Python, PyTorch, DCGAN, Torchvision.

### 6. [Speech_to_Text_Transcription](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Speech_to_Text_Transcription)
- **Objective**: Local offline audio transcription system running low-latency Speech-to-Text inference.
- **Key Features**: Processes real-time audio streams from PyAudio and executes transcription APIs.
- **Tech Stack**: Python, SpeechRecognition, PyAudio.

### 7. [Deep_Learning_Framework_Scratch (micrograd)](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Deep_Learning_Framework_Scratch)
- **Objective**: Build a clean scalar autograd engine and neural network module from scratch to demonstrate full backpropagation mechanics.
- **Key Features**: Formulates DAG computational graphs, tracks nodes and weights, and performs topological sorts to calculate gradients automatically.
- **Tech Stack**: Python, autograd math.

### 8. [Reinforcement_Learning_Trading](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Reinforcement_Learning_Trading)
- **Objective**: Train RL agents to trade stock assets using customizable Gymnasium environments.
- **Key Features**: Computes reward functions based on portfolio returns, implements continuous/discrete action spaces, and integrates Stable-Baselines3 agents.
- **Tech Stack**: Gymnasium, Stable-Baselines3, Pandas, NumPy.

### 9. [YOLO_Object_Detection](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/YOLO_Object_Detection)
- **Objective**: Real-time object tracking and label extraction.
- **Key Features**: Loads pre-trained YOLOv8 models, detects custom bounding boxes, and overlays frames with tracking annotations.
- **Tech Stack**: PyTorch, Ultralytics, OpenCV.

### 10. [Masked_Language_Model_NLP](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Masked_Language_Model_NLP)
- **Objective**: Context-aware bidirectional word prediction.
- **Key Features**: Tokenizes sequences and fine-tunes a BERT-style model to predict masked input spans.
- **Tech Stack**: Hugging Face Transformers, PyTorch.

### 11. [Vector_DB_From_Scratch](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Vector_DB_From_Scratch)
- **Objective**: A custom vector database implementing fast nearest-neighbor lookups.
- **Key Features**: Implements indexing, cosine distance formulations, and vector embeddings projection matches.
- **Tech Stack**: Python, NumPy.

### 12. [Defect Detection in Gear & Bearing](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Defect%20Detection%20in%20Gear%20&%20Bearing)
- **Objective**: Detect machinery faults using vibration signals.
- **Key Features**: Converts physical vibration time-series data into spectrograms and classifies anomalies with CNN.
- **Tech Stack**: PyTorch, Librosa, Signal Preprocessing.
