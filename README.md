# 🌟 Master Data Science, Artificial Intelligence & Machine Learning Portfolio

[![Author: Om Giri](https://img.shields.io/badge/Author-Om%20Giri-blue.svg?style=for-the-badge&logo=github)](https://github.com/Omgiri01)
[![Repository: Data-Science-Portfolio](https://img.shields.io/badge/Repository-Data--Science--Portfolio-10B981.svg?style=for-the-badge&logo=git)](https://github.com/Omgiri01/Data-Science-Portfolio)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Projects: 32](https://img.shields.io/badge/Total%20Projects-32-8B5CF6.svg?style=for-the-badge)](https://github.com/Omgiri01/Data-Science-Portfolio)

---

## 📌 Executive Summary & Portfolio Overview

Welcome to my master **Data Science, Artificial Intelligence, and Machine Learning Engineering Portfolio**. This repository represents a comprehensive showcase of end-to-end AI/ML systems engineering, containing **32 production-grade projects** spanning cutting-edge research and industrial applications.

The portfolio is structured into three primary engineering domains:

1. **🤖 Generative AI, Large Language Models (LLMs) & Stateful Agentic Systems** (17 Projects)  
   *Stateful agentic OS orchestration, multi-agent swarms, self-extending skill architectures, RAG with MMR vector retrieval, QLoRA instruction tuning, and prompt guardrail moderation APIs.*

2. **🧠 Deep Learning, Computer Vision & Physics-Informed AI (PINNs)** (11 Projects)  
   *Navier-Stokes Physics-Informed Neural Networks, DNN Finite Element Analysis surrogate models, CARLA self-driving autonomous vehicle control, EfficientNet deepfake detection, scalar autograd engines, and YOLOv8 real-time object tracking.*

3. **📊 Data Science, Predictive Engineering & MLOps** (4 Projects)  
   *Convolutional LSTM CFD fluid dynamics surrogates, C-MAPSS IoT turbofan engine predictive maintenance (RUL), Markowitz portfolio optimization, and inverted-index TF-IDF search engines built from scratch.*

---

## 🗺️ Master System Architecture & Domain Map

```
+-------------------------------------------------------------------------------------------------------------------+
|                                   OM GIRI AI & DATA SCIENCE PORTFOLIO ECOSYSTEM                                    |
+-------------------------------------------------------------------------------------------------------------------+
                                                          |
         +------------------------------------------------+------------------------------------------------+
         |                                                |                                                |
         v                                                v                                                v
+-----------------------------------+          +-----------------------------------+          +-----------------------------------+
| 1. GENERATIVE AI & AGENTS         |          | 2. DEEP LEARNING & CV             |          | 3. DATA SCIENCE & MLOPS           |
| - MangoOS (Agentic OS Layer)      |          | - Physics-Informed NNs (PINNs)    |          | - CFD Fluid Dynamics Surrogates   |
| - Self-Extending Agent (ADK/Gemini)|         | - FEA Surrogate (1000x Speedup)   |          | - IoT Predictive Maint. (RUL)     |
| - Financial Agent Crew (Agno)     |          | - CARLA Self-Driving Autonomous   |          | - Markowitz Portfolio Optimization|
| - Guardrail Moderation API        |          | - Micrograd Autograd Engine       |          | - Inverted Index TF-IDF Search    |
| - Fine-Tuned LLaMA-3 (QLoRA)      |          | - Deepfake EfficientNet Classifier|          | - Anomaly & Drift Analytics       |
| - Multi-Agent Research Swarm      |          | - Real-Time YOLOv8 Object Tracking|          | - Time-Series Feature Pipelines   |
| - Document & Transcript RAG       |          | - DCGAN Generative Art (CelebA)   |          | - Statistical Asset Modeling      |
+-----------------------------------+          +-----------------------------------+          +-----------------------------------+
         |                                                |                                                |
         +------------------------------------------------+------------------------------------------------+
                                                          |
                                                          v
+-------------------------------------------------------------------------------------------------------------------+
|                                        CORE FRAMEWORKS & INFRASTRUCTURE                                           |
| PyTorch • TensorFlow • LangGraph • ChromaDB • FAISS • OpenCV • FastAPI • Docker • Streamlit • NumPy • Pandas      |
+-------------------------------------------------------------------------------------------------------------------+
```

---

## 🔬 Key Mathematical & Theoretical Foundations

This portfolio emphasizes first-principles implementations. Key mathematical formulations utilized across projects include:

### 1. Physics-Informed Neural Networks (PINNs) — Navier-Stokes Fluid Flow
For 2D incompressible fluid flows governed by continuity and momentum conservation laws:
$$\nabla \cdot \mathbf{u} = 0 \quad \implies \quad \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$
The custom neural network loss function embeds physical constraints directly:
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_{cont} \mathcal{L}_{continuity} + \lambda_{mom} \mathcal{L}_{momentum}$$

### 2. Maximal Marginal Relevance (MMR) for Vector RAG Search
To eliminate redundant passages during contextual document retrieval:
$$\text{MMR} \equiv \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}_1(d_i, q) - (1 - \lambda) \max_{d_j \in S} \text{Sim}_2(d_i, d_j) \right]$$

### 3. Term Frequency-Inverse Document Frequency (TF-IDF) & Vector Space Search
Inverted-index text search engine scoring document $d$ against query $q$:
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}, \quad \text{IDF}(t, D) = \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$
$$\text{Score}(q, d) = \frac{\sum_{t \in q \cap d} \text{TF-IDF}(t, q) \cdot \text{TF-IDF}(t, d)}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

### 4. Markowitz Mean-Variance Portfolio Optimization
Finding optimal asset allocation vectors $\mathbf{w}$ minimizing portfolio variance $\sigma_p^2$:
$$\min_{\mathbf{w}} \mathbf{w}^T \mathbf{\Sigma} \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^T \mathbf{\mu} = \mu_{target}, \quad \mathbf{w}^T \mathbf{1} = 1$$

---

## 🗂️ Complete 32-Project Portfolio Catalog

### 🤖 Category 1: Generative AI, LLMs & Stateful Agentic Systems (17 Projects)

| # | Project Name | Tech Stack | Architectural & Algorithmic Highlights |
| :-: | :--- | :--- | :--- |
| 1 | **[MangoOS](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/Agentic_AI_Projects/MangoOS)** | LangGraph, FastAPI, Redis, Docker | Production-grade self-hosted Agentic AI Operating System executing commands inside isolated Docker sandboxes with state persistence. |
| 2 | **[Self-Extending Agent](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/Agentic_AI_Projects/Self_Extending_Agent)** | Google ADK, Gemini 2.5, Qwen | Agent architecture that dynamically generates, compiles, tests, and registers its own Python tools and skills at runtime. |
| 3 | **[Financial Analysis Agent Crew](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/Agentic_AI_Projects/Financial_Analysis_Agent_Crew)** | Agno (Phidata), Gemini 2.0, YFinance | Multi-agent financial intelligence swarm scraping market feeds, calculating financial ratios, and writing executive stock briefs. |
| 4 | **[Self-Correcting Data Validation Agent](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/Agentic_AI_Projects/Self_Correcting_Data_Validation_Agent)** | LangChain, Gemini, Firestore | Self-healing validation pipeline detecting input schema drift and auto-correcting ingestion anomalies in real-time. |
| 5 | **[AI Test Case Generator](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/Agentic_AI_Projects/AI_Test_Case_Generator)** | LangChain, OpenAI GPT-4o | Automated QA engineering agent parsing codebase structures and generating full unit/integration test suites. |
| 6 | **[Fine-Tuned Domain Chatbot](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/LLM_Projects/Fine_Tuned_Domain_Chatbot)** | LLaMA-3, QLoRA, PEFT, TRL, BitsAndBytes | Notebooks for 4-bit quantized instruction tuning, LoRA adapter merging, and ORPO preference alignment. |
| 7 | **[Guardrail Moderation API](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/LLM_Projects/Guardrail_Moderation_API)** | FastAPI, Regex, Transformers | High-throughput API gateway interceptor scanning for PII leakage, prompt injection vectors, and toxicity. |
| 8 | **[Real-Time Video Summarizer](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/LLM_Projects/Real_Time_Video_Summarizer)** | Diffusers, PyTorch, ModelScope | Text-to-video synthesis pipeline generating dynamic video sequences from descriptive text prompts. |
| 9 | **[Advanced Multi-Source RAG](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/RAG_Projects/Advanced_Multi_Source_RAG)** | LangChain, Tavily API, ChromaDB | Hybrid RAG routing queries dynamically between local document vector stores and live Tavily web search engines. |
| 10 | **[Chat With Codebase](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/RAG_Projects/Chat_With_Codebase)** | OpenAI Swarm, GPT-4o | Multi-agent software dev assistant performing architectural planning, code generation, and adversarial code reviews. |
| 11 | **[Multimodal Product Assistant](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/RAG_Projects/Multimodal_Product_Assistant)** | MedGemma, PyTorch, Streamlit | Clinical & product diagnostic assistant processing visual image inputs (X-rays/CT scans) alongside text parameters. |
| 12 | **[01_Generative_AI_Basics](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/01_Generative_AI_Basics)** | LangChain, OpenAI, Streamlit | Core LLM prompting templates, chain compositions, and the **CineSage** movie recommendation engine. |
| 13 | **[02_Document_RAG_ChromaDB](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/02_Document_RAG_ChromaDB)** | ChromaDB, OpenAI, Mistral | PDF document ingestion using 1000-char chunking and Maximal Marginal Relevance (MMR) diversity search. |
| 14 | **[03_Custom_Tool_Calling_Agents](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/03_Custom_Tool_Calling_Agents)** | LangChain, Python REPL | Dynamic tool binding framework allowing agents to execute local calculations, python code, and web lookups. |
| 15 | **[04_Multi_Agent_Research_System](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/04_Multi_Agent_Research_System)** | LangChain, Tavily, Streamlit | 4-agent autonomous research swarm (Searcher, Reader, Writer, Critic) with adversarial report scoring. |
| 16 | **[05_AI_Video_Assistant_RAG](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/05_AI_Video_Assistant_RAG)** | Whisper ASR, FAISS, Streamlit | Transcribing video/meeting recordings and performing semantic similarity search over transcript passages. |
| 17 | **[06_LangGraph_Agentic_Workflows](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Generative%20AI/06_LangGraph_Agentic_Workflows)** | LangGraph, MemorySaver | Cyclic stateful agent graphs featuring state reducers, checkpoint persistence, and human-in-the-loop validation. |

---

### 🧠 Category 2: Deep Learning, Computer Vision & Physics AI (11 Projects)

| # | Project Name | Tech Stack | Architectural & Algorithmic Highlights |
| :-: | :--- | :--- | :--- |
| 18 | **[FEA Surrogate Modeling](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/FEA_Surrogate_Modeling)** | PyTorch, SciPy, Matplotlib | Deep Neural Network surrogate replacing Finite Element Analysis solvers, delivering 1,000x structural evaluation speedups. |
| 19 | **[PINN Laminar Flow](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/PINN_Laminar_Flow)** | PyTorch, Automatic Diff | Physics-Informed Neural Network solving 2D incompressible Navier-Stokes equations for flow past a cylinder. |
| 20 | **[Autonomous Vehicle Simulation](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Autonomous_Vehicle_Simulation)** | CARLA, OpenCV, PyTorch | Autonomous driving stack inside CARLA combining lane segmentation, PID/MPC steering, and camera calibration. |
| 21 | **[Deepfake Detection](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Deepfake_Detection)** | TensorFlow, Keras, MTCNN | EfficientNet binary classifier intercepting video manipulations and facial forgery artifacts. |
| 22 | **[Generative Art GANs](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Generative_Art_GANs)** | PyTorch, Torchvision | Deep Convolutional GAN (DCGAN) performing mini-max adversarial training on MNIST and CelebA portraits. |
| 23 | **[Speech to Text Transcription](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Speech_to_Text_Transcription)** | Vosk, PocketSphinx, Whisper | Multi-engine speech recognition library supporting zero-latency offline models and cloud ASR APIs. |
| 24 | **[Deep Learning Framework Scratch (micrograd)](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Deep_Learning_Framework_Scratch)** | Pure Python, NumPy | Scalar-valued autograd engine building DAG computational graphs and topological sort backpropagation. |
| 25 | **[Reinforcement Learning Trading](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Reinforcement_Learning_Trading)** | Gymnasium, Stable-Baselines3 | Training PPO/DQN reinforcement learning agents to execute stock asset trading inside custom environments. |
| 26 | **[YOLO Object Detection](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/YOLO_Object_Detection)** | Ultralytics YOLOv8, OpenCV | Real-time 60+ FPS multi-object detection, bounding box localization, and camera tracking. |
| 27 | **[Masked Language Model NLP](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Masked_Language_Model_NLP)** | Transformers, PyTorch | Bidirectional BERT masked token prediction fine-tuning pipeline on custom domain text corpora. |
| 28 | **[Vector DB From Scratch](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/AI,%20ML,%20&%20DL/Vector_DB_From_Scratch)** | Pure Python, NumPy | Vector database engine supporting high-dimensional vector storage, cosine similarity math, and metadata filtering. |

---

### 📊 Category 3: Data Science, Predictive Analytics & MLOps (4 Projects)

| # | Project Name | Tech Stack | Architectural & Algorithmic Highlights |
| :-: | :--- | :--- | :--- |
| 29 | **[CFD Surrogate Modeling](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Data%20Science/CFD_Surrogate_Modeling)** | PyTorch, OpenFOAM, Keras | Convolutional LSTM predicting OpenFOAM supersonic shockwave fluid fields ($U, P, T$) in real-time. |
| 30 | **[Predictive Maintenance IoT](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Data%20Science/Predictive_Maintenance_IoT)** | Scikit-Learn, Pandas, NumPy | Machine health diagnostics on C-MAPSS turbofan engine sensor data predicting Remaining Useful Life (RUL). |
| 31 | **[financial_analysis_python](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Data%20Science/financial_analysis_python)** | Pandas, NumPy, Matplotlib | Statistical portfolio optimization, Markowitz Mean-Variance frontiers, Sharpe ratio tracking, and risk analysis. |
| 32 | **[Search Engine From Scratch](file:///c:/Users/OM%20GIRI/Downloads/DS%20Projects/Data%20Science/Search_Engine_From_Scratch)** | Pure Python, Math, Regex | Full Information Retrieval engine with tokenization, inverted index, logarithmic TF-IDF weights, and cosine matching. |

---

## 🔬 Flagship Project Deep Dives

### 🌟 Flagship 1: MangoOS — Self-Hosted Agentic AI Operating System
- **Core Technology**: LangGraph, FastAPI, Redis, Docker Sandboxes.
- **Problem Solved**: Provides a secure execution kernel for AI agents to run shell code, manipulate files, and manage background processes without risk to the host OS.
- **Architecture**: Ingests user goals $\rightarrow$ breaks down sub-tasks via LLM planner $\rightarrow$ dispatches tasks to isolated Docker execution containers $\rightarrow$ monitors progress with a React dynamic dashboard.

### 🌟 Flagship 2: Physics-Informed Neural Networks (PINNs)
- **Core Technology**: PyTorch, Automatic Differentiation ($\text{torch.autograd}$).
- **Problem Solved**: Solves non-linear 2D Navier-Stokes equations for incompressible fluid flow past a cylinder without grid discretization.
- **Key Metric**: Predicts continuous velocity ($u, v$) and pressure ($p$) fields from sparse boundary measurements while strictly enforcing momentum conservation.

### 🌟 Flagship 3: FEA & CFD Deep Learning Surrogate Models
- **Core Technology**: PyTorch, OpenFOAM, ConvLSTM2D, ResNet Skip Connections.
- **Problem Solved**: Replaces traditional numerical PDE solvers (which take hours per iteration) with deep neural surrogate networks.
- **Key Metric**: Achieves over **1,000x speedup** with $<2.5\%$ relative L2 error across node displacement and pressure fields.

---

## 🛠️ Complete Portfolio Technology Matrix

| Sub-Domain | Technologies, Frameworks & Tools |
| :--- | :--- |
| **Agentic Frameworks** | LangGraph, LangChain, Google Agent Development Kit (ADK), Agno (Phidata), OpenAI Swarm |
| **LLMs & Fine-Tuning** | OpenAI GPT-4o, Google Gemini 2.5, LLaMA-3, Mistral AI, Qwen, QLoRA, PEFT, TRL, BitsAndBytes |
| **Vector Databases & RAG** | ChromaDB, FAISS, Custom NumPy Vector Database Engine |
| **Deep Learning & Neural Networks** | PyTorch, PyTorch Geometric, TensorFlow 2.x, Keras, Ultralytics YOLOv8 |
| **Physics & Engineering AI** | Physics-Informed Neural Networks (PINNs), OpenFOAM CFD, CARLA Autonomous Simulator |
| **Data Science & Machine Learning** | Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn, XGBoost, LightGBM |
| **Audio, Vision & Speech** | Whisper ASR, Vosk, CMU PocketSphinx, OpenCV, MTCNN Face Detection |
| **Deployment & Middleware** | Docker, FastAPI, Streamlit, Uvicorn, Jupyter Notebooks, Redis, Firestore |

---

## 📂 Repository Directory Hierarchy

```
Data-Science-Portfolio/
├── AI, ML, & DL/                           # Deep Learning, Computer Vision & Physics AI (11 Projects)
│   ├── Autonomous_Vehicle_Simulation/      # CARLA simulator lane detection & steering control
│   ├── Deep_Learning_Framework_Scratch/    # Micrograd scalar autograd engine & backprop
│   ├── Deepfake_Detection/                 # EfficientNet deepfake video classifier
│   ├── Defect Detection in Gear & Bearing/ # Vibration signal CNN fault diagnosis
│   ├── FEA_Surrogate_Modeling/             # Deep neural network FEA surrogate (1000x speedup)
│   ├── Generative_Art_GANs/                # DCGAN image synthesis on CelebA & MNIST
│   ├── Masked_Language_Model_NLP/          # Bidirectional BERT masked token prediction
│   ├── PINN_Laminar_Flow/                  # Navier-Stokes Physics-Informed Neural Network
│   ├── Reinforcement_Learning_Trading/     # Gymnasium RL stock trading environment
│   ├── Speech_to_Text_Transcription/       # Multi-engine speech recognition (Vosk/Whisper)
│   ├── Vector_DB_From_Scratch/             # Pure Python high-dimensional vector database
│   └── YOLO_Object_Detection/              # Real-time 60+ FPS YOLOv8 object tracking
├── Data Science/                           # Data Science, Engineering & Analytics (4 Projects)
│   ├── CFD_Surrogate_Modeling/             # ConvLSTM OpenFOAM fluid flow surrogate
│   ├── Predictive_Maintenance_IoT/         # C-MAPSS turbofan engine RUL estimation
│   ├── Search_Engine_From_Scratch/         # Inverted index TF-IDF vector space search engine
│   └── financial_analysis_python/          # Markowitz portfolio optimization & risk analytics
├── Generative AI/                          # GenAI, RAG & Agentic Systems (17 Projects)
│   ├── 01_Generative_AI_Basics/            # LLM prompts & CineSage recommender
│   ├── 02_Document_RAG_ChromaDB/           # PDF ingestion & MMR retrieval search
│   ├── 03_Custom_Tool_Calling_Agents/      # Dynamic function binding pipelines
│   ├── 04_Multi_Agent_Research_System/     # 4-agent research swarm (Search/Read/Write/Critic)
│   ├── 05_AI_Video_Assistant_RAG/          # Whisper transcript FAISS semantic search
│   ├── 06_LangGraph_Agentic_Workflows/     # Stateful cyclic graphs with human-in-the-loop
│   ├── Agentic_AI_Projects/                # MangoOS, Self-Extending Agent, Financial Swarms
│   ├── LLM_Projects/                       # LLaMA-3 QLoRA fine-tuning & Guardrail API
│   └── RAG_Projects/                       # Advanced Multi-Source RAG & Codebase Assistant
├── LICENSE                                 # MIT Open-Source License Terms
└── README.md                               # Master Portfolio Landing README
```

---

## 🚀 Quick Start & Local Execution

### 1. Clone the Portfolio
```bash
git clone https://github.com/Omgiri01/Data-Science-Portfolio.git
cd Data-Science-Portfolio
```

### 2. Run a Web Interface (e.g. Streamlit Dashboard)
Each project contains its own self-contained environment guide. To launch a Streamlit dashboard:
```bash
cd "Generative AI/02_Document_RAG_ChromaDB"
pip install -r requirements.txt
streamlit run app.py
```

### 3. Run Standalone Python Code / Benchmarks
```bash
cd "AI, ML, & DL/Vector_DB_From_Scratch"
python demo.py
```

---

## 👤 Author & Profile

**Om Giri**  
- **GitHub Profile**: [@Omgiri01](https://github.com/Omgiri01)  
- **Portfolio Repository**: [Data-Science-Portfolio](https://github.com/Omgiri01/Data-Science-Portfolio)  

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
