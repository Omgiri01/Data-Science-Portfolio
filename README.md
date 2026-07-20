<div align="center">

# ⚡ Om Giri — AI & Data Science Engineering Portfolio

### *Building Production-Grade Intelligent Systems at the Intersection of Research & Engineering*

[![Author](https://img.shields.io/badge/Author-Om%20Giri-6366F1?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Omgiri01)
[![GitHub](https://img.shields.io/badge/GitHub-Omgiri01-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Omgiri01)
[![Portfolio](https://img.shields.io/badge/Portfolio-Data--Science-10B981?style=for-the-badge&logo=databricks&logoColor=white)](https://github.com/Omgiri01/Data-Science-Portfolio)
[![License](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)

[![Total Projects](https://img.shields.io/badge/Total%20Projects-32-8B5CF6?style=flat-square&logo=git&logoColor=white)](https://github.com/Omgiri01/Data-Science-Portfolio)
[![Generative AI](https://img.shields.io/badge/GenAI%20%26%20Agents-17%20Projects-EC4899?style=flat-square)](https://github.com/Omgiri01/Data-Science-Portfolio)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning%20%26%20CV-11%20Projects-3B82F6?style=flat-square)](https://github.com/Omgiri01/Data-Science-Portfolio)
[![Data Science](https://img.shields.io/badge/Data%20Science%20%26%20MLOps-4%20Projects-14B8A6?style=flat-square)](https://github.com/Omgiri01/Data-Science-Portfolio)

---

*"From writing custom autograd engines from scratch to deploying production multi-agent agentic operating systems — this portfolio reflects depth over breadth, first principles over abstraction, and engineering excellence over prototyping."*

</div>

---

## 📋 Table of Contents

| Section | Description |
| :--- | :--- |
| [🎯 Portfolio At a Glance](#-portfolio-at-a-glance) | Summary metrics and domain map |
| [🏆 Flagship Projects](#-flagship-projects) | Deep-dive into top 5 standout projects |
| [🤖 Generative AI & Agents (17)](#-generative-ai-llms--stateful-agentic-systems) | LLMs, RAG systems, multi-agent swarms |
| [🧠 Deep Learning & Physics AI (11)](#-deep-learning-computer-vision--physics-ai) | CV, PINNs, FEA surrogates, GANs |
| [📊 Data Science & MLOps (4)](#-data-science-predictive-analytics--mlops) | CFD surrogates, IoT analytics, finance |
| [⚙️ Technology Matrix](#️-complete-technology-matrix) | Full stack of 40+ frameworks |
| [🔬 Mathematical Foundations](#-mathematical--theoretical-foundations) | Core algorithms with formulations |
| [📂 Repository Structure](#-repository-structure) | Full directory tree |
| [🚀 Quick Start](#-quick-start) | Get running in 60 seconds |

---

## 🎯 Portfolio At a Glance

<table>
<tr>
<td width="33%" align="center">

### 🤖 Generative AI & Agents
**17 Projects**

`LangGraph` `LangChain` `Google ADK`  
`Gemini 2.5` `LLaMA-3 QLoRA` `FAISS`  
`ChromaDB` `FastAPI` `Docker`

Multi-agent orchestration • Self-extending runtimes • Stateful RAG pipelines • LLM fine-tuning

</td>
<td width="33%" align="center">

### 🧠 Deep Learning & CV
**11 Projects**

`PyTorch` `TensorFlow` `YOLOv8`  
`CARLA Simulator` `OpenCV` `MTCNN`  
`Gymnasium` `Stable-Baselines3`

PINNs • FEA Surrogates • Deepfake Detection • RL Trading • GAN Image Synthesis

</td>
<td width="33%" align="center">

### 📊 Data Science & MLOps
**4 Projects**

`Pandas` `Scikit-Learn` `OpenFOAM`  
`Matplotlib` `SciPy` `XGBoost`  
`ConvLSTM` `NumPy`

CFD Surrogates • IoT RUL • Portfolio Opt. • Custom Search Engines

</td>
</tr>
</table>

---

### 📈 Domain Breakdown

```
  Generative AI & Agents (53%)   ████████████████████████░░░░░░░░░░░░░░░░░░░░░░
  Deep Learning & CV     (34%)   ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  Data Science & MLOps   (13%)   █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

---

## 🏆 Flagship Projects

> The five projects below represent the most technically ambitious and production-complete systems in this portfolio.

---

### 🥇 Flagship 1 — MangoOS: Self-Hosted Agentic AI Operating System

[![MangoOS](https://img.shields.io/badge/Category-Agentic%20AI%20OS-6366F1?style=flat-square)](https://github.com/Omgiri01)
[![Stack](https://img.shields.io/badge/Stack-LangGraph%20%7C%20FastAPI%20%7C%20Redis%20%7C%20Docker-181717?style=flat-square)](https://github.com/Omgiri01)

**MangoOS** is a production-grade **self-hosted Agentic AI Operating System** that allows LLM-powered agents to autonomously plan, schedule, and execute commands inside isolated Docker containers — providing a fully sandboxed, secure AI compute kernel.

**How It Works:**
```
User Goal Input
      │
      ▼
┌─────────────────────────────┐
│  LangGraph Task Planner     │  ← LLM decomposes goal into sub-tasks
└────────────┬────────────────┘
             │
      ┌──────▼──────┐
      │ Task Queue  │  ← Redis-backed priority queue
      └──────┬──────┘
             │
   ┌─────────▼──────────┐
   │  Docker Executor   │  ← Isolated containerised execution kernel
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │  Real-Time Monitor │  ← FastAPI WebSocket + Live dashboard
   └────────────────────┘
```

| Component | Technology | Role |
| :--- | :--- | :--- |
| Orchestration | LangGraph (StatefulGraph) | Cyclic agent execution with checkpointing |
| API Gateway | FastAPI + Uvicorn | RESTful and WebSocket command interfaces |
| State Store | Redis (Pub/Sub) | Task queue, agent memory, event streaming |
| Execution Kernel | Docker SDK (Python) | Sandboxed, isolated process execution |
| Monitoring | Streamlit Dashboard | Real-time task progress and agent observability |

---

### 🥈 Flagship 2 — Physics-Informed Neural Network (PINN) for Navier-Stokes

[![PINN](https://img.shields.io/badge/Category-Physics%20AI%20%7C%20CFD-3B82F6?style=flat-square)](https://github.com/Omgiri01)
[![Stack](https://img.shields.io/badge/Stack-PyTorch%20%7C%20Automatic%20Diff-EE4C2C?style=flat-square)](https://github.com/Omgiri01)

A custom PyTorch neural network that learns to satisfy **2D incompressible Navier-Stokes fluid dynamics equations** as hard constraints — replacing traditional mesh-based CFD solvers.

**Governing Equations embedded as PDE loss terms:**

```
Continuity:      ∂u/∂x + ∂v/∂y = 0
Momentum (x):    ρ(u·∂u/∂x + v·∂u/∂y) = -∂p/∂x + μ·∇²u
Momentum (y):    ρ(u·∂v/∂x + v·∂v/∂y) = -∂p/∂y + μ·∇²v

Total Loss:      L = L_data + λ₁·L_continuity + λ₂·L_momentum
```

**Key Engineering Details:**
- Collocation point sampling on interior domain and boundary
- Adam → L-BFGS two-phase optimizer schedule for convergence
- Predicts continuous velocity `(u, v)` and pressure `p` fields from sparse boundary observations
- Zero mesh discretization — infers entire continuous field from collocation residuals

---

### 🥉 Flagship 3 — FEA Deep Neural Network Surrogate Model

[![FEA](https://img.shields.io/badge/Category-Scientific%20ML%20%7C%20Surrogates-F59E0B?style=flat-square)](https://github.com/Omgiri01)
[![Stack](https://img.shields.io/badge/Stack-PyTorch%20%7C%20SciPy%20%7C%20OpenFOAM-EE4C2C?style=flat-square)](https://github.com/Omgiri01)

A deep neural network trained to **replace Finite Element Analysis (FEA) solvers** — achieving near-exact structural response predictions 1,000× faster than traditional simulation.

| Metric | FEA Solver | DNN Surrogate |
| :--- | :---: | :---: |
| Solve Time (per sample) | ~2 hours | ~7 ms |
| Relative L2 Error | — | < 2.5% |
| GPU Parallelism | ✗ | ✓ |
| Real-Time Inference | ✗ | ✓ |
| Speedup Factor | 1× | **>1,000×** |

---

### 🏅 Flagship 4 — Self-Extending Agent (Google ADK + Gemini 2.5)

[![Self-Extend](https://img.shields.io/badge/Category-Self--Modifying%20Agentic%20AI-EC4899?style=flat-square)](https://github.com/Omgiri01)
[![Stack](https://img.shields.io/badge/Stack-Google%20ADK%20%7C%20Gemini%202.5%20%7C%20Qwen-4285F4?style=flat-square)](https://github.com/Omgiri01)

An agent that **writes, tests, and permanently registers its own new Python tools at runtime**, expanding its own capability surface area through live code generation without any human intervention.

```
User Request → Agent detects missing capability
                    │
                    ▼
          LLM generates new Python tool code
                    │
                    ▼
          Sandboxed execution & unit test pass
                    │
                    ▼
          Tool registered into agent skill registry
                    │
                    ▼
          Future requests invoke new skill ✓
```

---

### 🏅 Flagship 5 — CFD Surrogate with ConvLSTM (OpenFOAM + Deep Learning)

[![CFD](https://img.shields.io/badge/Category-Scientific%20ML%20%7C%20Fluid%20Dynamics-14B8A6?style=flat-square)](https://github.com/Omgiri01)
[![Stack](https://img.shields.io/badge/Stack-PyTorch%20%7C%20OpenFOAM%20%7C%20Keras%20ConvLSTM-EE4C2C?style=flat-square)](https://github.com/Omgiri01)

A **Convolutional LSTM deep learning model** trained on OpenFOAM supersonic flow simulation data to predict real-time velocity field `U`, pressure `P`, and temperature `T` distributions — replacing hour-long numerical simulations with sub-second inference.

---

## 🤖 Generative AI, LLMs & Stateful Agentic Systems

> 17 projects across agentic orchestration, RAG, LLM fine-tuning, and intelligent workflow automation.

### 🔷 Agentic AI Projects

| # | Project | Stack | What It Does |
| :-: | :--- | :--- | :--- |
| 1 | **[MangoOS](Generative%20AI/Agentic_AI_Projects/MangoOS)** | LangGraph · FastAPI · Redis · Docker | Self-hosted Agentic OS executing shell commands inside isolated Docker sandboxes with real-time task monitoring. |
| 2 | **[Self-Extending Agent](Generative%20AI/Agentic_AI_Projects/Self_Extending_Agent)** | Google ADK · Gemini 2.5 · Qwen | Agent that generates, validates, and registers its own Python tools and skills at runtime — perpetually expanding its capability. |
| 3 | **[Financial Analysis Agent Crew](Generative%20AI/Agentic_AI_Projects/Financial_Analysis_Agent_Crew)** | Agno (Phidata) · Gemini 2.0 · YFinance | Multi-agent financial intelligence swarm scraping live market data, computing financial ratios, and generating executive stock briefs. |
| 4 | **[Self-Correcting Data Validation Agent](Generative%20AI/Agentic_AI_Projects/Self_Correcting_Data_Validation_Agent)** | LangChain · Gemini · Firestore | Autonomous data quality pipeline detecting schema drift and self-correcting ingestion anomalies in streaming data. |
| 5 | **[AI Test Case Generator](Generative%20AI/Agentic_AI_Projects/AI_Test_Case_Generator)** | LangChain · OpenAI GPT-4o | Automated QA engineering agent parsing codebase structures and generating comprehensive unit/integration test suites. |

### 🔷 LLM Engineering Projects

| # | Project | Stack | What It Does |
| :-: | :--- | :--- | :--- |
| 6 | **[Fine-Tuned Domain Chatbot](Generative%20AI/LLM_Projects/Fine_Tuned_Domain_Chatbot)** | LLaMA-3 · QLoRA · PEFT · TRL · BitsAndBytes | End-to-end pipeline for 4-bit quantized instruction tuning, LoRA adapter merging, and ORPO preference alignment of domain-specific LLMs. |
| 7 | **[Guardrail Moderation API](Generative%20AI/LLM_Projects/Guardrail_Moderation_API)** | FastAPI · Regex · Transformers | High-throughput API interceptor gateway scanning for PII leakage, prompt injection attacks, and LLM output toxicity at the edge. |
| 8 | **[Real-Time Video Summarizer](Generative%20AI/LLM_Projects/Real_Time_Video_Summarizer)** | Diffusers · PyTorch · ModelScope | Text-to-video synthesis pipeline generating dynamic, coherent video sequences from structured text descriptions. |

### 🔷 RAG & Knowledge Retrieval Projects

| # | Project | Stack | What It Does |
| :-: | :--- | :--- | :--- |
| 9 | **[Advanced Multi-Source RAG](Generative%20AI/RAG_Projects/Advanced_Multi_Source_RAG)** | LangChain · Tavily · ChromaDB | Hybrid RAG system routing queries dynamically between local vector document stores and live Tavily web search APIs. |
| 10 | **[Chat With Codebase](Generative%20AI/RAG_Projects/Chat_With_Codebase)** | OpenAI Swarm · GPT-4o | Multi-agent software engineering assistant performing code planning, generation, and adversarial review over full codebases. |
| 11 | **[Multimodal Product Assistant](Generative%20AI/RAG_Projects/Multimodal_Product_Assistant)** | MedGemma · PyTorch · Streamlit | Clinical diagnostic assistant processing visual medical image inputs (X-rays/CT scans) alongside structured text parameters. |

### 🔷 LangChain & LangGraph Curriculum Projects

| # | Project | Stack | What It Does |
| :-: | :--- | :--- | :--- |
| 12 | **[01 — Generative AI Basics](Generative%20AI/01_Generative_AI_Basics)** | LangChain · OpenAI · Streamlit | Core LLM prompting patterns, chain compositions, and **CineSage** — an LLM-powered movie recommendation engine. |
| 13 | **[02 — Document RAG with ChromaDB](Generative%20AI/02_Document_RAG_ChromaDB)** | ChromaDB · OpenAI · Mistral | PDF ingestion pipeline with 1000-char chunking strategy and Maximal Marginal Relevance (MMR) for diversity-aware retrieval. |
| 14 | **[03 — Custom Tool-Calling Agents](Generative%20AI/03_Custom_Tool_Calling_Agents)** | LangChain · Python REPL | Dynamic function binding enabling agents to autonomously execute Python code, local calculations, and web lookups. |
| 15 | **[04 — Multi-Agent Research System](Generative%20AI/04_Multi_Agent_Research_System)** | LangChain · Tavily · Streamlit | 4-agent autonomous research swarm architecture: `Searcher → Reader → Writer → Critic` with adversarial quality scoring. |
| 16 | **[05 — AI Video Assistant RAG](Generative%20AI/05_AI_Video_Assistant_RAG)** | Whisper ASR · FAISS · Streamlit | Meeting recording transcription pipeline with FAISS-backed semantic similarity search over timestamped transcript segments. |
| 17 | **[06 — LangGraph Agentic Workflows](Generative%20AI/06_LangGraph_Agentic_Workflows)** | LangGraph · MemorySaver | Cyclic stateful agent execution graphs with state reducers, checkpoint memory persistence, and human-in-the-loop approval nodes. |

---

## 🧠 Deep Learning, Computer Vision & Physics AI

> 11 projects spanning neural architecture engineering, physics-constrained learning, and real-time vision systems.

| # | Project | Stack | Impact & Highlights |
| :-: | :--- | :--- | :--- |
| 18 | **[FEA Surrogate Modeling](AI%2C%20ML%2C%20%26%20DL/FEA_Surrogate_Modeling)** | PyTorch · SciPy · Matplotlib | DNN surrogate replacing Finite Element solvers. **>1,000× speedup** with <2.5% L2 error across structural fields. |
| 19 | **[PINN Laminar Flow](AI%2C%20ML%2C%20%26%20DL/PINN_Laminar_Flow)** | PyTorch · Autograd | Physics-Informed NN solving 2D incompressible Navier-Stokes past a cylinder with embedded PDE momentum residuals. |
| 20 | **[Autonomous Vehicle Simulation](AI%2C%20ML%2C%20%26%20DL/Autonomous_Vehicle_Simulation)** | CARLA · OpenCV · PyTorch | Full self-driving stack: semantic segmentation lane detection, PID/MPC steering control, and camera sensor calibration. |
| 21 | **[Deepfake Detection](AI%2C%20ML%2C%20%26%20DL/Deepfake_Detection)** | TensorFlow · Keras · MTCNN · EfficientNet | Binary classifier detecting GAN-generated facial forgeries with frame-level face alignment and attention-guided training. |
| 22 | **[Generative Art GANs](AI%2C%20ML%2C%20%26%20DL/Generative_Art_GANs)** | PyTorch · Torchvision | DCGAN with Wasserstein loss performing mini-max adversarial training on MNIST digits and CelebA portrait datasets. |
| 23 | **[Speech to Text Transcription](AI%2C%20ML%2C%20%26%20DL/Speech_to_Text_Transcription)** | Vosk · PocketSphinx · Whisper | Unified multi-engine speech recognition SDK supporting zero-latency offline (Vosk/CMU) and cloud (Whisper) transcription. |
| 24 | **[Deep Learning Framework from Scratch](AI%2C%20ML%2C%20%26%20DL/Deep_Learning_Framework_Scratch)** | Pure Python · NumPy | Custom scalar autograd engine building DAG computational graphs, topological-sort reverse-mode backpropagation, and MLP API. |
| 25 | **[Reinforcement Learning Trading](AI%2C%20ML%2C%20%26%20DL/Reinforcement_Learning_Trading)** | Gymnasium · Stable-Baselines3 | PPO and DQN RL agents trained inside custom Gymnasium market environments to execute autonomous stock trading strategies. |
| 26 | **[YOLO Object Detection](AI%2C%20ML%2C%20%26%20DL/YOLO_Object_Detection)** | Ultralytics YOLOv8 · OpenCV | Real-time **60+ FPS** multi-object detection and tracking with bounding box localization and confidence-threshold NMS filtering. |
| 27 | **[Masked Language Model NLP](AI%2C%20ML%2C%20%26%20DL/Masked_Language_Model_NLP)** | HuggingFace Transformers · PyTorch | Bidirectional BERT masked token prediction fine-tuning pipeline adapted to custom scientific and legal text corpora. |
| 28 | **[Vector DB from Scratch](AI%2C%20ML%2C%20%26%20DL/Vector_DB_From_Scratch)** | Pure Python · NumPy | Custom vector database engine with high-dimensional CRUD operations, cosine/dot-product similarity scoring, and metadata filtering. |

---

## 📊 Data Science, Predictive Analytics & MLOps

> 4 high-rigor engineering projects applying statistical and deep learning methods to real-world industrial problems.

| # | Project | Stack | Impact & Highlights |
| :-: | :--- | :--- | :--- |
| 29 | **[CFD Surrogate Modeling](Data%20Science/CFD_Surrogate_Modeling)** | PyTorch · OpenFOAM · Keras · ConvLSTM | ConvLSTM model predicting OpenFOAM supersonic shockwave fields — velocity `U`, pressure `P`, temperature `T` — in real-time. |
| 30 | **[Predictive Maintenance IoT](Data%20Science/Predictive_Maintenance_IoT)** | Scikit-Learn · Pandas · NumPy | C-MAPSS turbofan engine RUL (Remaining Useful Life) prediction pipeline with sensor feature engineering and degradation modeling. |
| 31 | **[Financial Analysis Python](Data%20Science/financial_analysis_python)** | Pandas · NumPy · Matplotlib · SciPy | Markowitz Mean-Variance frontier optimization, Sharpe ratio tracking, Monte Carlo portfolio simulations, and downside risk analytics. |
| 32 | **[Search Engine from Scratch](Data%20Science/Search_Engine_From_Scratch)** | Pure Python · Math · Regex | Full IR engine: tokenization, stemming, inverted index construction, logarithmic TF-IDF weighting, and cosine vector space ranking. |

---

## ⚙️ Complete Technology Matrix

<table>
<tr>
<th>Domain</th>
<th>Technologies & Tools</th>
</tr>
<tr>
<td><strong>🤖 Agentic Frameworks</strong></td>
<td>

`LangGraph` `LangChain` `Google Agent Dev Kit (ADK)` `Agno (Phidata)` `OpenAI Swarm`

</td>
</tr>
<tr>
<td><strong>🧠 LLMs & Fine-Tuning</strong></td>
<td>

`OpenAI GPT-4o` `Google Gemini 2.5` `LLaMA-3` `Mistral AI` `Qwen` `QLoRA` `PEFT` `TRL` `BitsAndBytes`

</td>
</tr>
<tr>
<td><strong>📦 Vector Databases & RAG</strong></td>
<td>

`ChromaDB` `FAISS` `Tavily Search API` `Custom NumPy Vector DB`

</td>
</tr>
<tr>
<td><strong>🔥 Deep Learning & Neural Networks</strong></td>
<td>

`PyTorch` `PyTorch Geometric` `TensorFlow 2.x` `Keras` `Ultralytics YOLOv8` `HuggingFace Transformers`

</td>
</tr>
<tr>
<td><strong>⚛️ Physics & Engineering AI</strong></td>
<td>

`Physics-Informed Neural Networks (PINNs)` `OpenFOAM CFD` `CARLA Autonomous Driving Simulator` `SciPy FEA`

</td>
</tr>
<tr>
<td><strong>📊 Data Science & Classical ML</strong></td>
<td>

`Pandas` `NumPy` `Scikit-Learn` `SciPy` `XGBoost` `LightGBM` `Matplotlib` `Seaborn` `Plotly`

</td>
</tr>
<tr>
<td><strong>🎙️ Audio, Vision & Multimodal</strong></td>
<td>

`Whisper ASR` `Vosk` `CMU PocketSphinx` `OpenCV` `MTCNN` `Diffusers` `ModelScope`

</td>
</tr>
<tr>
<td><strong>🚀 Deployment & Infrastructure</strong></td>
<td>

`Docker` `FastAPI` `Streamlit` `Uvicorn` `Redis` `Google Firestore` `Jupyter Notebooks`

</td>
</tr>
<tr>
<td><strong>🔧 Developer Tooling</strong></td>
<td>

`Git` `GitHub` `Python 3.10+` `Conda` `pip` `pytest` `black` `ruff`

</td>
</tr>
</table>

---

## 🔬 Mathematical & Theoretical Foundations

This portfolio emphasizes understanding algorithms at a first-principles level. Key mathematical frameworks implemented from scratch:

### 1. Physics-Informed Neural Networks — Navier-Stokes Fluid Flow

Continuity and momentum conservation laws embedded as neural network loss terms:

```
Continuity:      ∂u/∂x  +  ∂v/∂y  =  0

Momentum (x):    ρ(u·∂u/∂x + v·∂u/∂y)  =  -∂p/∂x  +  μ·(∂²u/∂x² + ∂²u/∂y²)

Momentum (y):    ρ(u·∂v/∂x + v·∂v/∂y)  =  -∂p/∂y  +  μ·(∂²v/∂x² + ∂²v/∂y²)

Total Loss:      L_total = L_data  +  λ₁·L_continuity  +  λ₂·L_momentum
```

---

### 2. Maximal Marginal Relevance (MMR) — Diversity-Aware RAG Retrieval

Eliminates redundant document passages during retrieval by jointly maximising query relevance and minimising inter-document similarity:

```
MMR = argmax  [  λ · Sim(dᵢ, q)  −  (1−λ) · max Sim(dᵢ, dⱼ)  ]
       dᵢ ∈ R\S                            dⱼ ∈ S
```

Where `q` = query, `R` = candidate set, `S` = selected set, `λ` = diversity-relevance tradeoff.

---

### 3. TF-IDF Vector Space Model — Information Retrieval Search Engine

Scoring function for document retrieval over inverted index:

```
TF(t, d)  =  f(t,d) / Σ f(t',d)           [Term frequency normalisation]

IDF(t, D) =  log( |D| / (1 + |{d: t∈d}|) ) [Inverse document frequency]

Cosine(q, d) = Σ TF-IDF(t,q)·TF-IDF(t,d)  [Cosine vector space similarity]
               ─────────────────────────────
                        ||q|| · ||d||
```

---

### 4. Markowitz Mean-Variance Portfolio Optimisation

Minimum variance efficient frontier optimisation:

```
Minimise:   wᵀ Σ w          [Portfolio variance]

Subject to: wᵀ μ = μ_target [Target expected return constraint]
            wᵀ 1 = 1        [Full capital deployment constraint]
            w ≥ 0           [Long-only constraint (optional)]
```

Where `w` = asset weight vector, `Σ` = covariance matrix, `μ` = expected return vector.

---

### 5. Backpropagation via Reverse-Mode Automatic Differentiation (micrograd)

Scalar-valued autograd engine using topological sort for gradient computation:

```
Forward Pass:  Build DAG of Value nodes with operations
Topo Sort:     Order nodes from outputs → inputs (reverse DFS)
Backward Pass: Chain rule: ∂L/∂xᵢ = Σⱼ (∂L/∂yⱼ)·(∂yⱼ/∂xᵢ)
```

---

## 📂 Repository Structure

```
Data-Science-Portfolio/
│
├── 🤖 Generative AI/                               ← 17 Projects: GenAI, RAG, Agents, LLM Fine-tuning
│   │
│   ├── 01_Generative_AI_Basics/                    ← LLM chains, prompt templates, CineSage recommender
│   ├── 02_Document_RAG_ChromaDB/                   ← PDF ingestion, chunking, MMR vector retrieval
│   ├── 03_Custom_Tool_Calling_Agents/              ← Dynamic Python tool-binding agents
│   ├── 04_Multi_Agent_Research_System/             ← 4-agent Search→Read→Write→Critic swarm
│   ├── 05_AI_Video_Assistant_RAG/                  ← Whisper ASR + FAISS transcript semantic search
│   ├── 06_LangGraph_Agentic_Workflows/             ← Cyclic stateful graphs + human-in-the-loop nodes
│   │
│   ├── Agentic_AI_Projects/
│   │   ├── MangoOS/                                ← LangGraph Agentic AI Operating System
│   │   ├── Self_Extending_Agent/                   ← Runtime self-skill generation (Google ADK)
│   │   ├── Financial_Analysis_Agent_Crew/          ← Multi-agent stock analysis swarm (Agno)
│   │   ├── Self_Correcting_Data_Validation_Agent/  ← Self-healing schema drift detection
│   │   └── AI_Test_Case_Generator/                 ← Automated QA test suite generation
│   │
│   ├── LLM_Projects/
│   │   ├── Fine_Tuned_Domain_Chatbot/              ← LLaMA-3 QLoRA 4-bit instruction tuning
│   │   ├── Guardrail_Moderation_API/               ← PII/injection/toxicity API interceptor
│   │   └── Real_Time_Video_Summarizer/             ← Text-to-video synthesis pipeline
│   │
│   └── RAG_Projects/
│       ├── Advanced_Multi_Source_RAG/              ← Hybrid local+web RAG routing system
│       ├── Chat_With_Codebase/                     ← Codebase-level GPT-4o multi-agent dev assistant
│       └── Multimodal_Product_Assistant/           ← MedGemma image+text clinical diagnostics
│
├── 🧠 AI, ML, & DL/                               ← 11 Projects: CV, Physics AI, DL Engineering
│   │
│   ├── Autonomous_Vehicle_Simulation/              ← CARLA lane segmentation + PID/MPC steering
│   ├── Deep_Learning_Framework_Scratch/            ← Custom scalar autograd engine (micrograd)
│   ├── Deepfake_Detection/                         ← EfficientNet + MTCNN deepfake classifier
│   ├── Defect_Detection_Gear_Bearing/              ← Vibration signal CNN fault diagnosis
│   ├── FEA_Surrogate_Modeling/                     ← DNN FEA solver surrogate (1000x speedup)
│   ├── Generative_Art_GANs/                        ← DCGAN CelebA & MNIST adversarial synthesis
│   ├── Masked_Language_Model_NLP/                  ← BERT masked token prediction fine-tuning
│   ├── PINN_Laminar_Flow/                          ← Navier-Stokes PDE-constrained neural network
│   ├── Reinforcement_Learning_Trading/             ← PPO/DQN Gymnasium stock trading agent
│   ├── Speech_to_Text_Transcription/               ← Multi-engine ASR (Vosk, Whisper, CMU)
│   ├── Vector_DB_From_Scratch/                     ← NumPy cosine similarity vector database
│   └── YOLO_Object_Detection/                      ← YOLOv8 real-time 60+ FPS object tracking
│
├── 📊 Data Science/                                ← 4 Projects: Statistical ML, Engineering Analytics
│   │
│   ├── CFD_Surrogate_Modeling/                     ← ConvLSTM OpenFOAM shockwave field predictor
│   ├── Predictive_Maintenance_IoT/                 ← C-MAPSS turbofan RUL estimation pipeline
│   ├── financial_analysis_python/                  ← Markowitz MVE frontier & Sharpe ratio analytics
│   └── Search_Engine_From_Scratch/                 ← TF-IDF inverted index information retrieval engine
│
├── LICENSE                                         ← MIT License
└── README.md                                       ← This master portfolio document
```

---

## 🚀 Quick Start

### Step 1 — Clone the Portfolio
```bash
git clone https://github.com/Omgiri01/Data-Science-Portfolio.git
cd Data-Science-Portfolio
```

### Step 2 — Install Python Dependencies for Any Project
Each project folder is self-contained with its own `requirements.txt`:
```bash
pip install -r "Generative AI/02_Document_RAG_ChromaDB/requirements.txt"
```

### Step 3 — Launch a Streamlit Dashboard
```bash
streamlit run "Generative AI/04_Multi_Agent_Research_System/app.py"
```

### Step 4 — Run a Standalone Script / Benchmark
```bash
python "AI, ML, & DL/Vector_DB_From_Scratch/demo.py"
python "Data Science/Search_Engine_From_Scratch/main.py"
```

### Step 5 — Run a Jupyter Notebook
```bash
jupyter notebook "AI, ML, & DL/Deepfake_Detection/notebooks/"
```

---

## 👤 Author

<div align="center">

### Om Giri

*AI & Machine Learning Engineer — specialising in Agentic Systems, Physics-Informed Deep Learning, and Scientific ML*

[![GitHub](https://img.shields.io/badge/GitHub-@Omgiri01-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Omgiri01)
[![Portfolio](https://img.shields.io/badge/Repository-Data--Science--Portfolio-10B981?style=for-the-badge&logo=git&logoColor=white)](https://github.com/Omgiri01/Data-Science-Portfolio)

</div>

---

## 📄 License

This repository is released under the [MIT License](LICENSE).  
© 2026 Om Giri — All Rights Reserved.

---

<div align="center">

*If you found this portfolio valuable, consider giving it a ⭐ on [GitHub](https://github.com/Omgiri01/Data-Science-Portfolio)!*

</div>
