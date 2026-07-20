# 🧠 Parameter-Efficient Fine-Tuning (PEFT) & LLaMA 3 Alignment

This project provides end-to-end notebooks demonstrating how to fine-tune LLaMA 3 on domain-specific instruction datasets. It utilizes parameter-efficient techniques such as **LoRA/QLoRA** and direct preference alignment using **ORPO (Odds Ratio Preference Optimization)**.

---

## 📂 Key Notebooks

- **`llama3_fine_tuning.ipynb`**:
  - Implements QLoRA parameter-efficient fine-tuning on custom instruction datasets.
  - Integrates Hugging Face **PEFT**, **TRL (Transformer Reinforcement Learning)**, and **SFT (Supervised Fine-Tuning)** engines.
  - Implements odds-ratio preference optimizations (ORPO) to align responses without a separate RLHF reward model.

---

## 🛠️ Tech Stack & Requirements

- **Models**: Meta LLaMA 3 (8B Instruct)
- **Frameworks**: PyTorch, PEFT, TRL, Transformers, Accelerate
- **Quantization**: BitsAndBytes (4-bit/8-bit precision loading)
- **Tracking**: Weights & Biases (Wandb)

---

## 🚀 Usage Guide

1. Ensure PyTorch with CUDA is installed:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. Install PEFT and Hugging Face dependencies:
   ```bash
   pip install transformers peft trl bitsandbytes accelerate datasets
   ```
3. Open `llama3_fine_tuning.ipynb` in Jupyter/Google Colab and run all cells to start the instruction-tuning loop.
