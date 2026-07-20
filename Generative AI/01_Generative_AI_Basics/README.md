# 🎬 CineSage & Generative AI Basics

A beginner-to-advanced project demonstrating the core building blocks of Generative AI, including Chat Models, Hugging Face integrations, local LLM execution, and a Streamlit-based movie recommendation system called **CineSage**.

## 📂 Project Structure
- `chatmodels/`: Contains model exploration scripts (`chat.py`, `chatbot.py`, `huggingface.py`, `localmodel.py`).
- `chatmodels/UIchatbot.py`: A Streamlit interface for general conversation.
- `CineSage/`: The main movie recommendation engine (`core.py`, `UICore.py`).
- `embeddingmodels/`: Embedding scripts (`embeddings.py`, `huggingface_embedding.py`).
- `requirements.txt`: Project dependencies.

## 🚀 Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the chatbot UI:
   ```bash
   streamlit run chatmodels/UIchatbot.py
   ```
3. Run the CineSage movie recommender:
   ```bash
   streamlit run CineSage/UICore.py
   ```
