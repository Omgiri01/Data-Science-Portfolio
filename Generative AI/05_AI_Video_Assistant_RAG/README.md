# 🎬 AI Video Assistant & Multimodal Transcript RAG Platform

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Generative AI & Large Language Models  

---

## 📌 Executive Summary

This project implements a complete local **Retrieval-Augmented Generation (RAG) System** designed to index, search, and answer complex queries over video content and meeting transcripts. By combining automated speech transcription, vector embedding indexing with **FAISS**, and context-aware LLM generation via **LangChain**, this system enables instant information retrieval across long video recordings.

---

## 🛠️ System Architecture

```
+--------------------------+      +---------------------------------+      +-----------------------+
|  Video / Audio Input     | ---> | Speech-to-Text Transcription    | ---> | Recursive Text Split  |
|  (MP4, WAV, MP3)         |      | (Whisper ASR Engine)            |      | (Chunk: 500, Overlap) |
+--------------------------+      +---------------------------------+      +-----------------------+
                                                                                       |
                                                                                       v
+--------------------------+      +---------------------------------+      +-----------------------+
| Interactive Web Query    | <--- | Contextual Answer Synthesis     | <--- | FAISS Vector Store    |
| (Streamlit Dashboard)    |      | (OpenAI / Gemini LLM)           |      | (Similarity Search)   |
+--------------------------+      +---------------------------------+      +-----------------------+
```

1. **Audio Extraction & ASR**: Converts incoming video streams into clean timestamped text transcripts.
2. **Chunking & Indexing**: Segments long transcripts into semantic chunks and encodes them into dense embedding space.
3. **Similarity Search Engine (`FAISS`)**: Performs ultra-fast vector similarity lookup to retrieve relevant transcript passages matching user queries.
4. **Interactive Assistant Interface (`app.py`)**: Renders a Streamlit interface for natural language Q&A against any video content.

---

## 📁 Repository Structure

```
05_AI_Video_Assistant_RAG/
├── app.py                 # Interactive Streamlit application interface
├── core/                  # Core RAG pipeline and transcript processing logic
├── utils/                 # Audio extraction & text parsing utility functions
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Install required Python packages:
```bash
pip install streamlit langchain langchain-community faiss-cpu openai python-dotenv
```

### 2. Environment Setup
Add your API keys to a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
```

### 3. Run the Application
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📈 Key Features & Capabilities

- **Timestamp Alignment**: Pinpoints exact video segments containing requested answers.
- **Fast Similarity Search**: Uses FAISS in-memory indexing for sub-millisecond query responses.
- **Context-Guarded Generation**: Ensures answers are anchored strictly in video transcript evidence.
