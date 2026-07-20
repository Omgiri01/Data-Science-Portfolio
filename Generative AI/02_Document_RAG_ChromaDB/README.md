# Enterprise Document RAG System with ChromaDB & LangChain

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Generative AI & Large Language Models  

---

## 📌 Executive Summary

This project implements an end-to-end **Retrieval-Augmented Generation (RAG) System** for ingesting, indexing, and querying PDF documents and books locally. Built using **LangChain**, **ChromaDB**, **OpenAI Embeddings**, and **Mistral AI**, it features Maximal Marginal Relevance (MMR) retrieval search to eliminate redundant document context and hallucination-guarded prompt orchestration.

---

## 🛠️ System Architecture

```
+-------------------+      +---------------------------------+      +-----------------------+
|  User PDF Upload  | ---> | PyPDFLoader & Text Chunking     | ---> | OpenAI Embeddings     |
|   (Books, Papers) |      | (Chunk: 1000, Overlap: 200)    |      | (text-embedding-3)    |
+-------------------+      +---------------------------------+      +-----------------------+
                                                                                |
                                                                                v
+-------------------+      +---------------------------------+      +-----------------------+
| Streamlit Web UI  | <--- | Context-Constrained LLM Answer  | <--- | ChromaDB Vector Store |
| (Ask Questions)   |      | (Mistral AI / GPT-4o Prompt)    |      | (MMR Search Engine)   |
+-------------------+      +---------------------------------+      +-----------------------+
```

1. **Document Processing & Chunking (`PyPDFLoader`)**:
   - Parses complex PDF files and breaks text into 1,000-character windows with a 200-character rolling overlap to preserve cross-chunk context boundaries.
2. **Vector Indexing (`ChromaDB`)**:
   - Encodes text chunks using OpenAI Embeddings and persists the vector representations to a local `chroma_db/` collection directory.
3. **Maximal Marginal Relevance (MMR) Search**:
   - Retrieves the top $k=4$ non-redundant passages by optimizing diversity ($\lambda_{mult}=0.5$) across $fetch\_k=10$ candidate matches.
4. **Strict Context-Bound Prompting**:
   - Instructs the LLM to rely strictly on retrieved context, gracefully returning fallback notices when information is absent.

---

## 📁 Repository Structure

```
02_Document_RAG_ChromaDB/
├── app.py                 # Interactive Streamlit Web Dashboard application
├── create_database.py     # Batch CLI script to initialize vector database
├── main.py                # Standalone script for document indexing & test queries
├── requirements.txt       # Project dependencies manifest
├── document loaders/      # Sample PDF documents
├── retrievers/            # Retrieval strategy definitions
└── vector store/          # Vector persistence helpers
```

---

## 🚀 Getting Started

### 1. Prerequisites
Install required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project directory with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

### 3. Launching the Streamlit Application
Run the interactive web interface:
```bash
streamlit run app.py
```

---

## 📈 Key Features & Capabilities

- **MMR Search Tuning**: Balance relevance vs. diversity in document context retrieval.
- **Persistent Vector Storage**: Save embedding indexes locally for zero-delay startup on subsequent sessions.
- **Zero-Hallucination Guard**: Prompt template forces explicit system fallback if the document lacks requested facts.
