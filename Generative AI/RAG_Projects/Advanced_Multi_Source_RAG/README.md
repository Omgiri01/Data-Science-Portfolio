# 🧠 Agentic Multi-Source RAG with Gemini Flash Thinking

An advanced, stateful Agentic Retrieval-Augmented Generation (RAG) system utilizing **Gemini 2.0 Flash Thinking** for multi-hop reasoning over complex inputs. This agent ingests files and URLs, dynamically rewrites queries, and falls back to neural search engines to supply cited, search-grounded answers.

---

## 🛠️ System Architecture & Reasoning Loop

```
           ┌───────────────────────┐
           │   User Input Query    │
           └──────────┬────────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │ Query Rewriter Agent  │  ◄── Reformulates query for search
           └──────────┬────────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │  Chroma Vector Store  │  ◄── Tries to find local documents
           └──────────┬────────────┘
                      │
             Found? ──┴── Not Found?
             │           │
             ▼           ▼
      [Retrieve Chunks] [Neural Web Search (Exa)]
             │           │
             └─────┬─────┘
                   │
                   ▼
           ┌───────────────────────┐
           │  Gemini Flash Model   │  ◄── Deep reasoning & synthesis
           │   (Thinking Mode)     │
           └──────────┬────────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │ Cited Markdown Answer │
           └───────────────────────┘
```

1. **Ingestion**: Ingests files and web pages, splits them into semantic chunks, and creates vectors using Gemini Embeddings stored in ChromaDB.
2. **Query Rewrite**: An intent evaluation node reformulates conversational inputs into optimized keyword vectors.
3. **Retrieval**: Queries the local vector store; if coverage is insufficient, it invokes the **Exa.ai Neural Search** engine to fetch clean web pages.
4. **Synthesis**: The `gemini-2.0-flash-thinking-exp` model reasons over the combined context to construct a markdown response containing source citations.

---

## ✨ Key Features

- **Gemini Thinking Engine**: Long-context reasoning that evaluates conflicting reports and formats multi-hop arguments.
- **Neural Web Fallback**: Integrates Exa API for clean, semantic web results.
- **Interactive Dashboard**: Streamlit interface to upload files, index webpages, and view search histories.

---

## 🚀 Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install streamlit agno google-generativeai chromadb bs4 langchain-community
   ```

2. **Configure API Keys**:
   Export keys in your environment:
   ```bash
   export GEMINI_API_KEY="your_gemini_key"
   export EXA_API_KEY="your_exa_key"
   ```

3. **Start Streamlit Dashboard**:
   ```bash
   streamlit run agentic_rag.py
   ```
