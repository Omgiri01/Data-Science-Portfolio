import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# We use a lightweight local embeddings provider or mock-fallback to ensure it runs out-of-the-box
from langchain_core.embeddings import Embeddings

class SimpleHashEmbeddings(Embeddings):
    """
    A lightweight, deterministic embedding model using hashing.
    Ensures zero-dependency execution without requiring OpenAI/HuggingFace keys,
    while maintaining the exact interface required by LangChain's vector stores.
    """
    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        # Simple deterministic hash-based vector generation
        state = sum(ord(c) for c in text)
        np_rand = []
        for i in range(self.dimension):
            state = (state * 1103515245 + 12345) & 0x7fffffff
            np_rand.append((state / 0x7fffffff) * 2.0 - 1.0)
        return np_rand

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

class VideoRAGEngine:
    def __init__(self):
        self.embeddings = SimpleHashEmbeddings()
        self.vector_store = None
        self.transcript_segments = []

    def load_transcript(self, transcript_text: str):
        """Loads and splits the transcript text into chunk segments for RAG."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        # Create documents
        docs = text_splitter.create_documents(
            texts=[transcript_text],
            metadatas=[{"source": "video_audio"}]
        )
        
        # Build FAISS vector store
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.transcript_segments = [doc.page_content for doc in docs]
        return len(docs)

    def query(self, user_query: str, k: int = 2) -> List[Dict[str, Any]]:
        """Queries the vector store for segments relevant to the user query."""
        if not self.vector_store:
            return []
        
        docs_and_scores = self.vector_store.similarity_search_with_score(user_query, k=k)
        
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        return results
