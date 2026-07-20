import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple

class SimpleVectorDB:
    def __init__(self, storage_path: str = "vector_store.json"):
        """
        Initializes a lightweight Vector Database from scratch.
        
        Args:
            storage_path (str): Filepath to persist vectors and metadata.
        """
        self.storage_path = storage_path
        self.vectors: Dict[str, np.ndarray] = {}  # doc_id -> numpy embedding vector
        self.metadata: Dict[str, Dict[str, Any]] = {}  # doc_id -> metadata dictionary
        self.load()

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes the cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def insert(self, doc_id: str, vector: List[float], metadata: Dict[str, Any] = None) -> None:
        """
        Inserts a vector and its associated metadata into the database.
        """
        self.vectors[doc_id] = np.array(vector, dtype=np.float32)
        self.metadata[doc_id] = metadata or {}
        self.save()

    def query(self, query_vector: List[float], k: int = 3, filter_dict: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Queries the database for top K most similar vectors, applying optional metadata filters.
        
        Args:
            query_vector (List[float]): Target query vector.
            k (int): Number of top results to return.
            filter_dict (Dict[str, Any]): Metadata fields and values that must match.
            
        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: List of (doc_id, similarity_score, metadata).
        """
        q_vec = np.array(query_vector, dtype=np.float32)
        results = []

        for doc_id, vec in self.vectors.items():
            meta = self.metadata[doc_id]
            
            # Apply metadata filtering
            if filter_dict:
                match = True
                for key, val in filter_dict.items():
                    if meta.get(key) != val:
                        match = False
                        break
                if not match:
                    continue

            similarity = self._cosine_similarity(q_vec, vec)
            results.append((doc_id, similarity, meta))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def save(self) -> None:
        """Persists the vectors and metadata to disk."""
        data_to_save = {}
        for doc_id, vec in self.vectors.items():
            data_to_save[doc_id] = {
                "vector": vec.tolist(),
                "metadata": self.metadata[doc_id]
            }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2)

    def load(self) -> None:
        """Loads vectors and metadata from disk if the file exists."""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for doc_id, item in data.items():
                    self.vectors[doc_id] = np.array(item["vector"], dtype=np.float32)
                    self.metadata[doc_id] = item["metadata"]
        except Exception as e:
            print(f"Error loading vector database: {e}")
