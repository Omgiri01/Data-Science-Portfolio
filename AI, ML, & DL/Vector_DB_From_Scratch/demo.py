import numpy as np
from vectordb import SimpleVectorDB

# Helper function to generate mock embeddings for demonstration without external API keys.
# In a production environment, you would use sentence-transformers, OpenAI, or Ollama embeddings.
def generate_mock_embedding(text: str, dimension: int = 8) -> list:
    """
    Generates a deterministic mock embedding for demonstration purposes.
    Simply sums characters and generates a pseudo-random normalized vector.
    """
    np.random.seed(sum(ord(c) for c in text))
    vec = np.random.randn(dimension)
    normalized = vec / np.linalg.norm(vec)
    return normalized.tolist()

def main():
    print("=== Initializing Custom Vector Database ===")
    db_file = "demo_vectors.json"
    db = SimpleVectorDB(storage_path=db_file)

    # Documents to index
    documents = [
        {"id": "doc1", "text": "Deep learning is a subset of machine learning.", "category": "AI", "author": "Alice"},
        {"id": "doc2", "text": "Stock market trading relies on advanced statistical forecasting.", "category": "Finance", "author": "Bob"},
        {"id": "doc3", "text": "Reinforcement learning trains agents through reward systems.", "category": "AI", "author": "Charlie"},
        {"id": "doc4", "text": "Quantitative finance uses mathematical models to price derivatives.", "category": "Finance", "author": "Alice"},
        {"id": "doc5", "text": "Large Language Models are changing how we do research.", "category": "AI", "author": "Bob"},
    ]

    print("\n--- Indexing Documents ---")
    for doc in documents:
        vector = generate_mock_embedding(doc["text"])
        metadata = {"text": doc["text"], "category": doc["category"], "author": doc["author"]}
        db.insert(doc["id"], vector, metadata)
        print(f"Indexed: [{doc['id']}] - Category: {doc['category']} - Text: {doc['text']}")

    # Let's perform a query
    query_text = "I want to learn about neural networks and artificial intelligence"
    query_vec = generate_mock_embedding(query_text)
    
    print(f"\n--- Performing Query: '{query_text}' (Top 3) ---")
    results = db.query(query_vec, k=3)
    for doc_id, score, meta in results:
        print(f"Match: [{doc_id}] | Similarity: {score:.4f} | Category: {meta['category']} | Text: {meta['text']}")

    # Query with Metadata Filter
    print(f"\n--- Performing Query with Filter: Category = 'Finance' ---")
    results_filtered = db.query(query_vec, k=3, filter_dict={"category": "Finance"})
    for doc_id, score, meta in results_filtered:
        print(f"Match: [{doc_id}] | Similarity: {score:.4f} | Category: {meta['category']} | Text: {meta['text']}")

if __name__ == "__main__":
    main()
