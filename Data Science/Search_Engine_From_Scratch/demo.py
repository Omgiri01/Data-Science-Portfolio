from engine import TFIDFSearchEngine

def main():
    print("=== TF-IDF Search Engine from Scratch ===")
    engine = TFIDFSearchEngine()

    # Simple corpus of documents
    corpus = {
        1: "The quick brown fox jumps over the lazy dog.",
        2: "Deep learning models require high performance GPU compute power.",
        3: "Search engines use term frequency and inverse document frequency (TF-IDF) algorithms.",
        4: "Machine learning algorithms utilize statistics to identify patterns in big data.",
        5: "A vector database computes cosine similarity between high-dimensional text embeddings."
    }

    # Index corpus
    print("\n--- Indexing Documents ---")
    for doc_id, text in corpus.items():
        engine.add_document(doc_id, text)
        print(f"Indexed [{doc_id}]: {text}")

    # Query 1
    query_1 = "machine learning data statistics"
    print(f"\n--- Searching Query: '{query_1}' ---")
    results = engine.search(query_1)
    for doc_id, score in results:
        print(f"Doc ID: {doc_id} | TF-IDF Cosine Score: {score:.4f} | Text: {corpus[doc_id]}")

    # Query 2
    query_2 = "high-dimensional GPU embeddings vector"
    print(f"\n--- Searching Query: '{query_2}' ---")
    results_2 = engine.search(query_2)
    for doc_id, score in results_2:
        print(f"Doc ID: {doc_id} | TF-IDF Cosine Score: {score:.4f} | Text: {corpus[doc_id]}")

if __name__ == "__main__":
    main()
