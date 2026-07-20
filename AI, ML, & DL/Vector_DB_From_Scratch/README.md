# Custom Vector Database Engine from Scratch

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Artificial Intelligence, Machine Learning & Deep Learning  

---

## 📌 Executive Summary

This project implements a lightweight, pure Python/NumPy **Vector Database Engine** built from first principles. It supports high-dimensional embedding storage, dynamic metadata filtering, similarity search algorithms (Cosine Similarity & Euclidean distance), and JSON/binary disk persistence. This project demonstrates fundamental mechanics behind modern vector stores (like ChromaDB, Pinecone, and FAISS) without relying on heavy external database frameworks.

---

## 🛠️ System Architecture & Similarity Math

```
+-----------------------------------+
| Input Query Vector + Filter Specs |
+-----------------------------------+
                  |
                  v
+-----------------------------------+
|  Metadata Filter Evaluation Engine|
+-----------------------------------+
                  |
                  v
+-----------------------------------+
| Cosine Similarity Vector Matrix   |
|   Sim(v1, v2) = (v1 · v2) / ||v1|| ||v2|| |
+-----------------------------------+
                  |
                  v
+-----------------------------------+
| Top-K Max-Heap / Priority Sorting |
+-----------------------------------+
                  |
                  v
+-----------------------------------+
| Ranked Matches & Metadata Output  |
+-----------------------------------+
```

### Mathematical Formulation
For any query vector $\mathbf{q} \in \mathbb{R}^d$ and stored vector $\mathbf{x}_i \in \mathbb{R}^d$:

$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{x}_i) = \frac{\mathbf{q} \cdot \mathbf{x}_i}{\|\mathbf{q}\|_2 \|\mathbf{x}_i\|_2} = \frac{\sum_{j=1}^d q_j x_{ij}}{\sqrt{\sum_{j=1}^d q_j^2} \sqrt{\sum_{j=1}^d x_{ij}^2}}$$

---

## 📁 Repository Structure

```
Vector_DB_From_Scratch/
├── vectordb.py            # Core SimpleVectorDB class implementation
├── demo.py                # Demonstration script indexing and querying vectors
└── demo_vectors.json      # Sample persisted vector store database file
```

---

## 🚀 Getting Started

### 1. Prerequisites
Only basic standard dependencies are required:
```bash
pip install numpy
```

### 2. Basic Usage Example

```python
from vectordb import SimpleVectorDB

# Initialize local vector database instance
db = SimpleVectorDB(storage_path="my_vectors.json")

# Insert documents with embedding vectors and metadata
db.insert(
    doc_id="doc_1", 
    vector=[0.12, 0.85, 0.43, 0.91], 
    metadata={"category": "AI", "year": 2026}
)
db.insert(
    doc_id="doc_2", 
    vector=[0.88, 0.10, 0.05, 0.22], 
    metadata={"category": "Finance", "year": 2025}
)

# Query top-K matches with optional metadata filtering
results = db.query(
    query_vector=[0.10, 0.80, 0.40, 0.90], 
    k=1, 
    filter_dict={"category": "AI"}
)

for doc_id, score, meta in results:
    print(f"Match ID: {doc_id} | Similarity Score: {score:.4f} | Meta: {meta}")
```

### 3. Running the Demo Script
Execute the interactive demonstration:
```bash
python demo.py
```

---

## 📈 Key Features & Performance Metrics

- **Zero Heavy Dependencies**: Pure NumPy linear algebra implementation.
- **Fast In-Memory Lookup**: Vectorized matrix transformations for low-latency similarity comparisons.
- **Persistent Disk Serialization**: Automatic state saving/loading using structured JSON serialization.
- **Metadata Filtering**: Pre-filtering query space before metric computation.
