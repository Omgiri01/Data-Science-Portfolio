# 🔍 Information Retrieval & TF-IDF Search Engine from Scratch

**Author:** Om Giri ([@Omgiri01](https://github.com/Omgiri01))  
**Category:** Data Science & Search Engineering  

---

## 📌 Executive Summary

This project implements a full **Information Retrieval (IR) and Inverted Index Search Engine** from first principles in pure Python. It features custom text tokenization, stop-word filtering, inverted index construction, Term Frequency-Inverse Document Frequency (TF-IDF) logarithmic weighting, and Vector Space Model (VSM) cosine similarity ranking.

---

## 🛠️ Mathematical Model & Architecture

```
+--------------------------+      +---------------------------------+      +-----------------------+
| Raw Text Document Corpus | ---> | Tokenization & Normalization    | ---> | Inverted Index Table  |
| (Files, Sentences)       |      | (Regex, Lowercasing, Stopwords) |      | (Term -> [Doc IDs])   |
+--------------------------+      +---------------------------------+      +-----------------------+
                                                                                       |
                                                                                       v
+--------------------------+      +---------------------------------+      +-----------------------+
| Ranked Search Results    | <--- | Vector Space Cosine Similarity  | <--- | TF-IDF Weight Matrix  |
| (Top-K Matching Docs)    |      | Sim(Q, D) = (Q · D) / ||Q||||D|||      | TF(t,d) * IDF(t,D)    |
+--------------------------+      +---------------------------------+      +-----------------------+
```

### Mathematical Foundations:

1. **Term Frequency ($\text{TF}$)**:
   $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

2. **Inverse Document Frequency ($\text{IDF}$)**:
   $$\text{IDF}(t, D) = \log \left( \frac{|D|}{1 + |\{d \in D : t \in d\}|} \right)$$

3. **TF-IDF Weight**:
   $$\text{W}(t, d) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

4. **Cosine Similarity Scoring**:
   $$\text{Score}(q, d) = \frac{\sum_{t \in q \cap d} \text{W}(t, q) \cdot \text{W}(t, d)}{\sqrt{\sum_{t \in q} \text{W}(t, q)^2} \sqrt{\sum_{t \in d} \text{W}(t, d)^2}}$$

---

## 📁 Repository Structure

```
Search_Engine_From_Scratch/
├── engine.py              # Inverted index building, TF-IDF engine, and vector similarity
├── demo.py                # Demonstration script indexing sample corpus and executing queries
└── README.md              # Documentation
```

---

## 🚀 Setup & Execution

### 1. Requirements
No external packages are required. Built with standard Python 3.8+ libraries (`math`, `re`, `collections`).

### 2. Running the Search Engine Demo
```bash
python demo.py
```

### 3. Using as a Library
```python
from engine import SearchEngine

corpus = {
    "doc1": "Deep learning models are revolutionizing artificial intelligence.",
    "doc2": "Data science relies on statistical analysis and machine learning.",
    "doc3": "Artificial intelligence applications include natural language processing."
}

engine = SearchEngine()
engine.fit(corpus)

results = engine.search("artificial intelligence models", top_k=2)
for doc_id, score in results:
    print(f"Doc ID: {doc_id} | Score: {score:.4f}")
```

---

## 📈 Key Advantages

- **Zero Third-Party Dependencies**: Native Python implementation.
- **Sub-linear Query Execution**: Inverted index lookups bypass full corpus scanning.
- **Normalized Vector Scores**: Returns cosine similarity bounded strictly in $[0.0, 1.0]$.
