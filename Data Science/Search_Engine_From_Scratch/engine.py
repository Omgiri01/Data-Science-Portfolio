import math
import re
from collections import Counter
from typing import List, Dict, Tuple

class TFIDFSearchEngine:
    def __init__(self):
        self.documents: Dict[int, str] = {}  # doc_id -> raw_text
        self.doc_tokens: Dict[int, List[str]] = {}  # doc_id -> list of tokens
        self.inverted_index: Dict[str, List[int]] = {}  # term -> list of doc_ids
        self.vocab = set()
        self.num_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        """Cleans and tokenizes text into lower-case alphanumeric words."""
        return re.findall(r'\b\w+\b', text.lower())

    def add_document(self, doc_id: int, text: str):
        """Indexes a document into the search engine."""
        tokens = self._tokenize(text)
        self.documents[doc_id] = text
        self.doc_tokens[doc_id] = tokens
        self.num_docs += 1
        
        # Build inverted index
        unique_tokens = set(tokens)
        for term in unique_tokens:
            self.vocab.add(term)
            if term not in self.inverted_index:
                self.inverted_index[term] = []
            self.inverted_index[term].append(doc_id)

    def _get_tf(self, term: str, doc_tokens: List[str]) -> float:
        """Computes Term Frequency (TF) using log normalization."""
        count = doc_tokens.count(term)
        if count == 0:
            return 0.0
        return 1.0 + math.log10(count)

    def _get_idf(self, term: str) -> float:
        """Computes Inverse Document Frequency (IDF)."""
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq == 0:
            return 0.0
        return math.log10(self.num_docs / doc_freq)

    def _get_doc_vector(self, doc_id: int) -> Dict[str, float]:
        """Computes the TF-IDF vector representing a document."""
        tokens = self.doc_tokens[doc_id]
        vector = {}
        for term in set(tokens):
            tf = self._get_tf(term, tokens)
            idf = self._get_idf(term)
            vector[term] = tf * idf
        return vector

    def search(self, query: str) -> List[Tuple[int, float]]:
        """
        Executes a vector space search for the query, returning ranked doc_ids and scores.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # 1. Compute query TF-IDF vector
        query_counts = Counter(query_tokens)
        query_vector = {}
        for term in query_counts:
            if term in self.vocab:
                tf = 1.0 + math.log10(query_counts[term])
                idf = self._get_idf(term)
                query_vector[term] = tf * idf

        if not query_vector:
            return []

        query_norm = math.sqrt(sum(val ** 2 for val in query_vector.values()))
        if query_norm == 0:
            return []

        results = []

        # 2. Score candidate documents containing query terms
        candidate_docs = set()
        for term in query_vector:
            candidate_docs.update(self.inverted_index.get(term, []))

        for doc_id in candidate_docs:
            doc_vector = self._get_doc_vector(doc_id)
            
            # Compute cosine similarity
            dot_product = sum(query_vector[term] * doc_vector.get(term, 0.0) for term in query_vector)
            doc_norm = math.sqrt(sum(val ** 2 for val in doc_vector.values()))
            
            if doc_norm > 0:
                score = dot_product / (query_norm * doc_norm)
                results.append((doc_id, float(score)))

        # Sort by relevance score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results
