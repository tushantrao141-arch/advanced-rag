import sys
import os
import numpy as np
import faiss
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embeddings import Embeddings
from retriever import Retriever


@pytest.fixture
def setup_faiss():
    dim = 768
    model = Embeddings("all-mpnet-base-v2")

    texts = [
        "Reinforcement learning from human feedback improves model alignment.",
        "Transformers use self-attention to process sequential data efficiently.",
        "FAISS enables fast approximate nearest neighbor search.",
        "Gradient descent optimizes neural network parameters iteratively.",
        "Large language models generate text using autoregressive decoding.",
    ]

    embeddings = model.get_embedding(texts)
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    documents = [{"text": t} for t in texts]

    return documents, model, index


class TestRetriever:
    def test_search_returns_correct_count(self, setup_faiss):
        documents, model, index = setup_faiss
        results = Retriever.search(documents, model, index, "What is RLHF?", top_k=3)
        assert len(results) == 3

    def test_search_returns_strings(self, setup_faiss):
        documents, model, index = setup_faiss
        results = Retriever.search(documents, model, index, "attention mechanism", top_k=2)
        assert all(isinstance(r, str) for r in results)

    def test_search_top_1_is_most_relevant(self, setup_faiss):
        documents, model, index = setup_faiss
        results = Retriever.search(documents, model, index, "nearest neighbor search", top_k=1)
        assert "FAISS" in results[0] or "neighbor" in results[0].lower()

    def test_search_top_k_greater_than_docs(self, setup_faiss):
        documents, model, index = setup_faiss
        results = Retriever.search(documents, model, index, "test query", top_k=100)
        assert len(results) <= len(documents)
