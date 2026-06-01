import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reranker import Reranker


@pytest.fixture
def reranker():
    return Reranker("sentence-transformers/msmarco-distilbert-base-v3")


@pytest.fixture
def sample_docs():
    return [
        "RLHF is a technique for aligning language models with human preferences.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Proximal Policy Optimization is used in reinforcement learning.",
        "The Eiffel Tower is located in Paris, France.",
        "Fine-tuning with human feedback improves response quality.",
    ]


class TestReranker:
    def test_rerank_returns_correct_count(self, reranker, sample_docs):
        results = reranker.rerank(sample_docs, "What is RLHF?", top_n=3)
        assert len(results) == 3

    def test_rerank_returns_tuples(self, reranker, sample_docs):
        results = reranker.rerank(sample_docs, "reinforcement learning", top_n=2)
        for doc, score in results:
            assert isinstance(doc, str)
            assert float(score) >= 0

    def test_rerank_most_relevant_first(self, reranker, sample_docs):
        results = reranker.rerank(sample_docs, "What is RLHF?", top_n=5)
        scores = [float(s) for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_top_n_greater_than_docs(self, reranker, sample_docs):
        results = reranker.rerank(sample_docs, "test", top_n=100)
        assert len(results) == len(sample_docs)
