import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate import recall_at_k, mean_reciprocal_rank, ndcg_at_k, evaluate_retrieval


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_recall(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "b", "y"]
        relevant = ["a", "b", "c"]
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=2) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank(["x", "a", "c"], ["a"]) == 0.5

    def test_not_found(self):
        assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c"]
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_relevance(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert ndcg_at_k(retrieved, relevant, k=3) == 0.0


class TestEvaluateRetrieval:
    def test_returns_all_metrics(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "c"]
        results = evaluate_retrieval(retrieved, relevant, k_values=[1, 5])
        assert "mrr" in results
        assert "recall@1" in results
        assert "recall@5" in results
        assert "ndcg@1" in results
        assert "ndcg@5" in results
