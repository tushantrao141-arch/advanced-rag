import math
from typing import List


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(top_k & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance_scores: List[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    actual_relevance = [1.0 if doc in relevant_set else 0.0 for doc in retrieved[:k]]
    ideal_relevance = sorted(actual_relevance, reverse=True)

    actual_dcg = dcg_at_k(actual_relevance, k)
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_retrieval(
    retrieved: List[str],
    relevant: List[str],
    k_values: List[int] = None,
) -> dict:
    if k_values is None:
        k_values = [1, 5, 10, 20]

    results = {
        "mrr": mean_reciprocal_rank(retrieved, relevant),
    }

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)

    return results
