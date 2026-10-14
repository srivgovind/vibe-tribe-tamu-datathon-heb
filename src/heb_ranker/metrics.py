"""Ranking metrics and the datathon's composite score.

composite = 0.30 * nDCG@10 + 0.30 * AP@20 + 0.25 * Recall@30 + 0.15 * Precision@10
Graded relevance is 0-3; a product counts as relevant for AP/P/R when grade >= 1.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

WEIGHTS = {"nDCG@10": 0.30, "AP@20": 0.30, "R@30": 0.25, "P@10": 0.15}


def ndcg_at_k(grades: list[int], k: int) -> float:
    """nDCG over the returned list, matching the official scorer (ideal = best
    ordering of the grades that were returned)."""

    def dcg(g: list[int]) -> float:
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(g[:k]))

    ideal = dcg(sorted(grades, reverse=True))
    return dcg(grades) / ideal if ideal > 0 else 0.0


def average_precision_at_k(hits: list[int], n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    found, total = 0, 0.0
    for i, h in enumerate(hits[:k], start=1):
        if h:
            found += 1
            total += found / i
    return total / n_relevant


def precision_at_k(hits: list[int], k: int) -> float:
    top = hits[:k]
    return sum(top) / len(top) if top else 0.0


def recall_at_k(hits: list[int], n_relevant: int, k: int) -> float:
    return sum(hits[:k]) / n_relevant if n_relevant else 0.0


def build_qrels(labels: list[dict]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in labels:
        qrels[row["query_id"]][row["product_id"]] = row["relevance"]
    return qrels


def score_query(ranked: list[str], judgments: dict[str, int]) -> dict[str, float]:
    grades = [judgments.get(pid, 0) for pid in ranked]
    hits = [1 if g >= 1 else 0 for g in grades]
    n_rel = sum(1 for g in judgments.values() if g >= 1)
    m = {
        "nDCG@10": ndcg_at_k(grades, 10),
        "AP@20": average_precision_at_k(hits, n_rel, 20),
        "P@10": precision_at_k(hits, 10),
        "R@30": recall_at_k(hits, n_rel, 30),
    }
    m["composite"] = sum(WEIGHTS[name] * m[name] for name in WEIGHTS)
    return m


def evaluate(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Mean of each metric over the queries in `rankings`."""
    per_query = [score_query(ranked, qrels.get(qid, {})) for qid, ranked in rankings.items()]
    out = {name: round(mean(q[name] for q in per_query), 4) for name in per_query[0]}
    out["queries"] = len(per_query)
    return out
