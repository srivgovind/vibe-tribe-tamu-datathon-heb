import math

import pytest

from heb_ranker.metrics import (
    average_precision_at_k,
    build_qrels,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_query,
)


def test_ndcg_perfect_and_reversed():
    assert ndcg_at_k([3, 2, 1, 0], 10) == pytest.approx(1.0)
    worst = ndcg_at_k([0, 1, 2, 3], 10)
    assert 0 < worst < 1


def test_ndcg_hand_computed():
    # DCG = (2^1-1)/log2(2) + (2^3-1)/log2(3); ideal = 7/1 + 1/log2(3)
    expected = (1 + 7 / math.log2(3)) / (7 + 1 / math.log2(3))
    assert ndcg_at_k([1, 3], 10) == pytest.approx(expected)


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k([0, 0, 0], 10) == 0.0


def test_average_precision():
    # hits at ranks 1 and 3, 2 relevant overall -> (1/1 + 2/3) / 2
    assert average_precision_at_k([1, 0, 1], 2, 20) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision_at_k([1, 0, 1], 0, 20) == 0.0


def test_precision_and_recall():
    hits = [1, 0, 1, 0]
    assert precision_at_k(hits, 2) == 0.5
    assert recall_at_k(hits, 4, 30) == 0.5
    assert recall_at_k(hits, 0, 30) == 0.0


def test_composite_weights_sum_to_metric():
    m = score_query(["a", "b"], {"a": 3, "b": 0, "c": 2})
    expected = 0.30 * m["nDCG@10"] + 0.30 * m["AP@20"] + 0.25 * m["R@30"] + 0.15 * m["P@10"]
    assert m["composite"] == pytest.approx(expected)
    assert m["R@30"] == 0.5  # found 1 of 2 relevant


def test_unjudged_products_count_as_not_relevant():
    m = score_query(["x", "y"], {"a": 3})
    assert m["composite"] == 0.0


def test_evaluate_averages_over_queries():
    qrels = build_qrels([
        {"query_id": "q1", "product_id": "a", "relevance": 3},
        {"query_id": "q2", "product_id": "b", "relevance": 3},
    ])
    res = evaluate({"q1": ["a"], "q2": ["z"]}, qrels)
    assert res["queries"] == 2
    assert res["R@30"] == 0.5
