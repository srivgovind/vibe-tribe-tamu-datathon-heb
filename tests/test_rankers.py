import numpy as np
import pytest

from heb_ranker.bm25 import BM25Ranker
from heb_ranker.data import split_queries
from heb_ranker.hybrid import HybridRanker, minmax
from heb_ranker.text import product_text, tokenize


def test_tokenize_keeps_hyphens_and_drops_punctuation():
    assert tokenize("H-E-B Organic, 2% Milk!") == ["h-e-b", "organic", "2", "milk"]


def test_product_text_boosts_title_and_handles_missing_fields():
    text = product_text({"title": "Milk", "brand": None, "category_path": "A -> B"})
    assert text.split().count("Milk") == 3
    assert text.split().count("A") == 2


def test_bm25_ranks_exact_match_first(products):
    ranker = BM25Ranker(products)
    assert ranker.rank("gluten free pasta", top_k=2)[0] == "p3"
    assert len(ranker.rank("milk", top_k=10)) == len(products)


def test_minmax_handles_constant_input():
    assert np.all(minmax(np.array([2.0, 2.0])) == 0)
    assert minmax(np.array([1.0, 3.0])).tolist() == [0.0, 1.0]


@pytest.mark.parametrize("norm", ["cosine01", "minmax"])
def test_hybrid_scores_bounded(products, encoder, norm):
    ranker = HybridRanker(products, encoder)
    s = ranker.scores("bbq chicken", bm25_weight=0.3, semantic_norm=norm)
    assert s.shape == (len(products),)
    assert s.min() >= 0 and s.max() <= 1 + 1e-9


def test_hybrid_weight_one_equals_bm25(products, encoder):
    hybrid = HybridRanker(products, encoder)
    bm25 = BM25Ranker(products)
    assert hybrid.rank("chicken soup", top_k=4, bm25_weight=1.0) == bm25.rank("chicken soup", top_k=4)


def test_hybrid_semantic_only_finds_related_product(products, encoder):
    ranker = HybridRanker(products, encoder)
    assert ranker.rank("almond milk", top_k=1, bm25_weight=0.0)[0] == "p1"


def test_embedding_cache_is_reused(products, encoder, tmp_path):
    calls = []

    def counting_encoder(texts):
        calls.append(len(texts))
        return encoder(texts)

    HybridRanker(products, counting_encoder, cache_dir=tmp_path, cache_key="m")
    HybridRanker(products, counting_encoder, cache_dir=tmp_path, cache_key="m")
    assert calls == [len(products)]  # catalog encoded once, then loaded from disk
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_split_is_deterministic_and_disjoint():
    qs = [{"query_id": f"s{i}"} for i in range(300)]
    tune, hold = split_queries(qs)
    _, hold_reordered = split_queries(list(reversed(qs)))
    assert {q["query_id"] for q in hold_reordered} == {q["query_id"] for q in hold}
    assert not {q["query_id"] for q in tune} & {q["query_id"] for q in hold}
    assert len(tune) + len(hold) == 300
    assert 0.1 < len(hold) / 300 < 0.3
