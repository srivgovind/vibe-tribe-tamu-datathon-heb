"""Okapi BM25 keyword ranker over boosted product text."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from .text import product_text, tokenize


class BM25Ranker:
    def __init__(self, products: list[dict]):
        self.product_ids = [p["product_id"] for p in products]
        self._bm25 = BM25Okapi([tokenize(product_text(p)) for p in products])

    def scores(self, query: str) -> np.ndarray:
        """Raw BM25 score for every product, in catalog order."""
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)

    def rank(self, query: str, top_k: int = 50) -> list[str]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:top_k]
        return [self.product_ids[i] for i in order]
