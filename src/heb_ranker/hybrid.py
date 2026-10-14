"""Hybrid ranker: weighted fusion of BM25 and sentence-embedding similarity."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from .bm25 import BM25Ranker
from .text import product_text

# An encoder maps a list of strings to an (n, d) array. Injected so tests can
# run without downloading a transformer model.
Encoder = Callable[[list[str]], np.ndarray]
SemanticNorm = Literal["cosine01", "minmax"]


def sentence_transformer_encoder(model_name: str, batch_size: int = 32) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

    return encode


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def _unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class HybridRanker:
    """score = w * minmax(BM25) + (1 - w) * norm(cosine).

    semantic_norm:
      - "cosine01": maps cosine from [-1, 1] to [0, 1] (the datathon submission).
      - "minmax": per-query min-max, so both signals span [0, 1] and the
        weight means what it says.
    """

    def __init__(
        self,
        products: list[dict],
        encoder: Encoder,
        cache_dir: str | Path | None = None,
        cache_key: str = "model",
    ):
        self.product_ids = [p["product_id"] for p in products]
        self.bm25 = BM25Ranker(products)
        self._encode = encoder
        texts = [product_text(p) for p in products]
        self._product_emb = _unit(self._embed_catalog(texts, cache_dir, cache_key))

    def _embed_catalog(self, texts: list[str], cache_dir, cache_key: str) -> np.ndarray:
        if cache_dir is None:
            return np.asarray(self._encode(texts))
        digest = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
        path = Path(cache_dir) / f"{cache_key.replace('/', '_')}-{digest}.npy"
        if path.exists():
            return np.load(path)
        emb = np.asarray(self._encode(texts))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        return emb

    def semantic_scores(self, query: str) -> np.ndarray:
        return self.semantic_scores_batch([query])[0]

    def semantic_scores_batch(self, queries: list[str]) -> np.ndarray:
        """(n_queries, n_products) cosine similarities; one encoder call."""
        q = _unit(np.asarray(self._encode(queries)))
        return q @ self._product_emb.T

    def scores(
        self,
        query: str,
        bm25_weight: float = 0.2,
        semantic_norm: SemanticNorm = "cosine01",
        _semantic: np.ndarray | None = None,
    ) -> np.ndarray:
        bm = minmax(self.bm25.scores(query))
        sem = self.semantic_scores(query) if _semantic is None else _semantic
        sem = (sem + 1) / 2 if semantic_norm == "cosine01" else minmax(sem)
        return bm25_weight * bm + (1 - bm25_weight) * sem

    def rank(self, query: str, top_k: int = 50, **kwargs) -> list[str]:
        s = self.scores(query, **kwargs)
        order = np.argsort(-s, kind="stable")[:top_k]
        return [self.product_ids[i] for i in order]
