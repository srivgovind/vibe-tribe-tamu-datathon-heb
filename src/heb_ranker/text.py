"""Text preparation shared by the BM25 and embedding rankers."""
from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9\s-]")

# Field boosts: repeating a field raises its term frequency in BM25 and its
# weight in the embedded text. Tuned by hand during the datathon.
TITLE_BOOST = 3
BRAND_BOOST = 2
CATEGORY_BOOST = 2


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation (keeping hyphens), split on whitespace."""
    return _NON_ALNUM.sub(" ", text.lower()).split()


def product_text(product: dict) -> str:
    """Build one searchable string per product with field boosting."""
    parts: list[str] = []
    if product.get("title"):
        parts += [product["title"]] * TITLE_BOOST
    if product.get("brand"):
        parts += [product["brand"]] * BRAND_BOOST
    if product.get("category_path"):
        levels = [c.strip() for c in product["category_path"].split("->") if c.strip()]
        parts += levels * CATEGORY_BOOST
    if product.get("description"):
        parts.append(product["description"])
    if product.get("ingredients"):
        parts.append(product["ingredients"])
    return " ".join(parts)
