import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VOCAB = ["milk", "almond", "chicken", "bbq", "pasta", "gluten", "free", "soup"]


def bag_of_words_encoder(texts: list[str]) -> np.ndarray:
    """Deterministic stand-in for a sentence-transformer: counts vocab words."""
    return np.array([[t.lower().split().count(w) + 0.01 for w in VOCAB] for t in texts], dtype=float)


@pytest.fixture
def products():
    return [
        {"product_id": "p1", "title": "Almond Milk Unsweetened", "brand": "Blue Diamond",
         "category_path": "Dairy -> Plant Milk", "description": "almond milk"},
        {"product_id": "p2", "title": "BBQ Chicken Breast", "brand": "H-E-B",
         "category_path": "Meat -> Chicken", "description": "smoked bbq chicken"},
        {"product_id": "p3", "title": "Gluten Free Pasta", "brand": "Barilla",
         "category_path": "Pantry -> Pasta", "description": "gluten free penne pasta"},
        {"product_id": "p4", "title": "Chicken Noodle Soup", "brand": "H-E-B",
         "category_path": "Pantry -> Soup", "description": "chicken soup"},
    ]


@pytest.fixture
def encoder():
    return bag_of_words_encoder
