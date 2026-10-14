"""Loading competition files and a deterministic tune / holdout split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all(data_dir: str | Path = DATA_DIR) -> dict:
    d = Path(data_dir)
    return {
        "products": load_json(d / "products.json"),
        "train_queries": load_json(d / "queries_synth_train.json"),
        "test_queries": load_json(d / "queries_synth_test.json"),
        "labels": load_json(d / "labels_synth_train.json"),
    }


def split_queries(queries: list[dict], holdout_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Hash-based split so a query always lands in the same bucket,
    independent of file order or random seeds."""

    def bucket(qid: str) -> float:
        return int(hashlib.md5(qid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    tune = [q for q in queries if bucket(q["query_id"]) >= holdout_frac]
    holdout = [q for q in queries if bucket(q["query_id"]) < holdout_frac]
    return tune, holdout
