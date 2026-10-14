"""Rank the unlabeled test queries and write outputs/submission.json.

Uses the best config from results/results.json if present, otherwise the
configuration submitted at the datathon.

Usage:
    python scripts/make_submission.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heb_ranker.data import load_all  # noqa: E402
from heb_ranker.hybrid import HybridRanker, sentence_transformer_encoder  # noqa: E402

DEFAULT = {"model": "sentence-transformers/all-mpnet-base-v2", "bm25_weight": 0.2, "semantic_norm": "cosine01"}
TOP_K = 50


def main() -> None:
    results_path = ROOT / "results" / "results.json"
    cfg = DEFAULT
    if results_path.exists():
        cfg = json.loads(results_path.read_text()).get("best_config", DEFAULT)
    print(f"Config: {cfg}")

    data = load_all()
    ranker = HybridRanker(data["products"], sentence_transformer_encoder(cfg["model"]), ROOT / ".cache", cfg["model"])
    rows = []
    for q in data["test_queries"]:
        ranked = ranker.rank(q["query"], TOP_K, bm25_weight=cfg["bm25_weight"], semantic_norm=cfg["semantic_norm"])
        rows += [{"query_id": q["query_id"], "rank": r, "product_id": pid} for r, pid in enumerate(ranked, start=1)]

    out = ROOT / "outputs" / "submission.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} rows for {len(data['test_queries'])} queries to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
