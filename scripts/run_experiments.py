"""Reproduce every number in the README.

1. Split the 288 labeled training queries into tune (~80%) and holdout (~20%).
2. Score the BM25 baseline.
3. For each embedding model, sweep the BM25 weight and score normalization on
   the tune split, pick the best config, and report it on the untouched holdout.
4. Write results/results.json and results/results.md, and refresh the results
   table in README.md.

Usage:
    python scripts/run_experiments.py              # full sweep (downloads models once)
    python scripts/run_experiments.py --bm25-only  # no model download
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heb_ranker.bm25 import BM25Ranker  # noqa: E402
from heb_ranker.data import load_all, split_queries  # noqa: E402
from heb_ranker.hybrid import HybridRanker, sentence_transformer_encoder  # noqa: E402
from heb_ranker.metrics import build_qrels, evaluate  # noqa: E402

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
]
BM25_WEIGHTS = [round(0.1 * i, 1) for i in range(11)]  # 0.0 (semantic only) .. 1.0 (BM25 only)
NORMS = ["cosine01", "minmax"]
DATATHON_CONFIG = {"model": "sentence-transformers/all-mpnet-base-v2", "bm25_weight": 0.2, "semantic_norm": "cosine01"}
TOP_K = 50
METRIC_COLS = ["composite", "nDCG@10", "AP@20", "P@10", "R@30"]


def _subset(rankings: dict[str, list[str]], queries: list[dict]) -> dict[str, list[str]]:
    return {q["query_id"]: rankings[q["query_id"]] for q in queries}


def run(models: list[str], bm25_only: bool) -> dict:
    data = load_all()
    products, train_q = data["products"], data["train_queries"]
    qrels = build_qrels(data["labels"])
    tune_q, holdout_q = split_queries(train_q)
    print(f"{len(products)} products | {len(train_q)} labeled queries -> {len(tune_q)} tune / {len(holdout_q)} holdout")

    results: dict = {
        "catalog_size": len(products),
        "labeled_queries": len(train_q),
        "judgments": len(data["labels"]),
        "tune_queries": len(tune_q),
        "holdout_queries": len(holdout_q),
        "systems": {},
        "sweep": [],
    }

    t0 = time.perf_counter()
    bm25 = BM25Ranker(products)
    bm25_rank = {q["query_id"]: bm25.rank(q["query"], TOP_K) for q in train_q}
    results["bm25_ms_per_query"] = round(1000 * (time.perf_counter() - t0) / len(train_q), 1)
    results["systems"]["BM25 baseline"] = {
        split: evaluate(_subset(bm25_rank, qs), qrels)
        for split, qs in [("all", train_q), ("tune", tune_q), ("holdout", holdout_q)]
    }

    if not bm25_only:
        best = None
        for model in models:
            print(f"Encoding catalog with {model} ...")
            hybrid = HybridRanker(products, sentence_transformer_encoder(model), ROOT / ".cache", model)
            sem = hybrid.semantic_scores_batch([q["query"] for q in train_q])
            sem_by_q = {q["query_id"]: sem[i] for i, q in enumerate(train_q)}

            def ranks(w: float, norm: str) -> dict[str, list[str]]:
                out = {}
                for q in train_q:
                    s = hybrid.scores(q["query"], bm25_weight=w, semantic_norm=norm, _semantic=sem_by_q[q["query_id"]])
                    out[q["query_id"]] = [hybrid.product_ids[i] for i in (-s).argsort(kind="stable")[:TOP_K]]
                return out

            for norm in NORMS:
                for w in BM25_WEIGHTS:
                    r = ranks(w, norm)
                    row = {
                        "model": model,
                        "bm25_weight": w,
                        "semantic_norm": norm,
                        "tune": evaluate(_subset(r, tune_q), qrels),
                        "holdout": evaluate(_subset(r, holdout_q), qrels),
                        "all": evaluate(r, qrels),
                    }
                    results["sweep"].append(row)
                    print(f"  {norm:8s} w={w:.1f}  tune={row['tune']['composite']:.4f}  holdout={row['holdout']['composite']:.4f}")
                    if best is None or row["tune"]["composite"] > best["tune"]["composite"]:
                        best = row

        for row in results["sweep"]:
            if all(row[k] == v for k, v in DATATHON_CONFIG.items()):
                results["systems"]["Hybrid, datathon submission (MPNet, w=0.2, cosine01)"] = {
                    k: row[k] for k in ("all", "tune", "holdout")
                }
        if best:
            label = f"Hybrid, best on tune ({best['model'].split('/')[-1]}, w={best['bm25_weight']}, {best['semantic_norm']})"
            results["systems"][label] = {k: best[k] for k in ("all", "tune", "holdout")}
            results["best_config"] = {k: best[k] for k in ("model", "bm25_weight", "semantic_norm")}
    return results


def to_markdown(res: dict) -> str:
    base = res["systems"]["BM25 baseline"]["holdout"]["composite"]
    lines = [
        f"Holdout = {res['holdout_queries']} labeled queries never used for tuning "
        f"({res['judgments']:,} graded judgments over a {res['catalog_size']:,}-product catalog).",
        "",
        "| System | Composite | nDCG@10 | AP@20 | P@10 | R@30 | vs BM25 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, splits in res["systems"].items():
        m = splits["holdout"]
        delta = m["composite"] - base
        rel = f"{delta * 100:+.1f} pts ({delta / base * 100:+.1f}%)" if name != "BM25 baseline" else "-"
        lines.append(f"| {name} | " + " | ".join(f"{m[c]:.3f}" for c in METRIC_COLS) + f" | {rel} |")
    full = {n: s["all"]["composite"] for n, s in res["systems"].items()}
    lines += ["", "Composite on all labeled queries: " + ", ".join(f"{n}: {v:.4f}" for n, v in full.items()) + "."]
    return "\n".join(lines)


def update_readme(table: str) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    new = re.sub(
        r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)",
        lambda m: m.group(1) + table + m.group(2),
        text,
        flags=re.S,
    )
    readme.write_text(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    res = run(args.models, args.bm25_only)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    table = to_markdown(res)
    (out / "results.md").write_text(table + "\n")
    update_readme(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
