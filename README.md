# H-E-B Product Search Ranking

Hybrid keyword + semantic search that ranks a 3,287-product H-E-B grocery catalog for natural-language queries such as *"fiber rich breakfast cereal"* or *"fragrance free wipes for babies"*. Built for the **TAMU Datathon 2025 H-E-B challenge**, where our team, Vibe Tribe, **placed 3rd of 40+ teams**.

**Results**, scored on 61 held-out labeled queries never used for tuning:
- **Datathon submission** (BM25 + MPNet): composite **0.568 → 0.632** over keyword-only BM25, **+6.4 points (+11.3%)**.
- **After a post-datathon sweep of 66 configs** (BM25 + MiniLM, per-query min-max fusion): **0.664**, **+9.6 points (+17.0%)**. Recall@30 rose from 0.629 to 0.750, and Precision@10 from 0.572 to 0.687.

## How it works

```
query ──► BM25 over boosted product text ──► min-max to [0,1] ──┐
      │                                                          ├─► w·BM25 + (1−w)·semantic ─► top 50
      └─► sentence embedding ─► cosine vs cached catalog vectors ┘
```

- **Product text:** title ×3, brand ×2, and category levels ×2, plus description and ingredients. Repeating a field boosts its weight in BM25 and in the embedded text.
- **Keyword signal:** Okapi BM25 (`rank-bm25`), under 2 ms per query on a laptop CPU, including index build.
- **Semantic signal:** sentence-transformer embeddings with cosine similarity (`all-MiniLM-L6-v2` in the final config; `all-mpnet-base-v2` at the datathon). Catalog vectors are computed once and cached on disk, keyed by a hash of the catalog text.
- **Fusion:** a weighted sum of the normalized scores. The datathon config used w = 0.2 with cosine mapped from [-1, 1] to [0, 1], which compresses the semantic signal into a narrow band. The final config applies per-query min-max to both signals so the weight reflects each signal's real influence, with w = 0.3.

## Evaluation

Scored with the competition's composite metric over graded relevance labels (0–3):

```
composite = 0.30·nDCG@10 + 0.30·AP@20 + 0.25·Recall@30 + 0.15·Precision@10
```

`src/heb_ranker/metrics.py` reimplements the official scorer. It reproduces the organizers' BM25 score exactly (0.6017 on all 288 labeled queries).

To keep tuning honest, the 288 labeled queries are split by query-ID hash into **227 tune** and **61 holdout** queries. Model and weight are chosen on tune only, and the holdout is reported untouched. The sweep covers 3 embedding models × 11 BM25 weights (0.0 to 1.0) × 2 normalizations, for 66 configurations.

### Results on the holdout split

<!-- RESULTS:START -->
Holdout = 61 labeled queries never used for tuning (10,974 graded judgments over a 3,287-product catalog).

| System | Composite | nDCG@10 | AP@20 | P@10 | R@30 | vs BM25 |
|---|---|---|---|---|---|---|
| BM25 baseline | 0.568 | 0.662 | 0.420 | 0.572 | 0.629 | - |
| Hybrid, datathon submission (MPNet, w=0.2, cosine01) | 0.632 | 0.730 | 0.480 | 0.643 | 0.691 | +6.4 pts (+11.3%) |
| Hybrid, best on tune (all-MiniLM-L6-v2, w=0.3, minmax) | 0.664 | 0.730 | 0.516 | 0.687 | 0.750 | +9.6 pts (+17.0%) |

Composite on all labeled queries: BM25 baseline: 0.6017, Hybrid, datathon submission (MPNet, w=0.2, cosine01): 0.6638, Hybrid, best on tune (all-MiniLM-L6-v2, w=0.3, minmax): 0.6797.
<!-- RESULTS:END -->

`make results` regenerates this table, `results/results.json`, and `results/results.md` from scratch.

### Findings

- **The smaller model won.** The 22M-parameter MiniLM beat both 110M-parameter MPNet models on tune and holdout. Switching models added +2.2 of the +3.2 holdout points gained over the datathon config, and min-max fusion with w = 0.3 added the remaining +1.0.
- **Neither signal wins alone.** Every model peaked at a BM25 weight of 0.2 to 0.4. Pure semantic search (w = 0) and pure BM25 (w = 1) both scored lower.
- **The tuning generalizes.** The config chosen on the tune split also ranks among the top 3 of 66 on the holdout (0.664, versus a best possible holdout score of 0.668).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the competition files in data/ (see data/README.md)

make test          # unit tests, no model download needed
make results       # full sweep; downloads 3 models (~1 GB) on first run
make submission    # writes outputs/submission.json for the 191 test queries
```

## Repository layout

```
src/heb_ranker/
  text.py        tokenization and field-boosted product text
  bm25.py        BM25 ranker
  hybrid.py      score fusion, normalization, embedding cache
  metrics.py     nDCG, AP, P, R, and the composite score
  data.py        loaders and deterministic tune/holdout split
scripts/
  run_experiments.py   sweep, holdout evaluation, README table refresh
  make_submission.py   ranks the unlabeled test queries
tests/                 metric correctness, ranker behavior, cache, split
```

## What I'd do next

- Re-rank the top 50 with a cross-encoder, which usually lifts nDCG@10 the most.
- Fine-tune the bi-encoder on the 10,974 graded pairs instead of using it zero-shot.
- Replace the dense matrix product with a FAISS index once the catalog grows past about 100K products.

## License

MIT. Competition data belongs to H-E-B and the datathon organizers and is not redistributed here.
