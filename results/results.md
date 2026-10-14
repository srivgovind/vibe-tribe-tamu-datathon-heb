Holdout = 61 labeled queries never used for tuning (10,974 graded judgments over a 3,287-product catalog).

| System | Composite | nDCG@10 | AP@20 | P@10 | R@30 | vs BM25 |
|---|---|---|---|---|---|---|
| BM25 baseline | 0.568 | 0.662 | 0.420 | 0.572 | 0.629 | - |
| Hybrid, datathon submission (MPNet, w=0.2, cosine01) | 0.632 | 0.730 | 0.480 | 0.643 | 0.691 | +6.4 pts (+11.3%) |
| Hybrid, best on tune (all-MiniLM-L6-v2, w=0.3, minmax) | 0.664 | 0.730 | 0.516 | 0.687 | 0.750 | +9.6 pts (+17.0%) |

Composite on all labeled queries: BM25 baseline: 0.6017, Hybrid, datathon submission (MPNet, w=0.2, cosine01): 0.6638, Hybrid, best on tune (all-MiniLM-L6-v2, w=0.3, minmax): 0.6797.
