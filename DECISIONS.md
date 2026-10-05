# Technical desicion

## Search Strategy

Evaluation : 300 query of test on SciFact, `python -m eval.eval_retrieval`.

| strategy | ndcg@10 | recall@10 | mrr@10 | recall@1 | latence_p50_ms | latence_p95_ms |
|---|---|---|---|---|---|---|
| dense | 0.7672 | 0.8433 | 0.7494 | 0.6567 | 9 | 12 |
| bm25 | 0.7537 | 0.8567 | 0.7298 | 0.6167 | 1 | 3 |
| hybrid | 0.8425 | 0.8767 | 0.8431 | 0.79 | 9 | 12 |
| hybrid_rerank | 0.7216 | 0.8533 | 0.6796 | 0.6 | 1374 | 1521 |

Decision : we will use the hybrid search because it presents the best results most of the metrics with an acceptable latency measure
