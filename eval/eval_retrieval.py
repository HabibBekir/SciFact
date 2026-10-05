"""Compare les stratégies de recherche sur les 300 requêtes de test SciFact."""

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from tqdm import tqdm

from app.data import load_qrels, load_queries
from app.search import STRATEGIES
from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

K = 10


def evaluate(strategy: str, limit: int | None = None) -> dict:
    queries, qrels = load_queries(), load_qrels(split="test")
    qids = [q for q in qrels if q in queries][:limit]
    fn = STRATEGIES[strategy]
    rows, latencies = [], []
    for qid in tqdm(qids, desc=strategy):
        t0 = time.perf_counter()
        hits = fn(queries[qid], k=K)
        latencies.append(time.perf_counter() - t0)
        got = [h["doc_id"] for h in hits]
        rel = {d for d, s in qrels[qid].items() if s > 0}
        rows.append(
            {
                "ndcg@10": ndcg_at_k(got, qrels[qid], K),
                "recall@10": recall_at_k(got, rel, K),
                "mrr@10": mrr_at_k(got, rel, K),
                "recall@1": recall_at_k(got, rel, 1),
            }
        )
    res = {m: round(mean(r[m] for r in rows), 4) for m in rows[0]}
    lat = sorted(latencies)
    res["latence_p50_ms"] = round(1000 * lat[len(lat) // 2])
    res["latence_p95_ms"] = round(1000 * lat[int(len(lat) * 0.95) - 1])
    res["n_requetes"] = len(qids)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    results = {s: evaluate(s, a.limit) for s in a.strategies}
    cols = ["ndcg@10", "recall@10", "mrr@10", "recall@1", "latence_p50_ms", "latence_p95_ms"]
    print("\n| stratégie | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for s, r in results.items():
        print(f"| {s} | " + " | ".join(str(r[c]) for c in cols) + " |")
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    out = Path("eval/results") / f"retrieval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nRésultats sauvegardés dans {out}")
