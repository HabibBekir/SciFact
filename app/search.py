"""Les 4 stratégies de recherche : dense, bm25, hybride (RRF), hybride + reranking."""

from qdrant_client import models

from app.config import settings
from app.models import dense_model, qdrant, reranker, sparse_model


def _q_dense(query: str) -> list[float]:
    return next(dense_model().query_embed(query)).tolist()


def _q_sparse(query: str) -> models.SparseVector:
    sv = next(sparse_model().query_embed(query))
    return models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())


def _to_hits(points) -> list[dict]:
    return [
        {
            "doc_id": p.payload["doc_id"],
            "title": p.payload["title"],
            "text": p.payload["text"],
            "score": float(p.score),
        }
        for p in points
    ]


def search_dense(query: str, k: int = 10) -> list[dict]:
    res = qdrant().query_points(
        settings.collection, query=_q_dense(query), using="dense", limit=k, with_payload=True
    )
    return _to_hits(res.points)


def search_bm25(query: str, k: int = 10) -> list[dict]:
    res = qdrant().query_points(
        settings.collection, query=_q_sparse(query), using="bm25", limit=k, with_payload=True
    )
    return _to_hits(res.points)


def search_hybrid(query: str, k: int = 10, prefetch_k: int = 50) -> list[dict]:
    res = qdrant().query_points(
        settings.collection,
        prefetch=[
            models.Prefetch(query=_q_dense(query), using="dense", limit=prefetch_k),
            models.Prefetch(query=_q_sparse(query), using="bm25", limit=prefetch_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return _to_hits(res.points)


def rerank(query: str, hits: list[dict], k: int) -> list[dict]:
    if not hits:
        return []
    texts = [f"{h['title']}. {h['text']}" for h in hits]
    scores = list(reranker().rerank(query, texts))
    for h, s in zip(hits, scores):
        h["rerank_score"] = float(s)
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:k]


def search_hybrid_rerank(query: str, k: int = 10) -> list[dict]:
    candidates = search_hybrid(query, k=settings.retrieve_k, prefetch_k=settings.retrieve_k)
    return rerank(query, candidates, k)


STRATEGIES = {
    "dense": search_dense,
    "bm25": search_bm25,
    "hybrid": search_hybrid,
    "hybrid_rerank": search_hybrid_rerank,
}


def retrieve(query: str) -> list[dict]:
    """Recherche utilisée par l'application (configurable)."""
    if settings.use_rerank:
        return search_hybrid_rerank(query, k=settings.final_k)
    return search_hybrid(query, k=settings.final_k)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "0-dimensional biomaterials show inductive properties."
    for name, fn in STRATEGIES.items():
        print(f"\n=== {name} ===")
        for h in fn(q, k=3):
            print(f"  [{h['doc_id']}] {h['score']:.3f}  {h['title'][:80]}")
