import time

from qdrant_client import models
from tqdm import tqdm

from app.config import settings
from app.data import doc_to_text, load_corpus
from app.models import dense_model, qdrant, sparse_model

BATCH = 64


def create_collection(recreate: bool = False) -> None:
    client = qdrant()
    if client.collection_exists(settings.collection):
        if not recreate:
            print(f"Collection '{settings.collection}' exists, skipping creation.")
            return
        client.delete_collection(settings.collection)
    dim = dense_model().embedding_size
    client.create_collection(
        settings.collection,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    print(f"Collection '{settings.collection}' created (dense {dim} dim. + bm25).")


def index_corpus(limit: int | None = None) -> int:
    client = qdrant()
    corpus = load_corpus()
    ids = list(corpus)[:limit] if limit else list(corpus)
    t0 = time.time()
    for start in tqdm(range(0, len(ids), BATCH), desc="Indexation"):
        batch_ids = ids[start : start + BATCH]
        texts = [doc_to_text(corpus[i]) for i in batch_ids]
        dense_vecs = list(dense_model().passage_embed(texts))
        sparse_vecs = list(sparse_model().passage_embed(texts))
        points = [
            models.PointStruct(
                id=int(doc_id),
                vector={
                    "dense": dv.tolist(),
                    "bm25": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                },
                payload={"doc_id": doc_id, "title": corpus[doc_id]["title"], "text": corpus[doc_id]["text"]},
            )
            for doc_id, dv, sv in zip(batch_ids, dense_vecs, sparse_vecs)
        ]
        client.upsert(settings.collection, points=points, wait=True)
    print(f"{len(ids)} documents indexed in {time.time() - t0:.0f} s")
    return len(ids)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--recreate", action="store_true", help="Delete and recreate the collection (default: False)"
    )
    p.add_argument("--limit", type=int, default=None, help="Index only N documents")
    a = p.parse_args()
    create_collection(recreate=a.recreate)
    index_corpus(limit=a.limit)
