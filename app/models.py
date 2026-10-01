"""Chargement paresseux des modèles (une seule fois par processus)."""

from functools import lru_cache

from app.config import settings


@lru_cache
def dense_model():
    from fastembed import TextEmbedding

    return TextEmbedding(settings.dense_model)


@lru_cache
def sparse_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(settings.sparse_model)


@lru_cache
def reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(settings.rerank_model)


@lru_cache
def qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(url=settings.qdrant_url, timeout=60)
