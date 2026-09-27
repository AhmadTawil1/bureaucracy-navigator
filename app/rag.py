"""Retrieval: e5 query embedding + Qdrant search over the Kol Zchut chunks (§5)."""
from functools import cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings

COLLECTION = "kolzchut"


@cache
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model, device="cpu")  # 2.2 GB: loaded on first use


@cache
def _qdrant() -> QdrantClient:
    return QdrantClient(path=settings.qdrant_path)  # local mode: only one process can open it


def search(question: str, k: int = 4) -> list[tuple[float, dict]]:
    """Top-k chunks as (cosine score, chunk). e5 needs the "query: " prefix."""
    vector = _embedder().encode(f"query: {question}", normalize_embeddings=True)
    hits = _qdrant().query_points(COLLECTION, query=vector.tolist(), limit=k).points
    return [(hit.score, hit.payload) for hit in hits]


def warm_up() -> None:
    """Load the model and open Qdrant at startup, so the first user doesn't wait."""
    search("בדיקה", k=1)
