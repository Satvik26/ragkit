"""BGE-M3 embeddings and Cross-Encoder reranking over HTTP.

The main API container stays free of PyTorch and calls this sidecar.
"""

from __future__ import annotations

import math
import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

model = SentenceTransformer(MODEL_NAME)
_reranker: CrossEncoder | None = None

app = FastAPI(title="ragkit-embedder")


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_NAME)
    return _reranker


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings")
def embeddings(body: EmbeddingRequest) -> dict[str, list[dict[str, object]]]:
    texts = [body.input] if isinstance(body.input, str) else list(body.input)
    if not texts:
        return {"data": []}
    # Cosine search in Qdrant expects unit-length vectors.
    vectors = model.encode(texts, normalize_embeddings=True)
    return {
        "data": [
            {"index": index, "embedding": vector.tolist()} for index, vector in enumerate(vectors)
        ]
    }


@app.post("/v1/rerank")
def rerank(body: RerankRequest) -> dict[str, list[dict[str, object]]]:
    if not body.documents:
        return {"results": []}
    cross_encoder = _get_reranker()
    pairs = [(body.query, doc) for doc in body.documents]
    logits = cross_encoder.predict(pairs)
    return {
        "results": [
            {"index": index, "score": _sigmoid(float(logit))} for index, logit in enumerate(logits)
        ]
    }
