# ==============================================================================
# 목적 : Embedding 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import List

from app.storage.embedding import OllamaEmbeddingProvider


def embed_texts(
    emb_provider: OllamaEmbeddingProvider,
    texts: List[str],
    *,
    max_batch_size: int,
    expected_dim: int,
) -> List[List[float]]:
    vectors: List[List[float]] = []
    for start in range(0, len(texts), max_batch_size):
        batch = texts[start : start + max_batch_size]
        batch_vecs = emb_provider.embed(batch)
        if not batch_vecs:
            raise RuntimeError("Embedding provider returned empty vectors.")
        for v in batch_vecs:
            if len(v) != expected_dim:
                raise RuntimeError(f"Unexpected embedding dim: {len(v)} (expected {expected_dim})")
        vectors.extend(batch_vecs)

    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(texts)}")
    return vectors