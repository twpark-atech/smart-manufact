# ==============================================================================
# 목적 : Embedding 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import requests
from dataclasses import dataclass
from typing import List

from app.ports.embedding import EmbeddingProvider
from app.common.runtime import request_with_retry


@dataclass(frozen=True)
class OllamaEmbeddingConfig:
    base_url: str
    model: str
    timeout_sec: int = 120
    max_batch_size: int = 16
    truncate: bool = True


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, cfg: OllamaEmbeddingConfig):
        self._cfg = cfg
        self._endpoint = cfg.base_url.rstrip("/") + "/api/embed"

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._cfg.max_batch_size):
            batch = texts[i : i + self._cfg.max_batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors
    
    def _embed_batch(self, inputs: List[str]) -> List[List[float]]:
        def _call():
            payload = {
                "model": self._cfg.model,
                "input": inputs,
                "truncate": self._cfg.truncate,
            }
            r = requests.post(self._endpoint, json=payload, timeout=self._cfg.timeout_sec)
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings")
            if not isinstance(embs, list) or not embs:
                raise ValueError(f"Ollama returned empty embeddings: {data}")
            return embs
        
        embs = request_with_retry(_call, retries=5)
        if len(embs) != len(inputs):
            raise ValueError(f"Embeddings count mismatch: {len(embs)} != {len(inputs)}")
        return embs