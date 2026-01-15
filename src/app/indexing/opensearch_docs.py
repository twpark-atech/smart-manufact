# ==============================================================================
# 목적 : OpenSearch Documents 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import time
from typing import Sequence, Dict, Any, List


def build_os_docs(
    chunks: Sequence[Dict[str, Any]],
    vectors: Sequence[List[float]],
    *,
    embedding_model: str,    
) -> List[Dict[str, Any]]:
    if len(chunks) != len(vectors):
        raise ValueError("normalized and vectors length mismatch")
    
    now_ms = int(time.time() * 1000)
    docs: List[Dict[str, Any]] = []

    for c, v in zip(chunks, vectors):
        docs.append(
            {
                "_id": c["chunk_id"],
                "_source": {
                    "doc_id": c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "doc_title": c["doc_title"],
                    "source_uri": c["source_uri"],
                    "sha256": c["sha256"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "order": c["order"],
                    "text": c["text"],
                    "image_ids": c.get("image_ids", []),
                    "embedding": v,
                    "embedding_model": embedding_model,
                    "ingested_at": now_ms,
                },
            }
        )

    return docs