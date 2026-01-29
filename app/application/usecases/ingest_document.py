from __future__ import annotations

from typing import Optional, Tuple

from app.common.runtime import now_utc
from app.application.services.chunking_service import build_chunks_from_md
from app.application.services.embedding_service import embed_texts
from app.infra.storage.opensearch import OpenSearchWriter
from app.infra.storage.ollama_embedding import OllamaEmbeddingProvider
from app.infra.storage.neo4j import Neo4jWriter


def index_md_document(
    *,
    doc_id: str,
    doc_title: str,
    source_uri: str,
    doc_sha: str,
    md_text: str,
    os_text: OpenSearchWriter,
    emb_provider: OllamaEmbeddingProvider,
    text_max_batch: int,
    text_expected_dim: int,
    text_embedding_model: str,
    bulk_size: int,
    chunk_max_chars: int,
    chunk_min_chars: int,
    neo4j_writer: Optional[Neo4jWriter] = None,
) -> Tuple[int, int]:
    chunks = build_chunks_from_md(
        doc_id=doc_id,
        doc_title=doc_title,
        source_uri=source_uri,
        doc_sha=doc_sha,
        md_text=md_text,
        max_chunk_chars=chunk_max_chars,
        min_chunk_chars=chunk_min_chars,
    )
    chunk_count = len(chunks)

    if neo4j_writer is not None and chunks:
        doc_meta = {
            "doc_id": doc_id,
            "title": doc_title or doc_id,
            "source_uri": source_uri,
            "sha256": doc_sha,
        }
        chunk_meta = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "page_start": int(c.get("page_start", 0)),
                "page_end": int(c.get("page_end", 0)),
                "order": int(c.get("order", 0)),
            }
            for c in chunks
        ]
        neo4j_writer.upsert_document_and_chunks(doc_meta, chunk_meta, rebuild_next=True)

    indexed_chunk_count = 0
    if chunks:
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(
            emb_provider,
            texts,
            max_batch_size=text_max_batch,
            expected_dim=text_expected_dim,
        )

        chunk_docs = []
        ingested_at = now_utc()
        for c, v in zip(chunks, vectors):
            chunk_id = c["chunk_id"]
            src = {
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "doc_title": c["doc_title"],
                "source_uri": c["source_uri"],
                "page_start": int(c["page_start"]),
                "page_end": int(c["page_end"]),
                "order": int(c["order"]),
                "text": c["text"],
                "image_ids": c.get("image_ids", []),
                "embedding": v,
                "embedding_model": text_embedding_model,
                "ingested_at": ingested_at,
            }
            chunk_docs.append({"_id": chunk_id, "_source": src})

        os_text.bulk_upsert(chunk_docs, batch_size=bulk_size)
        indexed_chunk_count = len(chunk_docs)

    return chunk_count, indexed_chunk_count
