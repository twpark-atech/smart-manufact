# ==============================================================================
# 목적 : Image Staging 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

import logging
from typing import List, Dict, Any

from app.common.runtime import now_utc
from app.common.ids import parse_s3_uri
from app.infra.storage.opensearch import OpenSearchWriter
from app.infra.storage.minio import MinIOReader
from app.infra.storage.ollama_embedding import OllamaEmbeddingProvider
from app.application.services.embedding_service import ImageEmbedConfig, embed_texts, embed_images_bytes_batch


_log = logging.getLogger(__name__)


def process_image_stage_batch(
    *,
    buf: List[Dict[str, Any]],
    os_image: OpenSearchWriter,
    os_images_stage: OpenSearchWriter,
    bulk_size: int,
    emb_provider: OllamaEmbeddingProvider,
    text_max_batch: int,
    text_expected_dim: int,
    text_embedding_model: str,
    image_embed_cfg: ImageEmbedConfig,
    image_embedding_model: str,
    minio_reader: MinIOReader,
) -> int:
    desc_texts = [b.get("desc_text", "") for b in buf]
    try:
        desc_vecs = embed_texts(
            emb_provider,
            desc_texts,
            max_batch_size=min(text_max_batch, len(desc_texts)),
            expected_dim=text_expected_dim,
        )
    except Exception as e:
        _log.warning("Desc batch embed failed. retry once. err=%s", e)
        try:
            desc_vecs = embed_texts(
                emb_provider,
                desc_texts,
                max_batch_size=min(text_max_batch, len(desc_texts)),
                expected_dim=text_expected_dim,
            )
        except Exception as e2:
            now = now_utc()
            fail_docs = []
            for b in buf:
                fail_docs.append({
                    "_id": b["image_id"],
                    "_source": {
                        **b,
                        "status": "failed",
                        "attempts": int(b.get("attempts", 0)) + 1,
                        "last_error": f"desc_embed_failed: {e2}",
                        "updated_at": now,
                    },
                })
            os_images_stage.bulk_upsert(fail_docs, batch_size=bulk_size)
            return 0

    try:
        image_bytes_list: List[bytes] = fetch_images_from_minio(buf=buf, minio_reader=minio_reader)
        img_vecs = embed_images_bytes_batch(image_bytes_list, cfg=image_embed_cfg)
    except Exception as e:
        now = now_utc()
        fail_docs = []
        for b in buf:
            fail_docs.append({
                "_id": b["image_id"],
                "_source": {
                    **b,
                    "status": "failed",
                    "attempts": int(b.get("attempts", 0)) + 1,
                    "last_error": f"image_embed_failed: {e}",
                    "updated_at": now,
                },
            })
        os_images_stage.bulk_upsert(fail_docs, batch_size=bulk_size)
        return 0

    now = now_utc()
    image_docs: List[Dict[str, Any]] = []
    done_docs: List[Dict[str, Any]] = []
    for b, dvec, ivec in zip(buf, desc_vecs, img_vecs):
        image_id = b["image_id"]
        src = {
            "doc_id": b["doc_id"],
            "image_id": image_id,
            "doc_title": b["doc_title"],
            "source_uri": b["source_uri"],
            "page_no": int(b["page_no"]),
            "order": int(b["order"]),
            "pdf_uri": b["pdf_uri"],
            "image_uri": b["image_uri"],
            "image_mime": b.get("image_mime") or "image/png",
            "image_sha256": b["image_sha256"],
            "width": int(b["width"]),
            "height": int(b["height"]),
            "bbox": b.get("bbox") or {},
            "desc_text": b.get("desc_text") or "",
            "desc_embedding": dvec,
            "image_embedding": ivec,
            "desc_embedding_model": text_embedding_model,
            "image_embedding_model": image_embedding_model,
            "ingested_at": now,
        }
        image_docs.append({"_id": image_id, "_source": src})

        done_docs.append({
            "_id": image_id,
            "_source": {
                **b,
                "status": "done",
                "attempts": int(b.get("attempts", 0)) + 1,
                "last_error": "",
                "updated_at": now,
            }
        })

    os_image.bulk_upsert(image_docs, batch_size=bulk_size)
    os_images_stage.bulk_upsert(done_docs, batch_size=bulk_size)

    return len(image_docs)


def fetch_images_from_minio(*, buf: List[Dict[str, Any]], minio_reader: MinIOReader) -> List[bytes]:
    images: List[bytes] = []
    for b in buf:
        uri = str(b.get("image_uri") or "")
        bucket, key = parse_s3_uri(uri)
        images.append(minio_reader.download_bytes(bucket=bucket, key=key))
    return images
