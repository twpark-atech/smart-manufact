# ==============================================================================
# 목적 : PDF에서 OCR 후 MD/Artifacts를 생성하는 코드 (fresh / from_pages_staging)
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from app.common.config import load_config
from app.common.hash import sha256_file, sha256_bytes
from app.common.ids import s3_uri, parse_s3_uri
from app.common.parser import get_value, pdf_to_page_pngs, build_md_from_pages
from app.common.runtime import now_utc

from app.parsing.image_extractor import extract_and_store_images_from_page, get_image_size_from_bytes
from app.parsing.ocr import ocr_page
from app.parsing.pdf import coerce_page_no_and_payload, materialize_png_payload

from app.storage.minio import MinIOConfig, MinIOWriter, MinIOReader
from app.storage.opensearch import OpenSearchConfig, OpenSearchWriter
from app.storage.embedding import OllamaEmbeddingConfig, OllamaEmbeddingProvider

from app.indexing.chunking import build_chunks_from_md
from app.indexing.embedding import (
    ImageEmbedConfig,
    embed_images_bytes_batch,
    embed_texts,
)
from app.indexing.index_bodies import (
    build_pdf_chunks_v1_body,
    build_pdf_images_v1_body,
    build_pdf_pages_staging_v1_body,
    build_pdf_images_staging_v1_body,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseArtifactsResult:
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str

    output_dir: str
    assets_root: str
    pages_dir: str
    md_path: str

    pdf_uri: str
    text_index: str
    image_index: str
    pages_staging_index: str
    images_staging_index: str

    page_count: int
    extracted_image_count: int
    generated_desc_count: int

    staged_page_count: int
    failed_page_count: int
    staged_image_count: int
    indexed_image_count: int

    chunk_count: int
    indexed_chunk_count: int

    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_pages_from_staging(
    *,
    os_pages_staging: OpenSearchWriter,
    doc_id: str,
) -> Tuple[List[str], int, int, int, Dict[str, Any]]:
    done_query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"status": "done"}},
                ]
            }
        },
        "sort": [{"page_no": "asc"}],
    }

    page_texts: List[str] = []
    meta: Dict[str, Any] = {}
    for hit in os_pages_staging.scan(query=done_query, size=500):
        src = hit.get("_source", {}) or {}
        if not meta:
            meta = src
        page_texts.append(str(src.get("ocr_text") or ""))

    all_query = {
        "query": {"term": {"doc_id": doc_id}},
        "_source": ["status"],
    }
    total = 0
    done = 0
    failed = 0
    for hit in os_pages_staging.scan(query=all_query, size=500):
        total += 1
        st = str((hit.get("_source", {}) or {}).get("status") or "")
        if st == "done":
            done += 1
        elif st == "failed":
            failed += 1

    return page_texts, total, done, failed, meta


def parse_image_description(
    *,
    config_path: Optional[Path] = None,
    input_pdf: Optional[Path] = None,
    mode: Optional[str] = None,
    doc_id_override: Optional[str] = None
) -> ParseArtifactsResult:
    cfg_path = config_path or Path("config/config.yaml")
    cfg = load_config(cfg_path)

    run_mode = (mode or str(get_value(cfg, "workflow.mode", "fresh"))).strip()
    if run_mode not in ("fresh", "from_pages_staging"):
        raise ValueError(f"Invalid mode: {run_mode}. use 'fresh' or 'from_pages_staging'")

    data_folder = Path(get_value(cfg, "paths.data_folder", "."))
    output_dir = Path(get_value(cfg, "paths.output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_doc_path: Optional[Path] = None
    if input_pdf is None:
        input_pdf_name = str(get_value(cfg, "paths.input_pdf", "")).strip()
        if input_pdf_name:
            input_doc_path = data_folder / input_pdf_name
    else:
        input_doc_path = input_pdf

    source_uri = ""
    doc_title = ""
    doc_sha = ""
    doc_id = ""

    if doc_id_override:
        doc_id = str(doc_id_override).strip()

    if input_doc_path is not None:
        if not input_doc_path.exists():
            raise FileNotFoundError(f"PDF not found: {input_doc_path}")
        source_uri = str(input_doc_path)
        doc_title = input_doc_path.stem
        doc_sha = sha256_file(input_doc_path)
        if not doc_id:
            doc_id = doc_sha
    else:
        if not doc_id:
            raise ValueError("from_pages_staging without PDF requires doc_id_override.")
        doc_sha = doc_id

    assets_root = output_dir / "assets" / doc_id
    assets_root.mkdir(parents=True, exist_ok=True)
    pages_dir = assets_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    vlm_url = get_value(cfg, "vlm.url", "")
    vlm_model = get_value(cfg, "vlm.model", "")
    vlm_api_key = get_value(cfg, "vlm.api_key", "")
    timeout_sec = int(get_value(cfg, "vlm.timeout_sec", 3600))

    prompt = get_value(cfg, "prompt_ocr", "")
    max_tokens = int(get_value(cfg, "generation.max_tokens", 2048))
    temperature = float(get_value(cfg, "generation.temperature", 0.0))

    do_image_desc = bool(get_value(cfg, "image_desc.enabled", False))
    img_desc_prompt = get_value(cfg, "prompt_img_desc", "")
    img_desc_max_tokens = int(get_value(cfg, "image_desc.max_tokens", 256))
    img_desc_temperature = float(get_value(cfg, "image_desc.temperature", 0.0))

    pad_ratio = float(get_value(cfg, "image_crop.pad_ratio", 0.01))
    debug_draw_bboxes = bool(get_value(cfg, "debug.draw_bboxes", False))
    render_scale = float(get_value(cfg, "render.scale", 1.0))

    filter_enabled = bool(get_value(cfg, "image_filter.enabled", False))
    stddev_min = float(get_value(cfg, "image_filter.stddev_min", 0.0))
    stddev_mode = str(get_value(cfg, "image_filter.mode", "grayscale"))

    minio_cfg = MinIOConfig(
        endpoint=str(get_value(cfg, "minio.endpoint", "")),
        access_key=str(get_value(cfg, "minio.access_key", "")),
        secret_key=str(get_value(cfg, "minio.secret_key", "")),
        bucket=str(get_value(cfg, "minio.bucket", "")),
        secure=bool(get_value(cfg, "minio.secure", "")),
    )
    if not minio_cfg.endpoint or not minio_cfg.bucket:
        raise ValueError("minio.endpoint/minio.bucket is required.")
    minio = MinIOWriter(minio_cfg)
    minio_reader = MinIOReader(minio_cfg)

    os_url = str(get_value(cfg, "opensearch.url", ""))
    if not os_url:
        raise ValueError("opensearch.url is required.")

    os_username = get_value(cfg, "opensearch.username", None)
    os_password = get_value(cfg, "opensearch.password", None)
    os_verify = bool(get_value(cfg, "opensearch.verify_certs", None))

    text_index = str(get_value(cfg, "opensearch.text_index", "pdf_chunks_v1"))
    image_index = str(get_value(cfg, "opensearch.image_index", "pdf_images_v1"))
    pages_staging_index = str(get_value(cfg, "opensearch.pages_staging_index", "pdf_pages_staging_v1"))
    images_staging_index = str(get_value(cfg, "opensearch.image_staging_index", "pdf_images_staging_v1"))
    bulk_size = int(get_value(cfg, "opensearch.bulk_size", 200))

    os_text = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=text_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_image = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=image_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_pages_staging = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=pages_staging_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_images_stage = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=images_staging_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))

    os_text.ensure_index(body=build_pdf_chunks_v1_body())
    os_image.ensure_index(body=build_pdf_images_v1_body())
    os_pages_staging.ensure_index(body=build_pdf_pages_staging_v1_body())
    os_images_stage.ensure_index(body=build_pdf_images_staging_v1_body())

    emb_cfg = OllamaEmbeddingConfig(
        base_url=str(get_value(cfg, "embed_text.ollama_base_url", "http://localhost:11434")),
        model=str(get_value(cfg, "embed_text.model", "")),
        timeout_sec=int(get_value(cfg, "embed_text.timeout_sec", 120)),
        truncate=bool(get_value(cfg, "embed_text.truncate", True)),
    )
    if not emb_cfg.model:
        raise ValueError("embed_text.model is required. (e.g. qwen3-embedding:8b)")
    emb_provider = OllamaEmbeddingProvider(emb_cfg)
    text_embedding_model = emb_cfg.model

    text_expected_dim = int(get_value(cfg, "embed_text.expected_dim", 4096))
    if text_expected_dim != 4096:
        raise ValueError(f"text embedding dim must be 4096. got={text_expected_dim}")
    text_max_batch = int(get_value(cfg, "embed_text.max_batch_size", 32))

    image_embed_cfg = ImageEmbedConfig(
        url=str(get_value(cfg, "embed_image.url", "http://127.0.0.1:8088/embed")),
        timeout_sec=float(get_value(cfg, "embed_image.timeout_sec", 60.0)),
        expected_dim=int(get_value(cfg, "embed_image.expected_dim", 1024)),
        dimension=int(get_value(cfg, "embed_image.dimension", 1024)),
        max_images_per_request=int(get_value(cfg, "embed_image.max_images_per_request", 8)),
        retry_once=bool(get_value(cfg, "embed_image.retry_once", True)),
        throttle_sec=float(get_value(cfg, "embed_image.throttle_sec", 0.0)),
        model=str(get_value(cfg, "embed_image.model", "jinaai/jina-clip-v2")),
    )
    if image_embed_cfg.expected_dim != 1024:
        raise ValueError(f"image embedding dim must be 1024. got={image_embed_cfg.expected_dim}")
    image_embedding_model = str(get_value(cfg, "embed_image.model", "jinaai/jina-clip-v2"))

    pdf_uri = ""
    if input_doc_path is not None and input_doc_path.exists():
        pdf_key = minio.build_pdf_key(doc_id, filename=input_doc_path.name)
        pdf_put = minio.upload_file_to_key(
            str(input_doc_path),
            object_key=pdf_key,
            content_type="application/pdf",
        )
        pdf_uri = s3_uri(pdf_put["bucket"], pdf_put["key"])
    else:
        pdf_uri = ""

    page_texts: List[str] = []
    page_count = 0

    extracted_image_count = 0
    generated_desc_count = 0
    staged_image_count = 0

    staged_page_count = 0
    failed_page_count = 0

    desc_cache: Dict[str, Dict[str, str]] = {}

    if run_mode == "fresh":
        if input_doc_path is None:
            raise ValueError("fresh mode requires input_pdf (or paths.input_pdf).")

        _log.info("Start OCR + Image ingest. doc_id=%s mode=%s", doc_id, run_mode)
        page_pngs = pdf_to_page_pngs(input_doc_path, scale=render_scale)

        for idx, item in enumerate(tqdm(page_pngs, total=len(page_pngs), desc="OCR + IMG", unit="page"), start=1):
            page_no, payload = coerce_page_no_and_payload(item, fallback_page_no=idx)
            png_path = materialize_png_payload(payload, out_dir=pages_dir, page_no=page_no)
            page_count += 1

            page_id = f"{doc_id}:p{int(page_no):04d}"
            now = now_utc()

            try:
                txt = ocr_page(
                    png_path.read_bytes(),
                    vlm_url, vlm_model, vlm_api_key,
                    prompt, max_tokens, temperature, timeout_sec
                )

                os_pages_staging.bulk_upsert([{
                    "_id": page_id,
                    "_source": {
                        "doc_id": doc_id,
                        "page_id": page_id,
                        "doc_title": doc_title,
                        "source_uri": source_uri,
                        "pdf_uri": pdf_uri,
                        "page_no": int(page_no),
                        "ocr_text": txt,
                        "ocr_model": str(vlm_model or ""),
                        "prompt": str(prompt or ""),
                        "status": "done",
                        "attempts": 1,
                        "last_error": "",
                        "created_at": now,
                        "updated_at": now,
                    }
                }], batch_size=bulk_size)
                staged_page_count += 1

            except Exception as e:
                os_pages_staging.bulk_upsert([{
                    "_id": page_id,
                    "_source": {
                        "doc_id": doc_id,
                        "page_id": page_id,
                        "doc_title": doc_title,
                        "source_uri": source_uri,
                        "pdf_uri": pdf_uri,
                        "page_no": int(page_no),
                        "ocr_text": "",
                        "ocr_model": str(vlm_model or ""),
                        "prompt": str(prompt or ""),
                        "status": "failed",
                        "attempts": 1,
                        "last_error": f"{type(e).__name__}: {e}",
                        "created_at": now,
                        "updated_at": now,
                    }
                }], batch_size=bulk_size)
                failed_page_count += 1
                _log.warning("Skip page(ocr failed): page_no=%s err=%s", page_no, e)
                continue

            new_txt, img_records, desc_records = extract_and_store_images_from_page(
                page_png_path=png_path,
                ocr_text=txt,
                assets_root=assets_root,
                doc_id=doc_id,
                doc_title=doc_title,
                source_uri=source_uri,
                sha256=doc_sha,
                page_no=page_no,
                pad_ratio=pad_ratio,
                debug_draw_bboxes=debug_draw_bboxes,
                do_image_desc=do_image_desc,
                img_desc_prompt=img_desc_prompt,
                img_desc_max_tokens=img_desc_max_tokens,
                img_desc_temperature=img_desc_temperature,
                vlm_url=vlm_url,
                vlm_model=vlm_model,
                vlm_api_key=vlm_api_key,
                vlm_timeout_sec=timeout_sec,
                desc_cache=desc_cache,
                filter_enabled=filter_enabled,
                stddev_min=stddev_min,
                stddev_mode=stddev_mode,
            )

            extracted_image_count += len(img_records)
            generated_desc_count += len(desc_records)
            page_texts.append(new_txt)

            desc_by_image_id: Dict[str, Dict[str, Any]] = {d["image_id"]: d for d in desc_records}

            stage_docs: List[Dict[str, Any]] = []
            for order, r in enumerate(img_records, start=1):
                image_id = r["image_id"]
                img_path = Path(r["image_path"])
                if not img_path.exists():
                    continue

                crop_bytes = img_path.read_bytes()
                image_sha = sha256_bytes(crop_bytes)

                image_key = minio.build_crop_image_key(doc_id, image_id, ext="png")
                img_put = minio.upload_bytes_to_key(
                    crop_bytes,
                    object_key=image_key,
                    content_type=r.get("image_mime") or "image/png",
                )
                image_uri = s3_uri(img_put["bucket"], img_put["key"])

                desc_text = ""
                drec = desc_by_image_id.get(image_id)
                if drec and isinstance(drec.get("description"), str):
                    desc_text = drec["description"].strip()
                if not desc_text:
                    desc_text = str(r.get("caption") or "").strip()
                if not desc_text:
                    continue

                width, height = get_image_size_from_bytes(crop_bytes)
                now2 = now_utc()

                stage_src = {
                    "doc_id": doc_id,
                    "image_id": image_id,
                    "doc_title": doc_title,
                    "source_uri": source_uri,
                    "page_no": int(page_no),
                    "order": int(order),
                    "pdf_uri": pdf_uri,
                    "image_uri": image_uri,
                    "image_mime": r.get("image_mime") or "image/png",
                    "image_sha256": image_sha,
                    "width": int(width),
                    "height": int(height),
                    "bbox": r.get("bbox") or {},
                    "desc_text": desc_text,
                    "status": "pending",
                    "attempts": 0,
                    "last_error": "",
                    "created_at": now2,
                    "updated_at": now2,
                }
                stage_docs.append({"_id": image_id, "_source": stage_src})

            if stage_docs:
                os_images_stage.bulk_upsert(stage_docs, batch_size=bulk_size)
                staged_image_count += len(stage_docs)

    else:
        _log.info("Start from_pages_staging. doc_id=%s", doc_id)
        page_texts, total_pages, done_cnt, failed_cnt, meta = _load_pages_from_staging(
            os_pages_staging=os_pages_staging,
            doc_id=doc_id,
        )

        if meta:
            if not doc_title:
                doc_title = str(meta.get("doc_title") or doc_id)
            if not source_uri:
                source_uri = str(meta.get("source_uri") or "")
            if not pdf_uri:
                pdf_uri = str(meta.get("pdf_uri") or "")

        page_count = total_pages
        staged_page_count = done_cnt
        failed_page_count = failed_cnt

        extracted_image_count = 0
        generated_desc_count = 0
        staged_image_count = 0

    md_cache_path = output_dir / f"{doc_title}.{doc_id[:12]}.md"
    md_text = build_md_from_pages(page_texts)
    md_cache_path.write_text(md_text, encoding="utf-8")

    chunks = build_chunks_from_md(
        doc_id=doc_id,
        doc_title=doc_title,
        source_uri=source_uri,
        doc_sha=doc_sha,
        md_text=md_text,
        max_chunk_chars=int(get_value(cfg, "chunking.max_chunk_chars", 1200)),
        min_chunk_chars=int(get_value(cfg, "chunking.min_chunk_chars", 80)),
    )
    chunk_count = len(chunks)

    indexed_chunk_count = 0
    if chunks:
        texts = [c["text"] for c in chunks]
        vectors = embed_texts(
            emb_provider,
            texts,
            max_batch_size=text_max_batch,
            expected_dim=4096,
        )

        chunk_docs: List[Dict[str, Any]] = []
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
                "ingested_at": now_utc(),
            }
            chunk_docs.append({"_id": chunk_id, "_source": src})

        os_text.bulk_upsert(chunk_docs, batch_size=bulk_size)
        indexed_chunk_count = len(chunk_docs)

    _log.info("Start embedding from image staging. doc_id=%s", doc_id)

    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"doc_id": doc_id}},
                    {"term": {"status": "pending"}},
                ]
            }
        },
        "sort": [{"page_no": "asc"}, {"order": "asc"}],
    }

    indexed_image_count = 0
    batch_size = int(get_value(cfg, "embed_image.batch_size", 16))
    if batch_size <= 0:
        batch_size = 16

    buf: List[Dict[str, Any]] = []
    for hit in os_images_stage.scan(query=query, size=500):
        src = hit.get("_source", {})
        if not src:
            continue
        buf.append(src)

        if len(buf) < batch_size:
            continue

        indexed_image_count += _process_stage_batch(
            buf=buf,
            os_image=os_image,
            os_images_stage=os_images_stage,
            bulk_size=bulk_size,
            emb_provider=emb_provider,
            text_max_batch=text_max_batch,
            text_expected_dim=4096,
            text_embedding_model=text_embedding_model,
            image_embed_cfg=image_embed_cfg,
            image_embedding_model=image_embedding_model,
            minio_reader=minio_reader,
        )
        buf = []

    if buf:
        indexed_image_count += _process_stage_batch(
            buf=buf,
            os_image=os_image,
            os_images_stage=os_images_stage,
            bulk_size=bulk_size,
            emb_provider=emb_provider,
            text_max_batch=text_max_batch,
            text_expected_dim=4096,
            text_embedding_model=text_embedding_model,
            image_embed_cfg=image_embed_cfg,
            image_embedding_model=image_embedding_model,
            minio_reader=minio_reader,
        )

    _log.info(
        "Done. doc_id=%s mode=%s pages(total=%d done=%d failed=%d) pdf_uri=%s "
        "images(extracted=%d desc=%d indexed=%d) chunks(total=%d indexed=%d) "
        "indexes(text=%s image=%s pages_stage=%s images_stage=%s)",
        doc_id, run_mode,
        page_count, staged_page_count, failed_page_count,
        pdf_uri,
        extracted_image_count, generated_desc_count, indexed_image_count,
        chunk_count, indexed_chunk_count,
        text_index, image_index, pages_staging_index, images_staging_index,
    )

    return ParseArtifactsResult(
        doc_id=doc_id,
        doc_sha256=doc_sha,
        doc_title=doc_title or doc_id,
        source_uri=source_uri,
        output_dir=str(output_dir),
        assets_root=str(assets_root),
        pages_dir=str(pages_dir),
        md_path=str(md_cache_path),
        pdf_uri=pdf_uri,
        text_index=text_index,
        image_index=image_index,
        pages_staging_index=pages_staging_index,
        images_staging_index=images_staging_index,
        page_count=page_count,
        extracted_image_count=extracted_image_count,
        generated_desc_count=generated_desc_count,
        staged_page_count=staged_page_count,
        failed_page_count=failed_page_count,
        staged_image_count=staged_image_count,
        indexed_image_count=indexed_image_count,
        chunk_count=chunk_count,
        indexed_chunk_count=indexed_chunk_count,
        mode=run_mode,
    )


def _process_stage_batch(
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
        image_bytes_list: List[bytes] = _fetch_images_from_minio(buf=buf, minio_reader=minio_reader)
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


def _fetch_images_from_minio(*, buf: List[Dict[str, Any]], minio_reader: MinIOReader) -> List[bytes]:
    images: List[bytes] = []
    for b in buf:
        uri = str(b.get("image_uri") or "")
        bucket, key = parse_s3_uri(uri)
        images.append(minio_reader.download_bytes(bucket=bucket, key=key))
    return images
