# ==============================================================================
# 목적 : PDF에서 OCR 후 데이터를 DB에 적재하는 코드 (OpenSearch/Postgres)
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-21
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from app.common.config import load_config, get_value
from app.common.hash import sha256_file, sha256_bytes
from app.common.ids import s3_uri, parse_s3_uri
from app.common.runtime import now_utc

from app.ports.ocr import ocr_page

from app.adapters.parsing.image_extractor import extract_and_store_images_from_page, get_image_size_from_bytes
from app.adapters.parsing.pdf_parser import (
    pdf_to_page_pngs,
    build_md_from_pages,
    coerce_page_no_and_payload,
    materialize_png_payload,
)

from app.infra.storage.minio import MinIOConfig, MinIOWriter, MinIOReader
from app.infra.storage.neo4j import Neo4jConfig, Neo4jWriter
from app.infra.storage.opensearch import OpenSearchConfig, OpenSearchWriter
from app.infra.storage.ollama_embedding import OllamaEmbeddingConfig, OllamaEmbeddingProvider
from app.infra.storage.postgres import PostgresConfig, PostgresWriter

from app.application.services.embedding_service import ImageEmbedConfig
from app.application.services.index_body_factory import (
    build_pdf_chunks_v1_body,
    build_pdf_images_v1_body,
    build_pdf_tables_v1_body,
    build_pdf_pages_staging_v1_body,
    build_pdf_images_staging_v1_body,
    build_pdf_tables_staging_v1_body,
)
from app.application.services.page_stage import load_pages_from_staging
from app.application.services.table_stage import stage_tables_from_ocr_text, finalize_tables_from_staging
from app.application.services.image_stage import process_image_stage_batch
from app.application.usecases.ingest_document import index_md_document
from app.domain.models import ParseArtifactsResult

_log = logging.getLogger(__name__)


def ingest_pdf(
    *,
    config_path: Optional[Path] = None,
    input_pdf: Optional[Path] = None,
    mode: Optional[str] = None,
    doc_id_override: Optional[str] = None,
) -> ParseArtifactsResult:
    """PDF를 OCR/이미지 추출/테이블 추출/임베딩 후 OpenSearch(+MinIO+PG)에 적재합니다.
    
    fresh
    - 로컬 PDF를 읽어 페이지 PNG를 렌더링합니다.
    - 각 페이지 OCR 수행 후 pages_staging에 done/failed로 기록합니다.
    - OCR 텍스트에서 테이블 stage + 이미지 crop 추출(+desc 생성 가능) 후 images_staging에 pending 적재합니다.
    from_pages_staging
    - 기존 pages_staging에서 OCR 텍스트를 읽어 파이프라인을 재개합니다.
    - 테이블 staging을 재생성합니다.
    최종 단계
    - build_md_from_pages → build_chunks_from_md → 텍스트 임베딩 → text_index upsert 합니다.
    - (tables_enabled) finalize_tables_from_staging을 호출합니다.
    - images_staging(pending) 배치로 desc embedding + image embedding 생성 후 image_index upsert 합니다.
    - images_staging을 done/failed로 업데이트합니다.

    Args:
        config_path: config.yaml 경로.
        input_pdf: 입력 PDF 경로.
        mode: 실행 모드("fresh" | "from_pages_staging").
        doc_id_override: from_pages_staging에서 필수.

    Returns:
        ParseArtifactsResult: 산출물 경로, 인덱스명, 처리 카운트 등을 포함한 리포트.

    Raises:
        ValueError:
            - mode 값이 허용되지 않은 경우.
            - fresh 모드인데 input_pdf가 없는 경우.
            - from_pages_staging인데 doc_id_override가 없는 경우.
            - 필수 설정이 누락된 경우.
            - embedding dim 등 계약이 맞지 않은 경우.
        FileNotFoundError: input_pdf가 지정되었는데 파일이 없는 경우.
    """
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
    table_index = str(get_value(cfg, "opensearch.table_index", "pdf_tables_v1"))
    pages_staging_index = str(get_value(cfg, "opensearch.pages_staging_index", "pdf_pages_staging_v1"))
    images_staging_index = str(get_value(cfg, "opensearch.images_staging_index", "pdf_images_staging_v1"))
    tables_staging_index = str(get_value(cfg, "opensearch.tables_staging_index", "pdf_tables_staging_v1"))
    bulk_size = int(get_value(cfg, "opensearch.bulk_size", 200))

    os_text = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=text_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_image = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=image_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))
    os_table = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=table_index,
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
    os_tables_stage = OpenSearchWriter(OpenSearchConfig(
        url=os_url, index=tables_staging_index,
        username=os_username, password=os_password, verify_certs=os_verify,
    ))

    os_text.ensure_index(body=build_pdf_chunks_v1_body())
    os_image.ensure_index(body=build_pdf_images_v1_body())
    os_table.ensure_index(body=build_pdf_tables_v1_body())
    os_pages_staging.ensure_index(body=build_pdf_pages_staging_v1_body())
    os_images_stage.ensure_index(body=build_pdf_images_staging_v1_body())
    os_tables_stage.ensure_index(body=build_pdf_tables_staging_v1_body())

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
        url=str(get_value(cfg, "embed_image.ollama_base_url", "http://127.0.0.1:8088/embed")),
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

    tables_enabled = bool(get_value(cfg, "tables.enabled", True))
    max_rows_embed = int(get_value(cfg, "tables.max_rows_embed", 500))

    pg_dsn = str(get_value(cfg, "postgres.dsn", "")).strip()
    if tables_enabled and not pg_dsn:
        raise ValueError("tables.enabled=true requires postgres.dsn")

    pg = None
    pg_cur = None
    if tables_enabled:
        pg = PostgresWriter(PostgresConfig(dsn=pg_dsn))
        pg.ensure_schema()
        pg_cur = pg.cursor()

    neo4j_enabled = bool(get_value(cfg, "neo4j.enabled", False))
    neo4j_writer: Optional[Neo4jWriter] = None
    if neo4j_enabled:
        neo4j_cfg = Neo4jConfig(
            uri=str(get_value(cfg, "neo4j.uri", "")),
            username=str(get_value(cfg, "neo4j.username", "")),
            password=str(get_value(cfg, "neo4j.password", "")),
            database=str(get_value(cfg, "neo4j.database", "neo4j")),
        )
        if not neo4j_cfg.uri or not neo4j_cfg.username:
            raise ValueError("neo4j.enabled=true requires neo4j.uri/neo4j.username(/password)")
        neo4j_writer = Neo4jWriter(neo4j_cfg)
        neo4j_writer.ensure_constraints()

    page_texts: List[str] = []
    page_count = 0

    extracted_image_count = 0
    generated_desc_count = 0
    staged_image_count = 0

    staged_page_count = 0
    failed_page_count = 0

    staged_table_count = 0
    indexed_table_docs_count = 0
    indexed_table_rows_count = 0

    desc_cache: Dict[str, Dict[str, str]] = {}

    try:
        if run_mode == "fresh":
            if input_doc_path is None:
                raise ValueError("fresh mode requires input_pdf (or paths.inpud_pdf).")

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

                if tables_enabled and pg_cur is not None:
                    try:
                        staged_table_count += stage_tables_from_ocr_text(
                            doc_id=doc_id,
                            doc_title=doc_title,
                            source_uri=source_uri,
                            pdf_uri=pdf_uri,
                            page_no=int(page_no),
                            ocr_text=txt,
                            os_tables_stage=os_tables_stage,
                            bulk_size=bulk_size,
                            pg_cur=pg_cur,
                        )
                        pg.commit()
                    except Exception as e:
                        if pg:
                            pg.rollback()
                        _log.warning("Table stage failed. page_no=%s err=%s", page_no, e)

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

                desc_by_image_id: Dict[str, Dict[str, Any]] = {
                    d.get("image_id"): d for d in (desc_records or []) if isinstance(d, dict) and d.get("image_id")
                }

                stage_docs: List[Dict[str, Any]] = []
                for order, r in enumerate(img_records, start=1):
                    image_id = str(r.get("image_id") or "")
                    img_path = Path(str(r.get("image_path") or ""))
                    if not image_id or not img_path.exists():
                        continue

                    stage_id = f"{doc_id}:p{int(page_no):04d}:i{int(order):04d}"

                    # ---- MinIO upload (per-image) with exception logging
                    try:
                        crop_bytes = img_path.read_bytes()
                        image_sha = sha256_bytes(crop_bytes)

                        # 충돌 방지: crop key도 stage_id 기반 권장
                        image_key = minio.build_crop_image_key(doc_id, stage_id, ext="png")
                        img_put = minio.upload_bytes_to_key(
                            crop_bytes,
                            object_key=image_key,
                            content_type=r.get("image_mime") or "image/png",
                        )
                        image_uri = s3_uri(img_put["bucket"], img_put["key"])
                    except Exception as e:
                        _log.exception(
                            "MinIO upload failed. doc_id=%s page_no=%s stage_id=%s image_id=%s err=%s",
                            doc_id, page_no, stage_id, image_id, e
                        )
                        continue

                    # ---- desc_text: 없으면 placeholder로라도 staging에 넣기
                    desc_text = ""
                    drec = desc_by_image_id.get(image_id)
                    if drec and isinstance(drec.get("description"), str):
                        desc_text = drec["description"].strip()
                    if not desc_text:
                        desc_text = str(r.get("caption") or "").strip()
                    if not desc_text:
                        desc_text = "(no description)"
                        _log.warning(
                            "IMG_DESC empty. page=%d image_id=%s sha=%s prompt_header=%r",
                            int(page_no), image_id, image_sha, (img_desc_prompt or "")[:80]
                        )

                    width, height = get_image_size_from_bytes(crop_bytes)
                    now2 = now_utc()

                    stage_src = {
                        "doc_id": doc_id,
                        "stage_id": stage_id,
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
                    stage_docs.append({"_id": stage_id, "_source": stage_src})

                if stage_docs:
                    try:
                        os_images_stage.bulk_upsert(stage_docs, batch_size=bulk_size)
                        staged_image_count += len(stage_docs)
                    except Exception as e:
                        _log.exception(
                            "OpenSearch stage upsert failed. doc_id=%s page_no=%s n=%d err=%s",
                            doc_id, page_no, len(stage_docs), e
                        )

        else:
            _log.info("Start from_pages_staging. doc_id=%s", doc_id)
            pages, total_pages, done_cnt, failed_cnt, meta = load_pages_from_staging(
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

            for p in pages:
                if isinstance(p, dict):
                    page_texts.append(str(p.get("ocr_text") or ""))
                else:
                    page_texts.append(str(p or ""))

            if tables_enabled and pg_cur is not None:
                for p in pages:
                    if not isinstance(p, dict):
                        continue
                    pn = int(p.get("page_no") or 0)
                    txt = str(p.get("ocr_text") or "")
                    if pn <= 0 or not txt:
                        continue
                    try:
                        staged_table_count += stage_tables_from_ocr_text(
                            doc_id=doc_id,
                            doc_title=doc_title or doc_id,
                            source_uri=source_uri,
                            pdf_uri=pdf_uri,
                            page_no=pn,
                            ocr_text=txt,
                            os_tables_stage=os_tables_stage,
                            bulk_size=bulk_size,
                            pg_cur=pg_cur,
                        )
                    except Exception as e:
                        _log.warning("Table stage(from_pages_staging) failed. page_no=%s err=%s", pn, e)
                pg.commit()

        md_cache_path = output_dir / f"{doc_title}.{doc_id[:12]}.md"
        md_text = build_md_from_pages(page_texts)
        md_cache_path.write_text(md_text, encoding="utf-8")

        try:
            chunk_count, indexed_chunk_count = index_md_document(
                doc_id=doc_id,
                doc_title=doc_title,
                source_uri=source_uri,
                doc_sha=doc_sha,
                md_text=md_text,
                os_text=os_text,
                emb_provider=emb_provider,
                text_max_batch=text_max_batch,
                text_expected_dim=text_expected_dim,
                text_embedding_model=text_embedding_model,
                bulk_size=bulk_size,
                chunk_max_chars=int(get_value(cfg, "chunking.max_chunk_chars", 1200)),
                chunk_min_chars=int(get_value(cfg, "chunking.min_chunk_chars", 80)),
                neo4j_writer=neo4j_writer,
            )
        except Exception as e:
            _log.exception("Chunk indexing failed. doc_id=%s err=%s", doc_id, e)
            chunk_count = 0
            indexed_chunk_count = 0

        if tables_enabled:
            _log.info("Start finalize tables from staging. doc_id=%s", doc_id)
            it, ir = finalize_tables_from_staging(
                doc_id=doc_id,
                os_table=os_table,
                os_tables_stage=os_tables_stage,
                emb_provider=emb_provider,
                text_max_batch=text_max_batch,
                text_expected_dim=text_expected_dim,
                text_embedding_model=text_embedding_model,
                bulk_size=bulk_size,
                max_rows_embed=max_rows_embed,
            )
            indexed_table_docs_count += it
            indexed_table_rows_count += ir

        _log.info("Start embedding from image staging. doc_id=%s", doc_id)

        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}},
                        {"terms": {"status": ["pending", "done"]}},
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

            indexed_image_count += process_image_stage_batch(
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
            indexed_image_count += process_image_stage_batch(
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
            table_index=table_index,
            pages_staging_index=pages_staging_index,
            images_staging_index=images_staging_index,
            tables_staging_index=tables_staging_index,
            page_count=page_count,
            extracted_image_count=extracted_image_count,
            generated_desc_count=generated_desc_count,
            staged_page_count=staged_page_count,
            failed_page_count=failed_page_count,
            staged_image_count=staged_image_count,
            indexed_image_count=indexed_image_count,
            staged_table_count=staged_table_count,
            indexed_table_docs_count=indexed_table_docs_count,
            indexed_table_rows_count=indexed_table_rows_count,
            chunk_count=chunk_count,
            indexed_chunk_count=indexed_chunk_count,
            mode=run_mode,
        )
    finally:
        if neo4j_writer is not None:
            try:
                neo4j_writer.close()
            except Exception:
                pass

        if pg_cur is not None:
            try:
                pg_cur.close()
            except Exception:
                pass
        if pg is not None:
            pg.close()


