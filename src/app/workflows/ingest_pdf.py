# ==============================================================================
# 목적 : PDF에서 OCR 후 데이터를 DB에 적재하는 코드 (OpenSearch/Postgres)
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-21
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import json, logging
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
from app.parsing.table_extractor import (
    extract_html_tables, parse_table, build_table_text, build_row_text, build_table_id
)
from app.parsing.ocr import ocr_page
from app.parsing.pdf import coerce_page_no_and_payload, materialize_png_payload

from app.storage.minio import MinIOConfig, MinIOWriter, MinIOReader
from app.storage.neo4j import Neo4jConfig, Neo4jWriter
from app.storage.opensearch import OpenSearchConfig, OpenSearchWriter
from app.storage.embedding import OllamaEmbeddingConfig, OllamaEmbeddingProvider
from app.storage.postgres import PostgresConfig, PostgresWriter

from app.indexing.chunking import build_chunks_from_md
from app.indexing.embedding import (
    ImageEmbedConfig,
    embed_images_bytes_batch,
    embed_texts,
)
from app.indexing.index_bodies import (
    build_pdf_chunks_v1_body,
    build_pdf_images_v1_body,
    build_pdf_tables_v1_body,
    build_pdf_pages_staging_v1_body,
    build_pdf_images_staging_v1_body,
    build_pdf_tables_staging_v1_body,
)
from app.indexing.opensearch_docs import load_pages_from_staging

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseArtifactsResult:
    """PDF 파싱/임베딩/인덱싱 파이프라인의 최종 산출물 DTO 클래스.
    ingest_pdf() 실행 후 생성되는 주요 식별자, 산출물 경로, 사용된 인덱스명, 처리량/성공/실패 카운트를 묶어 반환합니다.
    
    Attributes:
        doc_id: 문서 식별자.
        doc_sha256: 원본 PDF의 sha256 해시.
        doc_title: 문서 제목.
        source_uri: 입력 소스 경로.
        output_dir: 결과 루트 디렉토리.
        assets_root: 문서별 assets 루트.
        pages_dir: 페이지 PNG 저장 디렉토리.
        md_path: 페이지 OCR 결과를 합친 Markdown 캐시 파일 경로.
        pdf_uri: 업로드된 PDF의 s3://.. URI.
        text_index: 텍스트 청크 인덱스명.
        image_index: 이미지 인덱스명.
        pages_staging_index: OCR 페이지 staging 인덱스명.
        images_staging_index: 이미지 staging 인덱스명.
        table_index: 테이블 인덱스명.
        tables_staging_index: 테이블 staging 인덱스명.
        page_count: 전체 페이지 수.
        extracted_image_count: 이미지 crop 추출 개수.
        generated_desc_count: 이미지 description 생성 개수.
        staged_page_count: pages_staging에 status=done으로 적재된 페이지 수.
        failed_page_count: pages_staging에 status=failed으로 적재된 페이지 수.
        staged_image_count: images_staging에 적재된 이미지 수.
        indexed_image_count: image_index에 최종 적재된 이미지 수.
        staged_table_count: tables_staging에 적재된 테이블 수.
        indexed_table_docs_count: table_index에 적재된 table(doc_type=table) 문서 수.
        indexed_table_rows_count: table_index에 적재된 row(doc_type=row) 문서 수.
        chunk_count: 생성된 텍스트 청크 수.
        indexed_chunk_count: text_index에 적재된 텍스트 청크 수.
        mode: 실행 모드("fresh" | "from_pages_staging").
    """
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

    table_index: str
    tables_staging_index: str

    page_count: int
    extracted_image_count: int
    generated_desc_count: int

    staged_page_count: int
    failed_page_count: int
    staged_image_count: int
    indexed_image_count: int

    staged_table_count: int
    indexed_table_docs_count: int
    indexed_table_rows_count: int

    chunk_count: int
    indexed_chunk_count: int

    mode: str

    def to_dict(self) -> Dict[str, Any]:
        """dataclass를 dict로 변환합니다.
        
        Returns:
            dataclasses.asdict(self) 결과 dict.
        """
        return asdict(self)


def _stage_tables_from_ocr_text(
    *,
    doc_id: str,
    doc_title: str,
    source_uri: str,
    pdf_uri: str,
    page_no: int,
    ocr_text: str,
    os_tables_stage: OpenSearchWriter,
    bulk_size: int,
    pg_cur,
) -> int:
    """OCR 텍스트에서 HTML 테이블을 찾아 Staging(OpenSearch) + 원본(Postgres)로 적재합니다.
    
    1) extract_html_tables(ocr_text)로 <table>...</table> 블록을 추출합니다.
    2) parse_table(raw_html)로 header/rows를 파싱합니다.
    3) build_table_id(...)로 table_id + table_sha를 생성합니다.
    4) PostgresWriter.upsert_pg_table(...)로 원본 테이블을 PG에 저장합니다.
    5) tables_staging 인덱스에 status="pending" 문서로 bulk_upsert합니다.

    Args:
        doc_id: 문서 식별자.
        doc_title: 문서 제목.
        source_uri: 입력 소스 URI.
        pdf_uri: MinIO에 업로드된 PDF URI.
        page_no: 현재 페이지 번호.
        ocr_text: 해당 페이지 OCR 결과 텍스트.
        os_tables_stage: OpenSearch staging writer.
        bulk_size: OpenSearch bulk_upsert 배치 사이즈.
        pg_cur: Postgres cursor.

    Returns:
        이번 페이지에서 staging에 생성(적재)된 테이블 개수.
    """
    tables = extract_html_tables(ocr_text)
    if not tables:
        return 0

    stage_docs: List[Dict[str, Any]] = []
    created = 0
    now = now_utc()

    for order, raw_html in enumerate(tables, start=1):
        header, rows = parse_table(raw_html)
        if not header or not rows:
            continue

        table_id, table_sha = build_table_id(doc_id, page_no, order, raw_html)
        PostgresWriter.upsert_pg_table(pg_cur, table_id=table_id, raw_html=raw_html, header=header, rows=rows)

        stage_src = {
            "doc_id": doc_id,
            "table_id": table_id,
            "doc_title": doc_title,
            "source_uri": source_uri,
            "pdf_uri": pdf_uri,
            "page_no": int(page_no),
            "order": int(order),
            "table_sha256": table_sha,
            "raw_html": raw_html,
            "header_json": json.dumps(header, ensure_ascii=False),
            "rows_json": json.dumps(rows, ensure_ascii=False),
            "status": "pending",
            "attempts": 0,
            "last_error": "",
            "created_at": now,
            "updated_at": now,
        }
        stage_docs.append({"_id": table_id, "_source": stage_src})
        created += 1

    if stage_docs:
        os_tables_stage.bulk_upsert(stage_docs, batch_size=bulk_size)

    return created


def _finalize_tables_from_staging(
    *,
    doc_id: str,
    os_table: OpenSearchWriter,
    os_tables_stage: OpenSearchWriter,
    emb_provider: OllamaEmbeddingProvider,
    text_max_batch: int,
    text_expected_dim: int,
    text_embedding_model: str,
    bulk_size: int,
    max_rows_embed: int,
) -> Tuple[int, int]:
    """테이블 Staging(pending)을 최종 테이블 인덱스로 전환하며 row embedding까지 수행합니다.
    
    1) tables_staging에서 doc_id + status=pending를 조회합니다.
    2) header_json/rows_json을 로드 및 검증합니다.
    3) build_table_text(...)로 table_text 생성 후 doc_type="table" 문서를 생성합니다.
    4) rows를 순회하며 doc_type="row" 문서를 생성합니다.
    5) os_table에 table_doc + row_docs bulk_upsert를 진행합니다.
    6) tables_staging 해당 table_id 문서를 status=done 또는 failed로 업데이트합니다.

    Args:
        doc_id: 대상 문서 id.
        os_table: 최종 테이블 인덱스 writer.
        os_tables_stage: 테이블 staging 인덱스 writer.
        emb_provider: 텍스트 임베딩 제공자.
        text_max_batch: 텍스트 임베딩 최대 배치 크기.
        text_expected_dim: 임베딩 차원 검증 값.
        text_embedding_model: 저장할 임베딩 모델명.
        bulk_size: OpenSearch bulk_upsert 배치 크기.
        max_rows_embed: 임베딩을 생성할 최대 row 수.

    Returns:
        (indexed_tables, indexed_rows)

    Raises:
        ValueError: header/rows_json이 비정상인 경우.
    """
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

    indexed_tables = 0
    indexed_rows = 0
    now = now_utc()

    for hit in os_tables_stage.scan(query=query, size=200):
        src = hit.get("_source") or {}
        if not src:
            continue

        table_id = str(src.get("table_id") or "")
        if not table_id:
            continue

        try:
            header = json.loads(src.get("header_json") or "[]")
            rows = json.loads(src.get("rows_json") or "[]")
            if not isinstance(header, list) or not isinstance(rows, list) or not header:
                raise ValueError("invalid header/rows_json")

            table_text = build_table_text(header, rows, max_rows=50)
            table_doc = {
                "_id": table_id,
                "_source": {
                    "doc_type": "table",
                    "doc_id": src["doc_id"],
                    "doc_title": src.get("doc_title") or "",
                    "source_uri": src.get("source_uri") or "",
                    "pdf_uri": src.get("pdf_uri") or "",
                    "page_no": int(src.get("page_no") or 0),
                    "order": int(src.get("order") or 0),
                    "table_id": table_id,
                    "table_sha256": src.get("table_sha256") or "",
                    "header": header,
                    "row_count": int(len(rows)),
                    "col_count": int(len(header)),
                    "raw_html": src.get("raw_html") or "",
                    "table_text": table_text,
                    "ingested_at": now,
                },
            }

            row_docs: List[Dict[str, Any]] = []
            embed_targets: List[Tuple[int, Dict[str, str], str]] = []

            for r_idx, r in enumerate(rows):
                if not isinstance(r, list):
                    continue
                row_obj, row_text = build_row_text(header, [str(x) for x in r])
                if r_idx < max_rows_embed:
                    embed_targets.append((r_idx, row_obj, row_text))
                else:
                    row_docs.append(
                        {
                            "_id": f"{table_id}::r{int(r_idx):04d}",
                            "_source": {
                                "doc_type": "row",
                                "doc_id": src["doc_id"],
                                "doc_title": src.get("doc_title") or "",
                                "source_uri": src.get("source_uri") or "",
                                "pdf_uri": src.get("pdf_uri") or "",
                                "page_no": int(src.get("page_no") or 0),
                                "order": int(src.get("order") or 0),
                                "table_id": table_id,
                                "table_sha256": src.get("table_sha256") or "",
                                "header": header,
                                "row_idx": int(r_idx),
                                "row_obj_json": json.dumps(row_obj, ensure_ascii=False),
                                "row_text": row_text,
                                "row_embedding_model": text_embedding_model,
                                "ingested_at": now,
                            },
                        }
                    )

            if embed_targets:
                row_texts = [t[2] for t in embed_targets]
                row_vecs = embed_texts(
                    emb_provider,
                    row_texts,
                    max_batch_size=min(text_max_batch, len(row_texts)),
                    expected_dim=text_expected_dim,
                )

                for (r_idx, row_obj, row_text), vec in zip(embed_targets, row_vecs):
                    row_docs.append(
                        {
                            "_id": f"{table_id}::r{int(r_idx):04d}",
                            "_source": {
                                "doc_type": "row",
                                "doc_id": src["doc_id"],
                                "doc_title": src.get("doc_title") or "",
                                "source_uri": src.get("source_uri") or "",
                                "pdf_uri": src.get("pdf_uri") or "",
                                "page_no": int(src.get("page_no") or 0),
                                "order": int(src.get("order") or 0),
                                "table_id": table_id,
                                "table_sha256": src.get("table_sha256") or "",
                                "header": header,
                                "row_idx": int(r_idx),
                                "row_obj_json": json.dumps(row_obj, ensure_ascii=False),
                                "row_text": row_text,
                                "row_embedding": vec,
                                "row_embedding_model": text_embedding_model,
                                "ingested_at": now,
                            },
                        }
                    )

            os_table.bulk_upsert([table_doc], batch_size=bulk_size)
            if row_docs:
                os_table.bulk_upsert(row_docs, batch_size=bulk_size)

            indexed_tables += 1
            indexed_rows += len(row_docs)
    
            new_attempts = int(src.get("attempts") or 0)
            if (src.get("status") or "") == "pending":
                new_attempts += 1

            os_tables_stage.bulk_upsert(
                [
                    {
                        "_id": table_id,
                        "_source": {
                            **src,
                            "status": "done",
                            "attempts": new_attempts,
                            "last_error": "",
                            "updated_at": now,
                        },
                    }
                ],
                batch_size=bulk_size,
            )

        except Exception as e:
            os_tables_stage.bulk_upsert(
                [
                    {
                        "_id": table_id,
                        "_source": {
                            **src,
                            "status": "failed",
                            "attempts": int(src.get("attempts") or 0) + 1,
                            "last_error": f"{type(e).__name__}: {e}",
                            "updated_at": now,
                        },
                    }
                ],
                batch_size=bulk_size,
            )

    return indexed_tables, indexed_rows


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
    - (tables_enabled) _finalize_tables_from_staging을 호출합니다.
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
    os_pages_stage = OpenSearchWriter(OpenSearchConfig(
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
    os_pages_stage.ensure_index(body=build_pdf_pages_staging_v1_body())
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

                    os_pages_stage.bulk_upsert([{
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
                    os_pages_stage.bulk_upsert([{
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
                        staged_table_count += _stage_tables_from_ocr_text(
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
                os_pages_stage=os_pages_stage,
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
                        staged_table_count += _stage_tables_from_ocr_text(
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

        if neo4j_writer is not None:
            try:
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
            except Exception as e:
                _log.warning("Neo4j upsert skipped/failed. doc_id=%s err=%s", doc_id, e)

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

        if tables_enabled:
            _log.info("Start finalize tables from staging. doc_id=%s", doc_id)
            it, ir = _finalize_tables_from_staging(
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
    """images_staging의 pending 레코드를 배치로 처리해 최종 image_index에 적재합니다.
    
    1) buf에서 desc_text를 수집해 텍스트 임베딩을 생성합니다.
    2) MinIO에서 image_url를 내려받아 이미지 임베딩을 생성합니다.
    3) 최종 image_index에 image_docs를 upsert합니다.
    4) images_staging에 done으로 상태를 업데이트합니다.

    Args:
        buf: staging에서 읽은 이미지 레코드 리스트.
        os_image: 최종 이미지 인덱스 writer.
        os_image_stage: 이미지 staging 인덱스 writer.
        bulk_size: OpenSearch bulk_upsert 배치 크기.
        emb_provider: desc_text 임베딩을 위한 테긋트 임베딩 제공자.
        text_max_batch: 텍스트 임베딩 최대 배치 크기.
        text_expected_dim: desc_embedding 차원 검증 값.
        text_embedding_model: 저장할 desc_embedding 모델명.
        image_embed_cfg: 이미지 임베딩 API 설정.
        image_embedding_model: 저장할 image_embedding 모델명.
        minio_reader: image_uri를 다운로드할 Reader.

    Returns:
        이번 배치에서 최종 인덱스에 upsert된 이미지 문서 수.
    """
    desc_texts = [str(b.get("desc_text") or "").strip() for b in buf]

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
                stage_id = b.get("stage_id") or b.get("image_id")
                fail_docs.append({
                    "_id": stage_id,
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
            stage_id = b.get("stage_id") or b.get("image_id")
            fail_docs.append({
                "_id": stage_id,
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
        stage_id = b.get("stage_id") or image_id

        src = {
            "doc_id": b["doc_id"],
            "image_id": image_id,
            "stage_id": stage_id,
            "doc_title": b.get("doc_title") or "",
            "source_uri": b.get("source_uri") or "",
            "page_no": int(b.get("page_no") or 0),
            "order": int(b.get("order") or 0),
            "pdf_uri": b.get("pdf_uri") or "",
            "image_uri": b.get("image_uri") or "",
            "image_mime": b.get("image_mime") or "image/png",
            "image_sha256": b.get("image_sha256") or "",
            "width": int(b.get("width") or 0),
            "height": int(b.get("height") or 0),
            "bbox": b.get("bbox") or {},
            "desc_text": b.get("desc_text") or "",
            "desc_embedding": dvec,
            "image_embedding": ivec,
            "desc_embedding_model": text_embedding_model,
            "image_embedding_model": image_embedding_model,
            "ingested_at": now,
        }
        image_docs.append({"_id": stage_id, "_source": src})

        done_docs.append({
            "_id": stage_id,
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
    """staging 레코드들의 image_uri를 MinIO에서 다운로드해 bytes 리스트로 반환합니다.

    Args:
        buf: image_staging 레코드 리스트.
        minio_reader: MinIOReader 인스턴스.

    Returns:
        이미지 바이트 리스트.

    Raises:
        ValueError: image_uri가 invalid s3 uri일 경우.
    """
    images: List[bytes] = []
    for b in buf:
        uri = str(b.get("image_uri") or "")
        bucket, key = parse_s3_uri(uri)
        images.append(minio_reader.download_bytes(bucket=bucket, key=key))
    return images
