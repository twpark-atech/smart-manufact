# ==============================================================================
# 목적 : Docx에서 Parsing 후 MD/Artifacts를 생성하는 코드
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.common.config import load_config, get_value
from app.common.hash import sha256_file
from app.adapters.parsing.docx_parser import parse_docx_to_md
from app.domain.models import ParseDocxResult
from app.application.usecases.ingest_document import index_md_document
from app.application.services.index_body_factory import build_pdf_chunks_v1_body
from app.infra.storage.opensearch import OpenSearchConfig, OpenSearchWriter
from app.infra.storage.ollama_embedding import OllamaEmbeddingConfig, OllamaEmbeddingProvider
from app.infra.storage.neo4j import Neo4jConfig, Neo4jWriter

_log = logging.getLogger(__name__)


def ingest_docx(
    *, 
    config_path: Optional[Path] = None,
    input_docx: Optional[Path] = None,
) -> ParseDocxResult:
    """Docx를 텍스트/이미지/테이블 추출 및 임베딩 후 OpenSearch(+MinIO+PG)에 적재합니다.
    
    - 로컬 Docs를 읽어 페이지 PNG를 렌더링합니다.
    - 각 페이지 Parsing 수행 후 pages_staging에 done/failed로 기록합니다.
    - Parsing 텍스트에서 테이블 stage + 이미지 crop 추출(+desc 생성 가능) 후 images_staging에 pending 적재합니다.
    - build_md_from_pages → build_chunks_from_md → 텍스트 임베딩 → text_index upsert 합니다.
    - (tables_enabled) finalize_tables_from_staging을 호출합니다.
    - images_staging(pending) 배치로 desc embedding + image embedding 생성 후 image_index upsert 합니다.
    - images_staging을 done/failed로 업데이트합니다.

    Args:
        config_path: config.yaml 경로.
        input_docx: 입력 docx 경로.

    Returns:
        ParseArtifactsResult: 산출물 경로, 인덱스명, 처리 카운트 등을 포함한 리포트.

    Raises:
        ValueError: 필수 설정이 누락된 경우.
        FileNotFoundError: input_docs가 지정되었는데 파일이 없는 경우.
    """
    cfg_path = config_path or Path("config/docx_config.yml")
    cfg = load_config(cfg_path)
    pipeline_cfg = load_config(Path("config/config.yml"))

    data_folder = Path(get_value(cfg, "paths.data_folder", "."))
    output_dir = Path(get_value(cfg, "paths.output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_doc_path: Optional[Path] = None
    if input_docx is not None:
        input_doc_path = input_docx
    else:
        input_docx_name = str(get_value(cfg, "paths.input_docx", "")).strip()
        if input_docx_name:
            input_doc_path = data_folder / input_docx_name

    if input_doc_path is None:
        raise ValueError("input_docx is required. (arg input_docx or paths.input_docx in docx_config.yml)")
    if not input_doc_path.exists():
        raise FileNotFoundError(f"DOCX not found: {input_doc_path}")
    if input_doc_path.suffix.lower() != ".docx":
        raise ValueError(f"Only .docx is supported. got={input_doc_path.suffix}")

    source_uri = str(input_doc_path)
    doc_title = input_doc_path.stem
    doc_sha = sha256_file(input_doc_path)
    doc_id = doc_sha

    _log.info("Start DOCX -> MD. doc_id=%s source=%s", doc_id, source_uri)

    md_text = parse_docx_to_md(str(input_doc_path)).strip()

    md_cache_path = output_dir / f"{doc_title}.{doc_id[:12]}.md"
    md_cache_path.write_text(md_text + "\n", encoding="utf-8")

    _log.info("Done DOCX -> MD. md_path=%s chars=%d", str(md_cache_path), len(md_text))

    def cfg_value(path: str, default=None):
        return get_value(cfg, path, get_value(pipeline_cfg, path, default))

    chunk_count = 0
    indexed_chunk_count = 0

    os_url = str(cfg_value("opensearch.url", "") or "").strip()
    if os_url:
        text_index = str(cfg_value("opensearch.text_index", "pdf_chunks_v1"))
        bulk_size = int(cfg_value("opensearch.bulk_size", 500))
        if bulk_size <= 0:
            bulk_size = 500

        os_text = OpenSearchWriter(
            OpenSearchConfig(
                url=os_url,
                index=text_index,
                username=cfg_value("opensearch.username"),
                password=cfg_value("opensearch.password"),
                verify_certs=bool(cfg_value("opensearch.verify_certs", False)),
            )
        )
        os_text.ensure_index(body=build_pdf_chunks_v1_body())

        emb_provider = OllamaEmbeddingProvider(
            OllamaEmbeddingConfig(
                base_url=str(cfg_value("embed_text.ollama_base_url", "")),
                model=str(cfg_value("embed_text.model", "")),
                timeout_sec=int(cfg_value("embed_text.timeout_sec", 120)),
                max_batch_size=int(cfg_value("embed_text.max_batch_size", 8)),
                truncate=bool(cfg_value("embed_text.truncate", True)),
            )
        )
        text_expected_dim = int(cfg_value("embed_text.expected_dim", 4096))
        text_max_batch = int(cfg_value("embed_text.max_batch_size", 8))

        neo4j_writer = None
        if bool(cfg_value("neo4j.enabled", False)):
            neo4j_writer = Neo4jWriter(
                Neo4jConfig(
                    uri=str(cfg_value("neo4j.uri", "")),
                    username=str(cfg_value("neo4j.username", "")),
                    password=str(cfg_value("neo4j.password", "")),
                    database=str(cfg_value("neo4j.database", "neo4j")),
                )
            )

        chunk_count, indexed_chunk_count = index_md_document(
            doc_id=doc_id,
            doc_title=doc_title or doc_id,
            source_uri=source_uri,
            doc_sha=doc_sha,
            md_text=md_text,
            os_text=os_text,
            emb_provider=emb_provider,
            text_max_batch=text_max_batch,
            text_expected_dim=text_expected_dim,
            text_embedding_model=str(cfg_value("embed_text.model", "")),
            bulk_size=bulk_size,
            chunk_max_chars=int(cfg_value("chunking.max_chunk_chars", 1200)),
            chunk_min_chars=int(cfg_value("chunking.min_chunk_chars", 80)),
            neo4j_writer=neo4j_writer,
        )

    return ParseDocxResult(
        doc_id=doc_id,
        doc_sha256=doc_sha,
        doc_title=doc_title or doc_id,
        source_uri=source_uri,
        output_dir=str(output_dir),
        md_path=str(md_cache_path),
        chunk_count=chunk_count,
        indexed_chunk_count=indexed_chunk_count,
    )
