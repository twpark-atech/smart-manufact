# ==============================================================================
# 목적 : Table Staging 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

import json
from typing import List, Dict, Any, Tuple

from app.common.runtime import now_utc
from app.infra.storage.opensearch import OpenSearchWriter
from app.infra.storage.postgres import PostgresWriter
from app.infra.storage.ollama_embedding import OllamaEmbeddingProvider
from app.application.services.embedding_service import embed_texts
from app.adapters.parsing.table_extractor import (
    extract_html_tables,
    parse_table,
    build_table_text,
    build_row_text,
    build_table_id,
)


def stage_tables_from_ocr_text(
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


def finalize_tables_from_staging(
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
