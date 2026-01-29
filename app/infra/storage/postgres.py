# ==============================================================================
# 목적 : Postgres 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import json, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence

import psycopg2
from psycopg2.extras import execute_values

from app.common.runtime import now_utc, maybe_uuid


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL 접속 설정 클래스.
    psycopg2.connect에 전달할 DSN과 연결 타임아웃을 보관합니다.
    
    Attributes:
        dsn: PostgreSQL DSN 문자열.
        connect_timeout_sec: 연결 타임아웃(초).
    """
    dsn: str
    connect_timeout_sec: int = 10


class PostgresWriter:
    """PostgreSQL에 파이프라인 메타데이터를 저장하는 Writer 클래스.

    - documents: 원본 문서 단위 메타(sha256 unique, MinIO 위치 등).
    - chunks: 문서 청크 메타(chunk_id PK, doc_sha256 FK 성격, qdrant_point_id 등).
    - ingest_runs: 파이프라인 실행 단위 상태(running/success/fail).
    - ingest_events: 실행 중 발생한 이벤트/오류 로그(단계/타깃/상태).
    - doc_table/doc_table_cell: HTML 테이블 원본과 셀 단위 정규화 저장.
    """
    def __init__(self, cfg: PostgresConfig):
        """Writer를 초기화하고 DB 연결을 생성합니다.
        
        Args:
            cfg: PostgresConfig 설정 객체.

        Raises:
            psycopg2.OperationalError: 연결에 실패한 경우.
        """
        self._cfg = cfg
        self._conn = self._connect(cfg)

    def close(self) -> None:
        """DB 연결에 종료합니다.
        
        close 과정에서 예외가 발생해도 무시합니다.
        """
        try:
            self._conn.close()
        except Exception:
            pass
    
    def cursor(self):
        """새 커서를 반환합니다.
        
        Returns:
            psycopg2 cursor 객체.
        """
        return self._conn.cursor()
    
    def commit(self) -> None:
        """현재 트랜잭션을 커밋합니다."""
        self._conn.commit()

    def rollback(self) -> None:
        """현재 트랜잭션을 롤백합니다."""
        self._conn.rollback()

    def ensure_schema(self) -> None:
        """필요한 테이블/인덱스를 생성합니다.
        
        - documents, chunks(+ idx_chunks_doc_sha256)
        - ingest_runs, ingest_events(+ idx_ingest_events_run)
        - doc_table, doc_table_cell

        Raises:
            psycopg2.Error: DDL 실행 실패.
        """
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_pk UUID PRIMARY KEY,
                sha256 CHAR(64) UNIQUE NOT NULL,
                title TEXT,
                source_uri TEXT,
                mime_type TEXT,
                size_bytes BIGINT,
                minio_bucket TEXT,
                minio_key TEXT,
                minio_etag TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY, -- canonical key (source_id)
                doc_sha256 CHAR(64) NOT NULL,
                page_start INT,
                page_end INT,
                "order" INT,
                qdrant_point_id UUID,
                embedding_model TEXT,
                vector_dim INT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_sha256 ON chunks(doc_sha256);
            """,
            """
            CREATE TABLE IF NOT EXISTS ingest_runs (
                run_id UUID PRIMARY KEY,
                pipeline_version TEXT,
                config_hash TEXT,
                started_at TIMESTAMPTZ NOT NULL,
                ended_at TIMESTAMPTZ,
                status TEXT NOT NULL, -- running/success/fail
                input_count INT,
                success_count INT,
                fail_count INT,
                meta JSONB
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS ingest_events (
                event_id UUID PRIMARY KEY,
                run_id UUID NOT NULL,
                doc_sha256 CHAR(64),
                chunk_id TEXT,
                stage TEXT NOT NULL,
                target TEXT NOT NULL, -- minio/opensearch/qdrant/neo4j/postgres
                status TEXT NOT NULL, -- success/fail/retry
                occurred_at TIMESTAMPTZ NOT NULL,
                error_message TEXT,
                meta JSONB
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_ingest_events_run ON ingest_events(run_id);
            """,
            """
            CREATE TABLE IF NOT EXISTS doc_table(
                table_id TEXT PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL,
                raw_html TEXT NOT NULL,
                header_json JSONB NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS doc_table_cell (
                table_id TEXT NOT NULL,
                row_idx INT NOT NULL,
                col_name TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (table_id, row_idx, col_name)
            );
            """,
            """CREATE INDEX IF NOT EXISTS idx_doc_table_created_at ON doc_table(created_at DESC);""",
            """CREATE INDEX IF NOT EXISTS idx_doc_table_cell_table_id ON doc_table(table_id);""",
            """CREATE INDEX IF NOT EXISTS idx_doc_table_cell_col_name ON doc_table_cell(col_name);""",
            """
            CREATE TABLE IF NOT EXISTS paragraphs (
                id          BIGSERIAL PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                chunk_id    TEXT NOT NULL,
                page_no     INT,
                bbox        JSONB,
                text        TEXT NOT NULL,
                char_start  INT,
                char_end    INT,
                created_at  TIMESTAMPTZ NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL,
                UNIQUE  (doc_id, chunk_id)
            );
            """,
            """CREATE INDEX IF NOT EXISTS idx_paragraphs_doc_id ON paragraphs(doc_id);""",
            """CREATE INDEX IF NOT EXISTS idx_paragraphs_doc_page ON paragraphs(doc_id, page_no);""",
            """
            CREATE TABLE IF NOT EXISTS sentences (
                id          BIGSERIAL PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                paragraph_id BIGINT NOT NULL REFERENCES paragraghs(id) ON DELETE CASCADE,
                sentence_idx INT NOT NULL,
                page_no     INT,
                bbox        JSONB,
                text        TEXT NOT NULL,
                char_start  INT,
                char_end    INT,
                created_at  TIMESTAMPTZ NOT NULL,
                updated_at  TIMESTAMPTZ NOT NULL,
                UNIQUE  (paragraph_id, sentence_idx)
            );
            """,
            """CREATE INDEX IF NOT EXISTS idx_sentences_doc_id ON sentences(doc_id);""",
            """CREATE INDEX IF NOT EXISTS idx_sentences_para ON sentences(paragraph_id);""",
            """CREATE INDEX IF NOT EXISTS idx_sentences_doc_page ON sentences(doc_id, page_no);""",
        ]
        with self.cursor() as cur:
            for s in stmts:
                cur.execute(s)
        self._conn.commit()

    def upsert_document(
        self,
        sha256_hex: str,
        title: Optional[str] = None,
        source_uri: Optional[str] = None,
        mime_type: Optional[str] = None,
        size_bytes: Optional[str] = None,
        minio_bucket: Optional[str] = None,
        minio_key: Optional[str] = None,
        minio_etag: Optional[str] = None, 
    ) -> uuid.UUID:
        """문서 메타를 sha256 기준으로 업서트하고 doc_pk를 반환합니다.

        documents.sha256(UNIQUE) 충돌 시 DO UPDATE로 기존 레코드를 갱신합니다.
        각 필드는 COALESCE(EXCLUDED.x, documents.x)로 처리하여, 새 값이 None이면 기존 값을 보존합니다.
        updated_at은 항상 갱신됩니다.

        Args:
            sha256_hex: 문서 콘텐츠 sha256.
            title: 문서 제목.
            source_uri: 원본 위치 URI.
            mime_type: MIME 타입.
            size_bytes: 파일 크기.
            minio_bucket: MinIO bucket.
            minio_key: MinIO object key.
            minio_etag: MinIO etag.

        Returns:
            업서트된 documents.doc_pk(UUID).

        Raises:
            psycopg2.Error: INSERT/UPDATE에 실패한 경우. 
        """
        now = now_utc()
        doc_pk = uuid.uuid4()
        sql = """
        INSERT INTO documents (
            doc_pk, sha256, title, source_uri, mime_type, size_bytes,
            minio_bucket, minio_key, minio_etag,
            created_at, updated_at
        )
        VALUES (%(doc_pk)s, %(sha256)s, %(title)s, %(source_uri)s, %(mime_type)s, %(size_bytes)s,
                %(minio_bucket)s, %(minio_key)s, %(minio_etag)s,
                %(created_at)s, %(updated_at)s)
        ON CONFLICT (sha256) DO UPDATE SET
            title = COALESCE(EXCLUDED.title, documents.title),
            source_uri = COALESCE(EXCLUDED.source_uri, documents.source_uri),
            mime_type = COALESCE(EXCLUDED.mime_type, documents.mime_type),
            size_bytes = COALESCE(EXCLUDED.size_bytes, documents.size_bytes),
            minio_bucket = COALESCE(EXCLUDED.minio_bucket, documents.minio_bucket),
            minio_key = COALESCE(EXCLUDED.minio_key, documents.minio_key),
            minio_etag = COALESCE(EXCLUDED.minio_etag, documents.minio_etag),
            updated_at = EXCLUDED.updated_at
        RETURNING doc_pk;
        """
        with self.cursor() as cur:
            cur.execute(
                sql,
                {
                    "doc_pk": str(doc_pk),
                    "sha256": sha256_hex,
                    "title": title,
                    "source_uri": source_uri,
                    "mime_type": mime_type,
                    "size_bytes": size_bytes,
                    "minio_bucket": minio_bucket,
                    "minio_key": minio_key,
                    "minio_etag": minio_etag,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            returned = cur.fetchone()
        self._conn.commit()
        return returned[0]
    
    def upsert_chunks(
        self,
        doc_sha256: str,
        chunks: List[Dict[str, Any]],
        embedding_model: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ) -> None:
        """청크 메타를 chunk_id 기준으로 업서트합니다.

        chunks.chunk_id(PK) 충돌 시 DO UPDATE로 갱신합니다.
        qdrant_point_id/embedding_model/vector_dim은 새 값이 None이면 기존 값을 보존합니다.
        created_at/updated_at은 now_utc로 기록하며, 충돌 시 updated_at만 갱신됩니다.

        Args:
            doc_sha256: 상위 문서 sha256. chunks.doc_sha256에 저장.
            chunks: 청크 메타 dict 리스트
            - 필요 키: chunk_id
            - 선택 키: page_start, page_end, order, qdrant_point_id
            embedding_model: 청크 임베딩 모델명.
            vector_dim: 벡터 차원.

        Returns:
            None

        Raises:
            KeyError: chunks 원소에 chunk_id가 없을 경우.
            psycopg2.Error: INSERT/UPDATE에 실패한 겨우.
        """
        now = now_utc()
        sql = """
        INSERT INTO chunks (
            chunk_id, doc_sha256, page_start, page_end, "order",
            qdrant_point_id, embedding_model, vector_dim,
            created_at, updated_at
        )
        VALUES (
            %(chunk_id)s, %(doc_sha256)s, %(page_start)s, %(page_end)s, %(order)s,
            %(qdrant_point_id)s, %(embedding_model)s, %(vector_dim)s,
            %(created_at)s, %(updated_at)s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_sha256 = EXCLUDED.doc_sha256,
            page_start = EXCLUDED.page_start,
            page_end = EXCLUDED.page_end,
            "order" = EXCLUDED."order",
            qdrant_point_id = COALESCE(EXCLUDED.qdrant_point_id, chunks.qdrant_point_id),
            embedding_model = COALESCE(EXCLUDED.embedding_model, chunks.embedding_model),
            vector_dim = COALESCE(EXCLUDED.vector_dim, chunks.vector_dim),
            updated_at = EXCLUDED.updated_at;
        """
        with self.cursor() as cur:
            for c in chunks:
                cur.execute(
                    sql,
                    {
                        "chunk_id": c["chunk_id"],
                        "doc_sha256": doc_sha256,
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                        "order": c.get("order"),
                        "qdrant_point_id": maybe_uuid(c.get("qdrant_point_id")),
                        "embedding_model": embedding_model,
                        "vector_dim": vector_dim,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
        self._conn.commit()

    def upsert_paragraphs(
        self,
        *,
        doc_id: str,
        paragraphs: List[Dict[str, Any]],
    ) -> None:
        now = now_utc()
        if not paragraphs:
            return {}
        
        values = []
        for p in paragraphs:
            if "chunk_id" not in p or "text" not in p:
                raise KeyError("paragraph requires chunk_id and text")
            values.append((
                doc_id,
                p["chunk_id"],
                p.get("page_no"),
                json.dumps(p.get("bbox")) if p.get("bbox") is not None else None,
                p["text"],
                p.get("char_start"),
                p.get("char_end"),
                now,
                now,
            ))
    
        sql = """
        INSERT INTO paragraphs (
            doc_id, chunk_id, page_no, bbox, text, char_start, char_end, created_at, updated_at
        )
        VALUES %s
        ON CONFLICT (doc_id, chunk_id) DO UPDATE SET
            page_no     = EXCLUDED.page_no,
            bbox        = COALESCE(EXCLUDED.bbox, paragraphs.bbox),
            text        = EXCLUDED.text,
            char_start  = EXCLUDED.char_start,
            char_end    = EXCLUDED.char_end,
            updated_at  = EXCLUDED.updated_at
        RETURNING id, chunk_id;
        """

        out: Dict[str, int] = {}
        with self.cursor() as cur:
            execute_values(cur, sql, values, page_size=500)
            rows = cur.fetchall()
            for pid, chunk_id in rows:
                out[str(chunk_id)] = int(pid)

        self._conn.commit()
        return out

    def upsert_sentences(
        self,
        *,
        doc_id: str,
        sentences: List[Dict[str, Any]],
    ) -> None:
        now = now_utc()
        if not sentences:
            return {}
        
        values = []
        for s in sentences:
            if "paragraph_id" not in s or "sentence_idx" not in s or "text" not in s:
                raise KeyError("sentence requires paragraph_id, sentence_idx and text")
            values.append((
                doc_id,
                int(s["paragraph_id"]),
                int(s.get("sentence_idx")),
                s.get("page_no"),
                json.dumps(s.get("bbox")) if s.get("bbox") is not None else None,
                s["text"],
                s.get("char_start"),
                s.get("char_end"),
                now,
                now,
            ))
    
        sql = """
        INSERT INTO sentences (
            doc_id, paragraph_id, sentence_idx, page_no, bbox, text, char_start, char_end, created_at, updated_at
        )
        VALUES %s
        ON CONFLICT (paragraph_id, sentence_idx) DO UPDATE SET
            page_no     = EXCLUDED.page_no,
            bbox        = COALESCE(EXCLUDED.bbox, sentences.bbox),
            text        = EXCLUDED.text,
            char_start  = EXCLUDED.char_start,
            char_end    = EXCLUDED.char_end,
            updated_at  = EXCLUDED.updated_at
        RETURNING id, chunk_id;
        """

        with self.cursor() as cur:
            execute_values(cur, sql, values, page_size=500)

        self._conn.commit()

    def start_ingest_run(
        self,
        pipeline_version: Optional[str] = None,
        config_hash: Optional[str] = None,
        input_count: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        """ingest_runs에 실행(run) 레코드를 생성하고 run_id를 반환합니다.
        
        status는 'running'으로 시작합니다.

        Args:
            pipeline_version: 파이프라인 버전 문자열.
            config_hash: 설정 해시.
            input_count: 입력 문서 수.
            meta: 추가 메타.

        Returns:
            생성된 run_id(UUID).

        Raises:
            psycopg2.Error: INSERT에 실패한 경우.
        """
        run_id = uuid.uuid4()
        now = now_utc()
        sql = """
        INSERT INTO ingest_runs (
            run_id, pipeline_version, config_hash, started_at, status, input_count, meta
        )
        VALUES (%(run_id)s, %(pipeline_version)s, %(config_hash)s, %(started_at)s,
                'running', %(input_count)s, %(meta)s);
        """
        with self.cursor() as cur:
            cur.execute(
                sql,
                {
                    "run_id": str(run_id),
                    "pipeline_version": pipeline_version,
                    "config_hash": config_hash,
                    "started_at": now,
                    "input_count": input_count,
                    "meta": json.dumps(meta or {}),
                },
            )
        self._conn.commit()
        return run_id
    
    def finish_ingest_run(
        self,
        run_id: uuid.UUID,
        status: str,
        success_count: Optional[int] = None,
        fail_count: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """실행(run)을 종료 처리하고 결과를 기록합니다.
        
        ended_at을 now_utc로 설정하고 status/success_count/fail_count를 저장합니다.
        meta는 전달된 값이 None이 아니면 기존 meta를 덮어쓰지 않고 COALESCE로 병합합니다.

        Args:
            run_id: start_ingest_run에서 생성한 run_id.
            status: 실행 상태.
            success_count: 성공 처리 수.
            fail_count: 실패 처리 수.
            meta: 추가 메타(JSONB).

        Returns:
            None

        Raises:
            psycopg2.Error: UPDATE에 실패한 경우.
        """
        now = now_utc()
        sql = """
        UPDATE ingest_runs
        SET ended_at = %(ended_at)s,
            status = %(status)s,
            success_count = %(success_count)s,
            fail_count = %(fail_count)s,
            meta = COALESCE(%(meta)s::jsonb, meta)
        WHERE run_id = %(run_id)s;
        """
        with self.cursor() as cur:
            cur.execute(
                sql,
                {
                    "run_id": str(run_id),
                    "ended_at": now,
                    "status": status,
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "meta": json.dumps(meta) if meta is not None else None,
                },
            )
        self._conn.commit()

    def record_event(
        self,
        run_id: uuid.UUID,
        stage: str,
        target: str,
        status: str,
        doc_sha256: Optional[str] = None,
        chunk_id: Optional[str] = None,
        error_message: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        """ingest_events에 이벤트/오류 로그를 기록합니다.
        
        파이프라인의 각 단계(stage)에서 어떤 타깃에 대해 어떤 결과가 발생했는지 추적합니다.
        
        Args:
            run_id: ingest_runs.run_id.
            stage: 파이프라인 단계명.
            target: 대상 시스템.
            status: 상태.
            doc_sha256: 관련 문서 sha256.
            chunk_id: 관련 청크 id.
            error_message: 실패 시 에러 메시지.
            meta: 추가 메타(JSONB).

        Returns:
            생성된 event_id(UUID).

        Raises:
            psycopg2.Error: INSERT에 실패한 경우.
        """
        event_id = uuid.uuid4()
        now = now_utc()
        sql = """
        INSERT INTO ingest_events (
            event_id, run_id, doc_sha256, chunk_id, stage, target, status,
            occurred_at, error_message, meta
        )
        VALUES (
            %(event_id)s, %(run_id)s, %(doc_sha256)s, %(chunk_id)s, %(stage)s,
            %(target)s, %(status)s, %(occurred_at)s, %(error_message)s, %(meta)s
        );
        """
        with self.cursor() as cur:
            cur.execute(
                sql,
                {
                    "event_id": str(event_id),
                    "run_id": str(run_id),
                    "doc_sha256": doc_sha256,
                    "chunk_id": chunk_id,
                    "stage": stage,
                    "target": target,
                    "status": status,
                    "occurred_at": now,
                    "error_message": error_message,
                    "meta": json.dumps(meta or {}),
                },
            )
        self._conn.commit()
        return event_id
    
    @staticmethod
    def upsert_pg_table(
        cur, 
        *, 
        table_id: str, 
        raw_html: str, 
        header: Sequence[str], 
        rows: Sequence[Sequence[str]]
    ) -> None:
        """HTML 테이블을 doc_table / doc_table_cell에 업서트합니다.
        
        1) doc_table에 (table_id, created_at, raw_html, header_json)을 업서트합니다.
        2) rows를 순회하며 각 셀은 doc_table_cell(table_id, row_idx, col_name) PK로 업서트합니다.

        Args:
            cur: psycopg2 cursor.
            table_id: 테이블 식별자(PK).
            raw_html: 원본 HTML 문자열.
            header: 컬럼명 리스트.
            rows: 행 데이터 2D 리스트.
        
        Returns:
            None

        Raises:
            psycopg2.Error: INSERT/UPDATE에 실패한 경우.
        """
        def _clean(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, str):
                t = v.strip()
                if len(t) >= 2 and (t[0] == t[-1]) and t[0] in("'", '"'):
                    t = t[1:-1].strip()
                return t
            return str(v)
        
        cur.execute(
            """
            INSERT INTO doc_table(table_id, created_at, raw_html, header_json)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (table_id) DO UPDATE SET 
                raw_html=EXCLUDED.raw_html, 
                header_json=EXCLUDED.header_json
            """,
            (table_id, now_utc(), raw_html, json.dumps(header, ensure_ascii=False)),
        )

        for r_idx, r in enumerate(rows):
            for c_idx, col_name in enumerate(header):
                safe_col = _clean(col_name)
                val = r[c_idx] if (isinstance(r, (list, tuple)) and  c_idx < len(r)) else ""
                safe_val = _clean(val)
                
                cur.execute(
                    """
                    INSERT INTO doc_table_cell(table_id, row_idx, col_name, value) 
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (table_id, row_idx, col_name) DO UPDATE SET 
                        value=EXCLUDED.value
                    """,
                    (table_id, r_idx, col_name, val),
                )

    def _connect(self, cfg: PostgresConfig):
        """psycopg2로 DB 연결을 생성합니ㅏㄷ.
        
        Args:
            cfg: PostgresConfig.

        Returns:
            psycopg2 connection 객체.

        Raises:
            psycopg2.OperationalError: 연결에 실패한 경우.
        """
        return psycopg2.connect(cfg.dsn, connect_timeout=cfg.connect_timeout_sec)