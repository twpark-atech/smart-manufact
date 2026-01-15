# ==============================================================================
# 목적 : Postgres 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import json, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import psycopg2

from app.common.runtime import now_utc, maybe_uuid


@dataclass(frozen=True)
class PostgresConfig:
    dsn: str
    connect_timeout_sec: int = 10


class PostgresWriter:
    def __init__(self, cfg: PostgresConfig):
        self._cfg = cfg
        self._conn = self._connect(cfg)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _adapt_uuid(self, v):
        import uuid as _uuid
        if v is None:
            return None
        if isinstance(v, _uuid.UUID):
            return str(v)
        return v

    def ensure_schema(self) -> None:
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
        ]
        with self._cursor() as cur:
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
        with self._cursor() as cur:
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
        with self._cursor() as cur:
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

    def start_ingest_run(
            self,
            pipeline_version: Optional[str] = None,
            config_hash: Optional[str] = None,
            input_count: Optional[int] = None,
            meta: Optional[Dict[str, Any]] = None,
    ) -> uuid.UUID:
        run_id = uuid.uuid4()
        now = now_utc()
        sql = """
        INSERT INTO ingest_runs (
            run_id, pipeline_version, config_hash, started_at, status, input_count, meta
        )
        VALUES (%(run_id)s, %(pipeline_version)s, %(config_hash)s, %(started_at)s,
                'running', %(input_count)s, %(meta)s);
        """
        with self._cursor() as cur:
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
        with self._cursor() as cur:
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
        with self._cursor() as cur:
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
    
    def _connect(self, cfg: PostgresConfig):
        return psycopg2.connect(cfg.dsn, connect_timeout=cfg.connect_timeout_sec)
    
    def _cursor(self):
        return self._conn.cursor()