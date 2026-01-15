# ==============================================================================
# 목적 : Neo4j 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import logging
from dataclasses import dataclass
from typing import Dict, List, Any

from neo4j import GraphDatabase

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    username: str
    password: str


class Neo4jWriter:
    def __init__(self, cfg: Neo4jConfig):

        self._cfg = cfg
        self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))

    def close(self) -> None:
        self._driver.close()

    def ensure_constraints(self) -> None:
        stmts = [
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        ]
        with self._driver.session() as session:
            for s in stmts:
                try:
                    session.run(s)
                except Exception as e:
                    _log.info("Constraint creation skipped/failed: %s", e)

    def upsert_document_and_chunks(
        self,
        doc: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> None:
        cypher = """
        MERGE (d:Document {doc_id: $doc.doc_id})
        SET d.title = $doc.title,
            d.source_uri = $doc.source_uri,
            d.sha256 = $doc.sha256,
            d.updated_at = datetime()
        WITH d
        UNWIND $chunks AS row
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.doc_id = row.doc_id,
            c.page_start = row.page_start,
            c.page_end = row.page_end,
            c.order = row.order
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN count(*) AS n
        """

        cypher_next = """
        UNWIND $pairs AS p
        MATCH (a:Chunk {chunk_id: p.a})
        MATCH (b:Chunk {chunk_id: p.b})
        MERGE (a)-[:NEXT]->(b)
        RETURN count(*) AS n
        """

        with self._driver.session() as session:
            session.run(cypher, doc=doc, chunks=chunks)

            ordered = sorted(chunks, key=lambda x: x["order"])
            pairs = [{"a": ordered[i]["chunk_id"], "b": ordered[i + 1]["chunk_id"]} for i in range(len(ordered) - 1)]
            if pairs:
                session.run(cypher_next, pairs=pairs)