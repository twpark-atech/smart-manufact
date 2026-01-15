# ==============================================================================
# 목적 : OpenSearch 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenSearchConfig:
    url: str
    index: str
    username: Optional[str] = None
    password: Optional[str] = None
    verify_certs: bool = False


class OpenSearchWriter:
    def __init__(self, cfg: OpenSearchConfig):

        self._cfg = cfg
        host = cfg.url.replace("http://", "").replace("https://", "")
        if ":" in host:
            h, p = host.split(":", 1)
            port = int(p)
        else:
            h, port = host, 9200

        http_auth = None
        if cfg.username and cfg.password:
            http_auth = (cfg.username, cfg.password)

        self._client = OpenSearch(
            hosts=[{"host": h, "port": port}],
            http_auth=http_auth,
            use_ssl=cfg.url.startswith("https://"),
            verify_certs=cfg.verify_certs,
        )

    def ensure_index(self) -> None:
        if self._client.indices.exists(index=self._cfg.index):
            return
        
        body = {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "doc_title": {"type": "text"},
                    "source_uri": {"type": "keyword"},
                    "sha256": {"type": "keyword"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                    "order": {"type": "integer"},
                    "text": {"type": "text"},
                }
            },
        }
        self._client.indices.create(index=self._cfg.index, body=body)

    def bulk_upsert(self, docs: List[Dict[str, Any]], batch_size: int = 500) -> None:

        actions = []
        for d in docs:
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._cfg.index,
                    "_id": d["_id"],
                    "_source": d["_source"],
                }
            )

        for i in range(0, len(actions), batch_size):
            chunk = actions[i : i + batch_size]
            success, errors = bulk(self._client, chunk, raise_on_error=False)
            if errors:
                _log.warning("OpenSearch bulk partial errors: %s", errors[:3])
            _log.info("OpenSearch bulk indexed: %d", success)