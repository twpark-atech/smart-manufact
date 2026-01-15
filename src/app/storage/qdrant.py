# ==============================================================================
# 목적 : Qdrant 유틸 (현재 사용 X, 추후 Chunk Size 커지면 사용)
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import logging, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class QdrantConfig:
    url: str
    collection: str
    distance: str = "COSINE"
    api_key: Optional[str] = None


class QdrantWriter:
    _UUID_NAMESPACE = uuid.UUID("3d6f0a2e-3b22-4b4e-8ef0-5f9a7c1c2b11")

    def __init__(self, cfg: QdrantConfig):
        self._cfg = cfg
        self._client = QdrantClient(url=cfg.url, api_key=cfg.api_key)

    def ensure_collection(self, vector_size: int) -> None:
        dist_key = (self._cfg.distance or "").strip().upper()
        allowed = {"COSINE", "EUCLID", "DOT"}
        if dist_key not in allowed:
            raise ValueError(f"Invalid distance='{self._cfg.distnace}'. Allowed: {sorted(allowed)}")
        
        try:
            self._client.get_collection(self._cfg.collection)
            return
        except Exception as e:
            _log.info("get_collection failed (will try create): %s", e)

        dist = getattr(models.Distance, self._cfg.distance)
        self._client.create_collection(
            collection_name=self._cfg.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=dist),
        )

    def upsert(self, points: List[Dict[str, Any]], batch_size: int = 256, wait: bool = True) -> None:

        for i in range(0, len(points), batch_size):
            chunk = points[i : i + batch_size]
            qd_points = [self._to_point_struct(p) for p in chunk]
            self._client.upsert(
                collection_name=self._cfg.collection,
                points=qd_points,
                wait=wait,
            )
            _log.info("Qdrant upserted: %d", len(chunk))

    def _to_point_struct(self, p: Dict[str, Any]):
        raw_id = p.get("id")
        vector = p.get("vector")
        payload = dict(p.get("payload", {}) or {})

        if vector is None:
            raise ValueError("Point is missing required field: 'vector'")
        
        qdrant_id, source_id = self._normalize_point_id(raw_id)
        
        if source_id is not None:
            payload.setdefault("source_id", source_id)

        return models.PointStruct(
            id=qdrant_id,
            vector=vector,
            payload=payload,
        )
    
    def _normalize_point_id(self, raw_id: Any) -> Tuple[Union[int, str], Optional[str]]:
        if isinstance(raw_id, int):
            if raw_id < 0:
                raise ValueError(f"Invalid point id (negative int): {raw_id}")
            return raw_id, None
        
        if isinstance(raw_id, uuid.UUID):
            return str(raw_id), None
        
        if isinstance(raw_id, str):
            s = raw_id.strip()
            if not s:
                raise ValueError("Point id is empty string. Provide stable id per chunk for upsert.")
            
            try:
                return str(uuid.UUID(s)), None
            except Exception:
                deterministic = uuid.uuid5(self._UUID_NAMESPACE, s)
                return str(deterministic), s
            
        s = str(raw_id)
        if not s:
            raise ValueError(f"Point id is invalid type and empty after str(): {type(raw_id)}")
        deterministic = uuid.uuid5(self._UUID_NAMESPACE, s)
        return str(deterministic), s