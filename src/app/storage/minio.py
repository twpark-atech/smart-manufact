# ==============================================================================
# 목적 : MinIO 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import io
from dataclasses import dataclass
from typing import Dict, Any, Optional

from minio import Minio


@dataclass(frozen=True)
class MinIOConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False


class MinIOWriter:
    def __init__(self, cfg: MinIOConfig):

        self._cfg = cfg
        self._client = Minio(
            endpoint=cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
        )

    def ensure_bucket(self) -> None:
        if self._client.bucket_exists(self._cfg.bucket):
            return
        self._client.make_bucket(self._cfg.bucket)

    def build_object_key(self, sha256_hex: str, filename: str = "original") -> str:
        h = sha256_hex.lower()
        return f"sha256/{h[:2]}/{h[2:4]}/{h}/{filename}"
    
    def upload_bytes(
        self,
        data: bytes,
        sha256_hex: str,
        filename: str,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_bucket()
        object_key = self.build_object_key(sha256_hex, filename=filename)
        stream = io.BytesIO(data)
        size = len(data)

        result = self._client.put_object(
            bucket_name=self._cfg.bucket,
            object_name=object_key,
            data=stream,
            length=size,
            content_type=content_type or "application/octet_stream",
        )
        return {
            "bucket": self._cfg.bucket,
            "key": object_key,
            "etag": getattr(result, "etag", None),
            "size_bytes": size,
        }
    
    def upload_file(
        self,
        file_path: str,
        sha256_hex: str,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_bucket()
        name = filename or file_path.split("/")[-1]
        object_key = self.build_object_key(sha256_hex, filename=name)

        result = self._client.fput_object(
            bucket_name=self._cfg.bucket,
            object_name=object_key,
            file_path=file_path,
            content_type=content_type or "application/octet-stream",
        )
        stat = self._client.stat_object(self._cfg.bucket, object_key)
        return {
            "bucket": self._cfg.bucket,
            "key": object_key,
            "etag": getattr(result, "etag", None),
            "size_bytes": getattr(stat, "size", None),
        }
    
    def stat(self, object_key: str) -> Dict[str, Any]:
        st = self._client.stat_object(self._cfg.bucket, object_key)
        return {
            "bucket": self._cfg.bucket,
            "key": object_key,
            "etag": getattr(st, "etag", None),
            "size_bytes": getattr(st, "size", None),
            "content_type": getattr(st, "content_type", None),
            "last_modified": str(getattr(st, "last_modified", "")),
        }
