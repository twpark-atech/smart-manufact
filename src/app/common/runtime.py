# ==============================================================================
# 목적 : Runtime 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import time, logging, uuid
from datetime import datetime, timezone
from typing import Any, Optional

_log = logging.getLogger(__name__)


def request_with_retry(
    fn,
    *,
    retries: int = 5,
    base_sleep: float = 0.8,
    max_sleep: float = 8.0,
) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            sleep = min(max_sleep, base_sleep * (2 ** attempt))
            _log.warning("Request failed (attempt=%d/%d): %s -> sleep %.1fs", attempt + 1, retries, e, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"Request failed after {retries} retries: {last_err}") from last_err


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def maybe_uuid(v: Any) -> Optional[uuid.UUID]:
    if v is None:
        return None
    if isinstance(v, uuid.UUID):
        return v
    if isinstance(v, str):
        try:
            return uuid.UUID(v)
        except Exception:
            return None
    return None


def uuid5_from_chunk_id(chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.UUID("3d6f0a2e-3b22-4b4e-8ef0-5f9a7c1c2b11"))