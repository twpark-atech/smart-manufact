# ==============================================================================
# 목적 : ID 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import Optional


def chunk_id_of(doc_id: str, page_start: Optional[int], page_end: Optional[int], order: int) -> str:
    ps = page_start if page_start is not None else 0
    pe = page_end if page_end is not None else ps
    return f"{doc_id}:p{ps}-{pe}:{order}"