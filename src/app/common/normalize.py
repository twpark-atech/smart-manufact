# ==============================================================================
# 목적 : 정규화 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import Dict, Tuple, Any, Optional


def ensure_text(d: Dict[str, Any]) -> str:
    for k in ("text", "content", "markdown", "md"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    content = d.get("content")
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
        if texts:
            return "\n".join(texts).strip()
    return ""


def ensure_page_range(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    meta = d.get("meta") or d.get("metadata") or {}
    candidates = [
        ("page", "page"),
        ("page_start", "page_end"),
        ("pageStart", "pageEnd"),
        ("start_page", "end_page"),
    ]
    page = d.get("page") or meta.get("page")
    if isinstance(page, int):
        return page, page
    
    for a, b in candidates:
        ps = d.get(a) or meta.get(a)
        pe = d.get(b) or meta.get(b)
        if isinstance(ps, int) and isinstance(pe, int):
            return ps, pe
        
    return None, None


def ensure_order(idx: int, d: Dict[str, Any]) -> int:
    for k in ("order", "idx", "index"):
        v = d.get(k)
        if isinstance(v, int):
            return v
    return idx