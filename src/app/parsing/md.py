# ==============================================================================
# 목적 : MD 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import List, Tuple

from app.parsing.regex import _RE_PAGE_HEADER


def split_md_by_pages(md_text: str) -> List[Tuple[int, str]]:
    if not md_text:
        return []
    
    matches = list(_RE_PAGE_HEADER.finditer(md_text))
    if not matches:
        return [(0, md_text)]
    
    pages: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        page_no = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        block = md_text[start:end].strip("\n")
        pages.append((page_no, block))
    return pages