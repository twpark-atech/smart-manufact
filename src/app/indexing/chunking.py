# ==============================================================================
# 목적 : Chunking 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import List, Dict, Any

from app.parsing.md import split_md_by_pages
from app.parsing.regex import _RE_IMG_TOKEN


def build_chunks_from_md(
    doc_id: str,
    doc_title: str,
    source_uri: str,
    doc_sha: str,
    md_text: str,
    *,
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 80,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    order = 0

    for page_no, block in split_md_by_pages(md_text):
        text = "\n".join([ln.strip() for ln in block.splitlines() if ln.strip()]).strip()
        if len(text) < min_chunk_chars:
            continue

        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        buf = ""

        for p in paras:
            if not buf:
                buf = p
                continue
            if len(buf) + 2 + len(p) <= max_chunk_chars:
                buf = f"{buf}\n\n{p}"
            else:
                if len(buf) >= min_chunk_chars:
                    chunk_id = f"{doc_id}:p{page_no}-{page_no}:{order}"
                    normalized.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "doc_title": doc_title,
                        "source_uri": source_uri,
                        "sha256": doc_sha,
                        "page_start": page_no,
                        "page_end": page_no,
                        "order": order,
                        "text": buf,
                        "image_ids": _RE_IMG_TOKEN.findall(buf),
                    })
                    order += 1
                buf = p
        
        if buf and len(buf) >= min_chunk_chars:
            chunk_id = f"{doc_id}:p{page_no}-{page_no}:{order}"
            normalized.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "doc_title": doc_title,
                "source_uri": source_uri,
                "sha256": doc_sha,
                "page_start": page_no,
                "page_end": page_no,
                "order": order,
                "text": buf,
                "image_ids": _RE_IMG_TOKEN.findall(buf),
            })
            order += 1

    return normalized