# ==============================================================================
# 목적 : Parse 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path

import fitz


def get_value(cfg: dict, path: str, default=None):
    """중첩 dict에서 dotted path로 값을 안전하게 가져옵니다.
    
    Args:
        cfg: 설정 dict
        path: dotted path 문자열
        default: 키가 없을 때 반환할 기본값

    Returns:
        조회된 값 또는 default

    Raises:

    Examples:
        >>> _get({"a": {"b": 1}}, "a.b", 0)
        1
    """
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pdf_to_page_pngs(pdf_path: Path, scale: float = 2.0) -> list[bytes]:
    r"""PDF를 페이지별 PNG bytes로 렌더링합니다.
    
    Args:
        pdf_path: PDF 경로
        scale: 렌더링 확대 비율

    Returns:
        페이지별 PNG bytes 리스트

    Raises:
        fitz.FileDataError: PDF 파일이 손상/읽기 불가한 경우 
        RuntimeError: PyMuPDF 내부 렌더링 실패 등

    Examples:
        >>> imgs = _pdf_to_page_pngs(Path("example.pdf"), scale=2.0)
        >>> isinstance(imgs, list)
        True
    """
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(scale, scale)
    pages = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def build_md_from_pages(page_texts: list[str]) -> str:
    """페이지별 OCR 텍스트를 Docling Chunking 친화적인 Markdown로 결합
    
    Args:
        page_texts: 페이지별 OCR 결과 텍스트 리스트

    Returns:
        결합된 Markdown 문자열

    Raises:

    Examples:
        >>> md = _build_md_from_pages(["hello", "world"])
        ## Page 1
        hello
        ## Page 2
        world
    """
    parts = []
    for i, txt in enumerate(page_texts, start=1):
        parts.append(f"## Page {i}\n\n{txt.strip()}\n")
        parts.append("\n---\n")
    return "\n".join(parts).strip() + "\n"