# ==============================================================================
# 목적 : PDF parse 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path
from typing import Union, Tuple, Any

import fitz


def pdf_to_page_pngs(pdf_path: Path, scale: float = 2.0) -> list[bytes]:
    r"""PDF를 페이지별 PNG bytes로 렌더링합니다.

    PyMuPDF(fitz)를 사용해 PDF의 각 페이지를 렌더링하고, PNG 포맷의 bytes로 반환합니다.
    scale은 렌더링 해상도에 영향을 주며(확대 비율), 값이 클수록 이미지가 커지고 처리 비용이 증가합니다.
    
    Args:
        pdf_path: 입력 PDF 파일 경로.
        scale: 렌더링 확대 비율.

    Returns:
        PDF 페이지 순서대로 정렬된 PNG bytes 리스트.

    Raises:
        fitz.FileDataError: PDF 파일이 손상되었거나 읽을 수 없는 경우. 
        RuntimeError: PyMuPDF 내부 렌더링 실패 등.
        OSError: 파일 접근/읽기 과정에서 발생하는 OS 레벨 오류.
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
    """페이지별 OCR 텍스트를 Docling Chunking 친화적인 Markdown로 결합합니다.

    각 페이지의 텍스트를 '## Page N' 헤더로 구분하고, 페이지 사이에 '---' 구분선을 삽입합니다.
    각 페이지 텍스트는 strip() 처리 후 삽입하며, 결과 문자열은 마지막에 개행('\\n')으로 끝나도록 반환합니다.
    
    Args:
        page_texts: 페이지별 OCR 결과 텍스트 리스트.

    Returns:
        결합된 Markdown 문자열.
    """
    parts = []
    for i, txt in enumerate(page_texts, start=1):
        parts.append(f"## Page {i}\n\n{txt.strip()}\n")
        parts.append("\n---\n")
    return "\n".join(parts).strip() + "\n"


def coerce_page_no_and_payload(
    item: Union[str, Path, bytes, bytearray, Tuple[int, Any]],
    fallback_page_no: int,
) -> Tuple[int, Any]:
    """입력 아이템을 (page_no, payload) 형태로 정규화합니다.

    item이 (page_no, payload) 형태의 길이 2 튜플이면 page_no를 int로 강제 변환해 반환합니다.
    그 외 타입이면 page_no는 fallback_page_no를 사용하고 payload는 item 자체를 반환합니다.

    Args:
        item: Page payload 또는 (page_no, payload) 튜플.
        fallback_page_no: item에 page_no가 없을 때 사용할 기본 페이지 번호.

    Returns:
        (page_no, payload) 튜플.

    Raises:
        ValueError: item이 튜플이고 첫 원소가 int로 변환 불가한 경우.   
    """
    if isinstance(item, tuple) and len(item) == 2:
        return int(item[0]), item[1]
    return fallback_page_no, item


def materialize_png_payload(payload: Any, *, out_dir: Path, page_no: int) -> Path:
    """PNG payload를 파일 경로(Path)로 물리화(Materialize)합니다.
    
    payload 타입에 따라 Path를 반환합니다.
    - Path: 그대로 반환
    - str: Path(str)로 변환해 반환
    - bytes/bytearray: out_dir 아래 "p{page_no:04d}.png"로 저장 후 경로 반환

    Args:
        payload: PNG payload.
        out_dir: bytes/bytesarray 저장 시 사용할 출력 디렉토리.
        page_no: 파일명 생성에 사용할 페이지 번호.

    Returns:
        PNG 파일 경로.

    Raises:
        TypeError: 지원하지 않는 payload 타입인 경우.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(payload, Path):
        return payload
    if isinstance(payload, str):
        return Path(payload)
    if isinstance(payload, (bytes, bytearray)):
        png_path = out_dir / f"p{page_no:04d}.png"
        if not png_path.exists():
            png_path.write_bytes(bytes(payload))
        return png_path
    
    raise TypeError(f"Unsupported png payload type: {type(payload)!r}")
