# ==============================================================================
# 목적 : DOCX -> (MD + Provenance) + LibreOffice DOCX->PDF + PDF(page_no+bbox) grounding
# 작성 : 2026-01-26
# 요구사항:
#  - 테이블은 DOCX 원래 위치 보존(HTML <table>로 MD에 삽입)
#  - UI 페이지 이동 우선: PDF page_no 기반
#  - 하이라이트 차선: PDF bbox 기반(라인/단어 bbox)
# ==============================================================================

from __future__ import annotations

import json
import logging
import re
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 프로젝트 공용 (있으면 사용)
try:
    from app.common.config import load_config
    from app.common.hash import sha256_file
    from app.common.parser import get_value
except Exception:
    load_config = None  # type: ignore
    sha256_file = None  # type: ignore
    get_value = None  # type: ignore

# converter.py에서 이미 갖고 있는 정규화 유틸을 재사용(없으면 fallback)
try:
    from app.common.converter import normalize_table_html_to_tr_td
except Exception:
    def normalize_table_html_to_tr_td(html: str, *, keep_br: bool = True) -> str:
        # 최소 fallback: 개행 제거만
        return (html or "").replace("\r", "").replace("\n", "")

_log = logging.getLogger(__name__)

# ---------------------------
# Data models
# ---------------------------

@dataclass(frozen=True)
class Block:
    block_id: str             # p:000001, tbl:000010 등
    kind: str                 # "text" | "table"
    md_payload: str           # MD에 들어갈 실제 payload (text or <table>..)
    anchor_text: str          # PDF 매칭용 텍스트(정규화 전 원문)
    table_id: Optional[str] = None  # table일 때 docx_table_0001 등


@dataclass(frozen=True)
class PdfGrounding:
    block_id: str
    page_no: int              # UI용 1-based
    bboxes: List[List[float]] # [[x0,y0,x1,y1], ...]
    confidence: float         # 0~1
    matched_tokens: int


@dataclass(frozen=True)
class ParseResult:
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str
    output_dir: str
    md_path: str
    pdf_path: str
    provenance_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------
# DOCX XML parsing (position preserved for tables)
# ---------------------------

_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

_RE_WS = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w가-힣]+")
RE_HTML_TABLE = re.compile(r"(<table\b[^>]*>.*?</table>)", re.IGNORECASE | re.DOTALL)

def normalize_tables_in_md(md_text: str) -> str:
    """
    md 안에 들어있는 <table>...</table>을 정규화:
    - normalize_table_html_to_tr_td로 <tr><td>만 남김
    - 개행 제거(검색/저장 안정성)
    """
    if not md_text or "<table" not in md_text.lower():
        return md_text

    def repl(m: re.Match) -> str:
        raw = m.group(1)
        out = normalize_table_html_to_tr_td(raw, keep_br=True)
        return out.replace("\r", "").replace("\n", "")

    return RE_HTML_TABLE.sub(repl, md_text)

def _open_docx_document_xml(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path, "r") as zf:
        return zf.read("word/document.xml").decode("utf-8", errors="replace")


def _para_text(p: ET.Element) -> str:
    """w:p 내 텍스트 결합(최소)."""
    parts: List[str] = []
    for node in p.iter():
        tag = node.tag
        if tag.endswith("}t") and node.text:
            parts.append(node.text)
        elif tag.endswith("}tab"):
            parts.append("\t")
        elif tag.endswith("}br"):
            parts.append("\n")
    return "".join(parts).strip()


def _cell_text(tc: ET.Element) -> str:
    """w:tc(셀) 내부 문단을 줄바꿈으로 연결."""
    paras = tc.findall(".//w:p", _W_NS)
    out: List[str] = []
    for p in paras:
        t = _para_text(p)
        if t:
            out.append(t)
    return "\n".join(out).strip()


def _strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "").strip()


def _table_to_html_and_anchor(tbl: ET.Element) -> Tuple[str, str]:
    """
    w:tbl -> HTML(<table><tr><td>) + anchor_text(셀 텍스트를 평탄화)
    - merge(colspan/rowspan) 정교 반영은 추후 확장 포인트
    """
    rows: List[List[str]] = []
    anchor_parts: List[str] = []

    for tr in tbl.findall("./w:tr", _W_NS):
        row: List[str] = []
        for tc in tr.findall("./w:tc", _W_NS):
            txt = _cell_text(tc)
            # anchor용 텍스트는 공백 기반으로 평탄화
            if txt:
                anchor_parts.append(txt)
                txt_html = "<br/>".join([x.strip() for x in txt.split("\n") if x.strip()])
            else:
                txt_html = ""
            row.append(txt_html)
        if row:
            rows.append(row)

    max_cols = max((len(r) for r in rows), default=0)
    if max_cols <= 0:
        html = "<table></table>"
        return normalize_table_html_to_tr_td(html, keep_br=True), ""

    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    parts: List[str] = ["<table>"]
    for r in rows:
        parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    parts.append("</table>")

    html = "".join(parts)
    html = normalize_table_html_to_tr_td(html, keep_br=True).replace("\r", "").replace("\n", "")

    anchor = " ".join(anchor_parts).strip()
    anchor = _RE_WS.sub(" ", anchor)
    return html, anchor


def parse_docx_blocks_preserve_tables(docx_path: Path) -> List[Block]:
    """
    DOCX body를 순서대로 순회하며:
    - 문단(w:p) => text block
    - 표(w:tbl) => table block (HTML)
    """
    xml = _open_docx_document_xml(docx_path)
    root = ET.fromstring(xml)
    body = root.find(".//w:body", _W_NS)
    if body is None:
        return []

    blocks: List[Block] = []
    p_idx = 0
    t_idx = 0

    for child in list(body):
        tag = child.tag
        if tag.endswith("}p"):
            txt = _para_text(child)
            if not txt:
                continue
            p_idx += 1
            block_id = f"p:{p_idx:06d}"
            blocks.append(
                Block(
                    block_id=block_id,
                    kind="text",
                    md_payload=txt,
                    anchor_text=txt,
                )
            )
        elif tag.endswith("}tbl"):
            t_idx += 1
            table_id = f"docx_table_{t_idx:04d}"
            html, anchor = _table_to_html_and_anchor(child)
            block_id = f"tbl:{t_idx:06d}"
            blocks.append(
                Block(
                    block_id=block_id,
                    kind="table",
                    md_payload=html,
                    anchor_text=anchor,
                    table_id=table_id,
                )
            )
        else:
            continue

    return blocks


def blocks_to_grounded_md(blocks: List[Block]) -> str:
    """
    기존 grounded 포맷과 유사하게 ref를 남김.
    - table은 <|ref|>table...<|det|>table_id
    - text는 <|ref|>text
    """
    out: List[str] = []
    for b in blocks:
        if b.kind == "table":
            det = b.table_id or b.block_id
            out.append(f"<|ref|>table<|/ref|><|det|>{det}<|/det|>")
            out.append(b.md_payload)
            out.append("")
        else:
            out.append("<|ref|>text<|/ref|>")
            out.append(b.md_payload)
            out.append("")
    return "\n".join(out).strip()


# ---------------------------
# LibreOffice conversion
# ---------------------------

def convert_docx_to_pdf_libreoffice(
    *,
    docx_path: Path,
    out_dir: Path,
    soffice_bin: str = "soffice",
    timeout_sec: int = 120,
) -> Path:
    """
    LibreOffice headless 변환.
    - out_dir에 <stem>.pdf 생성됨
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        soffice_bin,
        "--headless",
        "--nologo",
        "--nolockcheck",
        "--nodefault",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(docx_path),
    ]
    _log.info("LibreOffice convert: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, timeout=timeout_sec)

    pdf_path = out_dir / f"{docx_path.stem}.pdf"
    if not pdf_path.exists():
        # 일부 환경에서 확장자/파일명이 다르게 나올 수 있어 탐색
        cand = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cand:
            return cand[0]
        raise FileNotFoundError(f"PDF not generated in {out_dir}")
    return pdf_path


# ---------------------------
# PDF parsing (page_no + bbox)
# ---------------------------

def _normalize_token(s: str) -> str:
    s = (s or "").lower()
    s = _RE_PUNCT.sub(" ", s)
    s = _RE_WS.sub(" ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    s = _normalize_token(s)
    return [t for t in s.split(" ") if t]


def _union_bbox(boxes: List[Tuple[float, float, float, float]]) -> List[float]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [float(x0), float(y0), float(x1), float(y1)]


def _group_bboxes_by_line(words: List[Tuple[float, float, float, float, str, int, int, int]]) -> List[List[float]]:
    """
    PyMuPDF words: (x0,y0,x1,y1,word,block_no,line_no,word_no)
    -> line_no 단위로 bbox를 합쳐 여러 박스로 반환
    """
    by_line: Dict[Tuple[int, int], List[Tuple[float, float, float, float]]] = {}
    for x0, y0, x1, y1, w, bno, lno, wno in words:
        key = (bno, lno)
        by_line.setdefault(key, []).append((x0, y0, x1, y1))
    merged: List[List[float]] = []
    for k, boxes in by_line.items():
        merged.append(_union_bbox(boxes))
    # 위에서 아래로 정렬
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


def parse_pdf_words(pdf_path: Path) -> List[List[Tuple[float, float, float, float, str, int, int, int]]]:
    """
    PDF 전체 페이지를 words 단위로 파싱.
    return: pages_words[page_idx] = list(words_tuple)
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF(fitz)가 필요합니다. 설치 후 실행하세요.") from e

    doc = fitz.open(str(pdf_path))
    pages: List[List[Tuple[float, float, float, float, str, int, int, int]]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        words = page.get_text("words")  # (x0,y0,x1,y1,word,block_no,line_no,word_no)
        # type cast for mypy-ish
        pages.append([tuple(w) for w in words])  # type: ignore
    return pages


# ---------------------------
# Matching DOCX blocks -> PDF words (page_no + bbox)
# ---------------------------

@dataclass(frozen=True)
class _MatchResult:
    page_idx: int
    start: int
    end: int
    score: float
    matched_tokens: int


def _find_best_match_on_page(
    page_words: List[Tuple[float, float, float, float, str, int, int, int]],
    anchor_tokens: List[str],
    *,
    min_tokens: int = 6,
    max_tokens: int = 24,
) -> Optional[_MatchResult]:
    """
    단어 시퀀스 매칭:
    - anchor_tokens에서 길이를 잘라(우선순위: max_tokens까지) 페이지 words의 연속 구간과 비교
    - 1차: 완전 일치(정규화 토큰)
    - 2차: jaccard 유사도 기반 근사 매칭
    """
    if not anchor_tokens:
        return None

    # anchor 길이 결정(너무 길면 매칭 실패율 증가)
    tokens = anchor_tokens[:max_tokens]
    if len(tokens) < min_tokens:
        return None

    # 페이지 토큰화
    page_toks = [_normalize_token(w[4]) for w in page_words]
    page_toks = [t for t in page_toks]  # keep length

    n = len(page_toks)
    m = len(tokens)
    if n < m:
        return None

    # 1) exact contiguous match
    for i in range(0, n - m + 1):
        if page_toks[i : i + m] == tokens:
            return _MatchResult(page_idx=-1, start=i, end=i + m, score=1.0, matched_tokens=m)

    # 2) fuzzy: best window by Jaccard similarity on tokens set (and small bonus for order overlap)
    target_set = set(tokens)
    best: Optional[_MatchResult] = None
    for i in range(0, n - m + 1):
        window = page_toks[i : i + m]
        win_set = set(window)
        inter = len(target_set & win_set)
        union = len(target_set | win_set) or 1
        j = inter / union

        # order bonus: count exact positions matched
        pos_match = sum(1 for a, b in zip(tokens, window) if a == b)
        bonus = pos_match / m
        score = 0.85 * j + 0.15 * bonus

        if best is None or score > best.score:
            best = _MatchResult(page_idx=-1, start=i, end=i + m, score=float(score), matched_tokens=inter)

    if best and best.score >= 0.35:
        return best
    return None


def ground_blocks_to_pdf(
    blocks: List[Block],
    pdf_pages_words: List[List[Tuple[float, float, float, float, str, int, int, int]]],
) -> List[PdfGrounding]:
    """
    각 블록의 anchor를 PDF 전체 페이지에서 찾아 page_no + bbox를 생성.
    - 결과 confidence 낮으면(bad match) 제외하지 않고 confidence로 표시(UX 방어 가능)
    """
    grounded: List[PdfGrounding] = []

    for b in blocks:
        anchor_tokens = _tokenize(b.anchor_text)
        if not anchor_tokens:
            continue

        best_page: Optional[int] = None
        best_match: Optional[_MatchResult] = None

        for page_idx, page_words in enumerate(pdf_pages_words):
            mr = _find_best_match_on_page(page_words, anchor_tokens)
            if mr is None:
                continue
            mr = _MatchResult(page_idx=page_idx, start=mr.start, end=mr.end, score=mr.score, matched_tokens=mr.matched_tokens)

            if best_match is None or mr.score > best_match.score:
                best_match = mr
                best_page = page_idx

            # 완전일치면 조기 종료
            if mr.score >= 0.999:
                break

        if best_match is None or best_page is None:
            continue

        page_words = pdf_pages_words[best_page]
        matched_words = page_words[best_match.start : best_match.end]

        # 여러 줄에 걸칠 수 있으니 line 기준으로 bbox 리스트 구성
        bboxes = _group_bboxes_by_line(matched_words)

        grounded.append(
            PdfGrounding(
                block_id=b.block_id,
                page_no=int(best_page + 1),  # UI 1-based
                bboxes=bboxes,
                confidence=float(best_match.score),
                matched_tokens=int(best_match.matched_tokens),
            )
        )

    return grounded


# ---------------------------
# Orchestration
# ---------------------------

def ingest_docx_and_ground_to_pdf(
    *,
    input_docx: Path,
    output_dir: Path,
    soffice_bin: str = "soffice",
) -> ParseResult:
    if not input_docx.exists():
        raise FileNotFoundError(str(input_docx))
    if input_docx.suffix.lower() != ".docx":
        raise ValueError("Only .docx is supported")

    output_dir.mkdir(parents=True, exist_ok=True)

    # doc_id/sha
    if sha256_file is None:
        # 최소 fallback: 파일명 기반(권장 아님)
        doc_sha = input_docx.name
    else:
        doc_sha = sha256_file(input_docx)

    doc_id = doc_sha
    doc_title = input_docx.stem
    source_uri = str(input_docx)

    md_path = output_dir / f"{doc_title}.{str(doc_id)[:12]}.md"
    prov_path = output_dir / f"{doc_title}.{str(doc_id)[:12]}.provenance.json"
    pdf_out_dir = output_dir / "pdf"
    pdf_path = convert_docx_to_pdf_libreoffice(docx_path=input_docx, out_dir=pdf_out_dir, soffice_bin=soffice_bin)

    # 1) DOCX -> blocks (table position preserved) + MD 생성
    blocks = parse_docx_blocks_preserve_tables(input_docx)
    md_text = blocks_to_grounded_md(blocks)
    md_text = normalize_tables_in_md(md_text)
    md_path.write_text(md_text, encoding="utf-8")

    # 2) PDF -> words(with bbox)
    pdf_pages_words = parse_pdf_words(pdf_path)

    # 3) blocks -> pdf grounding(page_no+bbox)
    pdf_grounding = ground_blocks_to_pdf(blocks, pdf_pages_words)

    # provenance payload
    payload: Dict[str, Any] = {
        "doc_id": doc_id,
        "doc_title": doc_title,
        "source_uri": source_uri,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "blocks": [asdict(b) for b in blocks],
        "pdf_grounding": [asdict(g) for g in pdf_grounding],
        "meta": {
            "soffice_bin": soffice_bin,
            "pdf_pages": len(pdf_pages_words),
        },
    }
    prov_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _log.info("Done. md=%s pdf=%s prov=%s blocks=%d grounded=%d",
              str(md_path), str(pdf_path), str(prov_path), len(blocks), len(pdf_grounding))

    return ParseResult(
        doc_id=str(doc_id),
        doc_sha256=str(doc_sha),
        doc_title=doc_title,
        source_uri=source_uri,
        output_dir=str(output_dir),
        md_path=str(md_path),
        pdf_path=str(pdf_path),
        provenance_path=str(prov_path),
    )


# ---------------------------
# Optional: config-driven entry (프로젝트 스타일 유지)
# ---------------------------

def ingest_with_config(*, config_path: Optional[Path] = None) -> ParseResult:
    if load_config is None or get_value is None:
        raise RuntimeError("project config utils(load_config/get_value)가 없어 config 방식 실행 불가")

    cfg_path = config_path or Path("config/docx_config.yml")
    cfg = load_config(cfg_path)

    data_folder = Path(get_value(cfg, "paths.data_folder", "."))
    output_dir = Path(get_value(cfg, "paths.output_dir", "results"))

    input_name = str(get_value(cfg, "paths.input_docx", "")).strip()
    if not input_name:
        raise ValueError("paths.input_docx is required in config")

    input_docx = data_folder / input_name
    soffice_bin = str(get_value(cfg, "libreoffice.soffice_bin", "soffice"))

    return ingest_docx_and_ground_to_pdf(
        input_docx=input_docx,
        output_dir=output_dir,
        soffice_bin=soffice_bin,
    )
