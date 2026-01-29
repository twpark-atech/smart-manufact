# ==============================================================================
# 목적 : DOCX parse 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

from html import escape
from typing import Any, Iterable, List, Optional, Tuple

from docling.document_converter import DocumentConverter

from app.adapters.parsing.regex import RE_BULLET, RE_NUM, RE_SEP


def parse_docx_to_md(path: str) -> str:
    """DOCX 파일을 Docling으로 변환해 grounded Markdown을 반환합니다."""
    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document
    md_text = export_grounded_md(doc)
    md_text = restore_lists(md_text)
    md_text = md_table_block_to_html(md_text, escape_cells=False)
    return md_text


def restore_lists(md: str) -> str:
    """Parsing 과정에서 깨진 리스트를 복구합니다."""
    out: List[str] = []
    for line in md.splitlines():
        m = RE_BULLET.match(line)
        if m:
            indent, _, body = m.groups()
            level = max(0, len(indent) // 2)
            out.append(" " * level + f"- {body}")
            continue

        m = RE_NUM.match(line)
        if m:
            indent, num, body = m.groups()
            level = max(0, len(indent) // 2)
            out.append(" " * level + f"{num}. {body}")
            continue

        out.append(line)
    return "\n".join(out)


def _safe_int(x: Any) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return 0


def _bbox_to_list(bbox: Any) -> Optional[List[int]]:
    if bbox is None:
        return None

    for keys in (("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
        if all(hasattr(bbox, k) for k in keys):
            return [_safe_int(getattr(bbox, k)) for k in keys]

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return [_safe_int(v) for v in bbox]

    if isinstance(bbox, dict):
        for keys in (("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
            if all(k in bbox for k in keys):
                return [_safe_int(bbox[k]) for k in keys]

    return None


def _get_page_no_and_bbox(item: Any) -> Tuple[Optional[int], Optional[List[int]]]:
    prov = getattr(item, "prov", None)
    if not prov:
        return None, None
    try:
        last = prov[-1]
    except Exception:
        return None, None

    page_no = getattr(last, "page_no", None)
    bbox = getattr(last, "bbox", None)
    return (int(page_no) if page_no is not None else None), _bbox_to_list(bbox)


def _ref_block(ref: str, det: Optional[List[int]]) -> str:
    if det:
        return f'<|ref|>{ref}<|/ref|><|det|>[[{det[0]}, {det[1]}, {det[2]}, {det[3]}]]<|/det|>'
    return f"<|ref|>{ref}<|/ref|>"


def _iter_doc_items(doc: Any) -> Iterable[Any]:
    for attr in ("iterate_items", "iter_items"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            try:
                yield from fn()
                return
            except Exception:
                pass

    body = getattr(doc, "body", None)
    if body is not None:
        children = getattr(body, "children", None) or getattr(body, "items", None)
        if isinstance(children, list):
            for x in children:
                yield x
            return


def _guess_kind(item: Any) -> str:
    name = item.__class__.__name__.lower()
    if "table" in name:
        return "table"
    if "picture" in name or "image" in name or "figure" in name:
        return "image"
    if "caption" in name:
        return "image_caption"
    if "title" in name or "heading" in name or "header" in name:
        return "sub_title"
    if "list" in name and "item" in name:
        return "list_item"
    return "text"


def _flatten_text(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, str):
        return x.strip()

    if isinstance(x, (int, float, bool)):
        return str(x)

    if isinstance(x, list):
        parts = []
        for it in x:
            t = _flatten_text(it)
            if t:
                parts.append(t)
        return " ".join(parts).strip()

    for key in ("text", "content", "value", "orig", "plain_text"):
        v = getattr(x, key, None)
        if v is not None:
            t = _flatten_text(v)
            if t:
                return t

    for m in ("export_to_text", "to_text", "get_text"):
        fn = getattr(x, m, None)
        if callable(fn):
            try:
                t = fn()
                t = _flatten_text(t)
                if t:
                    return t
            except Exception:
                pass

    try:
        return str(x).strip()
    except Exception:
        return ""


def _get_text(item: Any) -> str:
    for key in ("text", "content", "value", "orig"):
        v = getattr(item, key, None)
        t = _flatten_text(v)
        if t:
            return t
    return ""


def _get_table_html(item: Any) -> str:
    for m in ("export_to_html", "to_html"):
        fn = getattr(item, m, None)
        if callable(fn):
            try:
                s = fn()
                if isinstance(s, str) and s.strip():
                    return s
            except Exception:
                pass

    for key in ("raw_html", "html"):
        v = getattr(item, key, None)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def export_grounded_md(doc: Any) -> str:
    lines: List[str] = []
    current_page: Optional[int] = None

    for item in _iter_doc_items(doc):
        kind = _guess_kind(item)
        page_no, det = _get_page_no_and_bbox(item)
        if page_no is None:
            page_no = current_page if current_page is not None else 1

        payload: List[str] = []

        if kind == "table":
            html = _get_table_html(item)
            if html:
                payload.append(html)
            else:
                txt = _get_text(item)
                if txt:
                    payload.append(txt)

        elif kind == "image_caption":
            txt = _get_text(item)
            if txt:
                payload.append(f"<center>[{txt}]</center>")

        elif kind == "sub_title":
            txt = _get_text(item)
            if txt:
                payload.append(txt if txt.lstrip().startswith("#") else f"## {txt}")

        elif kind == "list_item":
            txt = _get_text(item)
            if txt:
                payload.append(f"- {txt}")

        else:
            txt = _get_text(item)
            if txt:
                payload.append(txt)

        if not payload:
            continue

        if current_page != page_no:
            if lines:
                lines.append("\n---\n")
            lines.append(f"## Page {page_no}\n")
            current_page = page_no

        lines.append(_ref_block(kind, det))
        lines.extend(payload)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _split_md_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip("|").split("|")]


def md_table_block_to_html(md: str, *, escape_cells: bool = False) -> str:
    lines = md.splitlines()
    out: List[str] = []
    i = 0

    def cell(x: str) -> str:
        return escape(x) if escape_cells else x

    def recent_has_table_ref(out_lines: list[str], *, window: int = 3) -> bool:
        for j in range(1, min(window, len(out_lines)) + 1):
            if "<|ref|>table<|/ref|>" in out_lines[-j]:
                return True
        return False

    while i < len(lines):
        line = lines[i]
        is_table_ref = recent_has_table_ref(out, window=3)

        if is_table_ref and (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and RE_SEP.match(lines[i + 1] or "")
        ):
            header = _split_md_row(lines[i])
            i += 2

            rows: List[List[str]] = []
            while i < len(lines):
                row_line = lines[i]
                if not row_line.strip().startswith("|"):
                    break
                rows.append(_split_md_row(row_line))
                i += 1

            col_n = len(header)
            html = ["<table>"]
            html.append("<tr>" + "".join(f"<td>{cell(h)}</td>" for h in header) + "</tr>")

            for r in rows:
                if len(r) < col_n:
                    r = r + [""] * (col_n - len(r))
                elif len(r) > col_n:
                    r = r[:col_n]
                html.append("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>")

            html.append("</table>")
            out.append("".join(html))
            continue

        out.append(line)
        i += 1

    return "\n".join(out)
