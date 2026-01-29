from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Dict
import logging

_log = logging.getLogger(__name__)


# -----------------------------
# Low-level helpers
# -----------------------------
def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)  # type: ignore[arg-type]
    except Exception:
        return None


def _safe_dir(obj: Any) -> List[str]:
    try:
        return sorted(dir(obj))
    except Exception:
        return []


def _is_iterable(x: Any) -> bool:
    if x is None:
        return False
    try:
        iter(x)
        return True
    except Exception:
        return False


def _try_call(fn: Any, *args: Any, **kwargs: Any) -> Tuple[bool, Any]:
    if not callable(fn):
        return False, None
    try:
        return True, fn(*args, **kwargs)
    except Exception as e:
        return False, e


def _short_repr(x: Any, limit: int = 200) -> str:
    try:
        s = repr(x)
    except Exception:
        s = f"<repr_failed type={type(x).__name__}>"
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s


def _class_name(x: Any) -> str:
    try:
        return x.__class__.__name__
    except Exception:
        return type(x).__name__


def _pick_keys(names: Sequence[str], pool: Sequence[str]) -> List[str]:
    pool_set = set(pool)
    return [n for n in names if n in pool_set]


def _iter_any_children(container: Any) -> Iterable[Any]:
    """list/tuple 뿐 아니라 iterable이면 순회"""
    if container is None:
        return
    if isinstance(container, (list, tuple)):
        for x in container:
            yield x
        return
    if _is_iterable(container):
        try:
            for x in container:
                yield x
        except Exception:
            return


# -----------------------------
# Prov extraction
# -----------------------------
@dataclass(frozen=True)
class ProvInfo:
    page_no: Optional[int]
    bbox: Optional[Any]
    raw_last: str


def _extract_prov_info(item: Any) -> Optional[ProvInfo]:
    prov = _safe_getattr(item, "prov", None)
    if not prov:
        return None
    try:
        last = prov[-1]
    except Exception:
        return None

    page_no = _safe_getattr(last, "page_no", None)
    bbox = _safe_getattr(last, "bbox", None)

    try:
        page_no_i = int(page_no) if page_no is not None else None
    except Exception:
        page_no_i = None

    return ProvInfo(page_no=page_no_i, bbox=bbox, raw_last=_short_repr(last))


# -----------------------------
# Robust item iterator (그대로 유지)
# -----------------------------
def iter_doc_items_robust(doc: Any) -> Iterable[Any]:
    """
    docling 버전/문서 타입 변화에 대비해, 가능한 경로를 순차 탐색.
    - doc.iterate_items / doc.iter_items
    - doc.pages[*].iterate_items / iter_items / items / children
    - doc.body.children / doc.body.items
    - doc 자체가 iterable인 경우
    """
    # 1) doc 전체 iterate
    for attr in ("iterate_items", "iter_items"):
        fn = _safe_getattr(doc, attr, None)
        ok, out = _try_call(fn)
        if ok:
            yield from _iter_any_children(out)
            return

    # 2) pages 경로(여긴 "순회"만, 진단 출력은 inspect_docling_document에서 함)
    pages = _safe_getattr(doc, "pages", None) or _safe_getattr(_safe_getattr(doc, "document", None), "pages", None)
    if pages is not None:
        for p in _iter_any_children(pages):
            for attr in ("iterate_items", "iter_items"):
                fn = _safe_getattr(p, attr, None)
                ok, out = _try_call(fn)
                if ok:
                    yield from _iter_any_children(out)
                    return

            for key in ("items", "children", "content"):
                v = _safe_getattr(p, key, None)
                if v is not None:
                    yield from _iter_any_children(v)
                    return

    # 3) body 경로
    body = _safe_getattr(doc, "body", None)
    if body is not None:
        for key in ("children", "items", "content"):
            v = _safe_getattr(body, key, None)
            if v is not None:
                yield from _iter_any_children(v)
                return

    # 4) doc 자체 iterable
    if _is_iterable(doc):
        yield from _iter_any_children(doc)
        return

    return


# -----------------------------
# Main inspector
# -----------------------------
def inspect_docling_document(doc: Any, *, max_items: int = 5, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    lg = logger or _log

    doc_dir = _safe_dir(doc)
    focus_attrs = _pick_keys(
        [
            "iterate_items", "iter_items", "pages", "body",
            "export_to_markdown", "export_to_text", "export_to_html", "to_html",
        ],
        doc_dir,
    )

    report: Dict[str, Any] = {
        "doc_type": _class_name(doc),
        "doc_focus_attrs": focus_attrs,
        "doc_has_iterate_items": callable(_safe_getattr(doc, "iterate_items", None)),
        "doc_has_iter_items": callable(_safe_getattr(doc, "iter_items", None)),
        "doc_has_pages": _safe_getattr(doc, "pages", None) is not None,
        "doc_has_body": _safe_getattr(doc, "body", None) is not None,
        "doc_export_to_markdown": callable(_safe_getattr(doc, "export_to_markdown", None)),
        "doc_dir_sample": doc_dir[:80],
        "pages_sample": [],
        "sample_items": [],
        "notes": [],
    }

    lg.info("[DOC] type=%s attrs=%s", report["doc_type"], report["doc_focus_attrs"])
    lg.info(
        "[DOC] has_iterate_items=%s has_iter_items=%s has_pages=%s has_body=%s export_to_markdown=%s",
        report["doc_has_iterate_items"],
        report["doc_has_iter_items"],
        report["doc_has_pages"],
        report["doc_has_body"],
        report["doc_export_to_markdown"],
    )

    # --- pages 구조 확인 (핵심) ---
    pages = _safe_getattr(doc, "pages", None) or _safe_getattr(_safe_getattr(doc, "document", None), "pages", None)
    if pages is not None:
        report["pages_type"] = _class_name(pages)
        report["pages_len"] = _safe_len(pages)
        lg.info("[DOC] pages type=%s len=%s", report["pages_type"], report["pages_len"])

        # 앞 3페이지 샘플링
        sample_pages: List[Tuple[int, Any]] = []
        try:
            for pi, p in enumerate(pages, start=1):
                sample_pages.append((pi, p))
                if pi >= 3:
                    break
        except Exception as e:
            lg.warning("[DOC] pages iterate failed: %s", e)
            sample_pages = []

        for pi, p in sample_pages:
            p_dir = _safe_dir(p)
            page_focus = _pick_keys(
                ["iterate_items", "iter_items", "children", "items", "content", "prov", "page_no", "number", "index"],
                p_dir,
            )

            children = _safe_getattr(p, "children", None)
            items = _safe_getattr(p, "items", None)
            content = _safe_getattr(p, "content", None)

            page_info: Dict[str, Any] = {
                "page_index": pi,
                "page_type": _class_name(p),
                "page_focus_attrs": page_focus,
                "children_type": _class_name(children) if children is not None else None,
                "children_len": _safe_len(children) if children is not None else None,
                "items_type": _class_name(items) if items is not None else None,
                "items_len": _safe_len(items) if items is not None else None,
                "content_type": _class_name(content) if content is not None else None,
                "content_len": _safe_len(content) if content is not None else None,
                "has_iterate_items": callable(_safe_getattr(p, "iterate_items", None)),
                "has_iter_items": callable(_safe_getattr(p, "iter_items", None)),
                "sample_item_types": [],
            }

            # page에서 실제 아이템 타입 2개만 확인
            try:
                if page_info["has_iterate_items"]:
                    ok, out = _try_call(_safe_getattr(p, "iterate_items", None))
                    if ok:
                        for si, x in enumerate(_iter_any_children(out), start=1):
                            page_info["sample_item_types"].append(_class_name(x))
                            if si >= 2:
                                break
                else:
                    src = children if children is not None else (items if items is not None else content)
                    for si, x in enumerate(_iter_any_children(src), start=1):
                        page_info["sample_item_types"].append(_class_name(x))
                        if si >= 2:
                            break
            except Exception:
                pass

            report["pages_sample"].append(page_info)
            lg.info(
                "[PAGE#%d] type=%s attrs=%s children=%s(%s) items=%s(%s) content=%s(%s) iterate_items=%s sample_item_types=%s",
                pi,
                page_info["page_type"],
                page_focus,
                page_info["children_type"], page_info["children_len"],
                page_info["items_type"], page_info["items_len"],
                page_info["content_type"], page_info["content_len"],
                page_info["has_iterate_items"],
                page_info["sample_item_types"],
            )
    else:
        lg.info("[DOC] pages: None")

    # --- body 구조(기존 그대로) ---
    body = _safe_getattr(doc, "body", None)
    if body is not None:
        report["body_type"] = _class_name(body)
        children = _safe_getattr(body, "children", None) or _safe_getattr(body, "items", None)
        report["body_children_type"] = _class_name(children) if children is not None else None
        report["body_children_len"] = _safe_len(children) if children is not None else None
        lg.info(
            "[DOC] body type=%s children type=%s len=%s",
            report["body_type"],
            report["body_children_type"],
            report["body_children_len"],
        )

    # --- 샘플 아이템 추출(기존 그대로) ---
    cnt = 0
    for it in iter_doc_items_robust(doc):
        item_dir = _safe_dir(it)
        key_candidates = ["text", "content", "value", "orig", "raw_html", "html", "prov"]
        present_keys = _pick_keys(key_candidates, item_dir)

        snap: Dict[str, Any] = {
            "type": _class_name(it),
            "present_keys": present_keys,
        }
        for k in ("text", "content", "value", "orig", "raw_html", "html"):
            v = _safe_getattr(it, k, None)
            if v is not None:
                snap[k] = _short_repr(v, 220)

        prov_info = _extract_prov_info(it)
        snap["prov"] = (
            {
                "page_no": prov_info.page_no,
                "bbox": _short_repr(prov_info.bbox, 120),
                "last": prov_info.raw_last,
            }
            if prov_info
            else None
        )

        report["sample_items"].append(snap)
        lg.info("[ITEM#%d] type=%s keys=%s prov=%s", cnt + 1, snap["type"], snap["present_keys"], "yes" if snap["prov"] else "no")

        cnt += 1
        if cnt >= max_items:
            break

    return report


def inspect_docling_result(result: Any, *, max_items: int = 5, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    lg = logger or _log
    doc = _safe_getattr(result, "document", None)
    if doc is None:
        lg.warning("[RESULT] result.document is None. result=%s", _short_repr(result))
        return {"error": "result.document is None", "result_repr": _short_repr(result)}
    return inspect_docling_document(doc, max_items=max_items, logger=lg)


if __name__ == "__main__":
    import sys
    from docling.document_converter import DocumentConverter

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python debug_docling.py /path/to/file.docx")
        raise SystemExit(2)

    path = sys.argv[1]
    converter = DocumentConverter()
    res = converter.convert(path)

    # raw iterate_items 3개 샘플(기존 그대로)
    doc = res.document
    it = doc.iterate_items()
    for idx, x in enumerate(it, start=1):
        print(f"[RAW#{idx}] type={type(x)} repr={repr(x)[:300]}")
        if isinstance(x, tuple):
            print(f"        tuple_len={len(x)} elem_types={[type(e).__name__ for e in x]}")
        if idx >= 3:
            break

    report = inspect_docling_result(res, max_items=8)
    print("\n===== INSPECTION REPORT (SUMMARY) =====")
    print("doc_type:", report.get("doc_type"))
    print("pages_type:", report.get("pages_type"))
    print("pages_len:", report.get("pages_len"))
    print("pages_sample:", report.get("pages_sample"))
    print("sample_items_count:", len(report.get("sample_items") or []))
