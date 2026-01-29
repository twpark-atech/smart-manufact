# ==============================================================================
# 목적 : DOCX에서 Markdown으로 변환하는 유틸 (Table 위치 보존 + HTML table 출력)
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-23
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import json
import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.common.config import load_config
from app.common.hash import sha256_file
from app.common.parser import get_value
from app.common.converter import (
    normalize_table_html_to_tr_td,
    ProvenanceEntry,
    DocxMediaImage,
    ExportedImage,
    make_docx_image_id,
)

from app.parsing.regex import RE_BULLET, RE_NUM, RE_HTML_TABLE

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseDocxResult:
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str
    output_dir: str
    md_path: str
    provenance_path: str
    images_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def restore_lists(md: str) -> str:
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


def normalize_tables_in_md(md_text: str) -> str:
    if not md_text or "<table" not in md_text.lower():
        return md_text

    def repl(m: re.Match) -> str:
        raw = m.group(1)
        out = normalize_table_html_to_tr_td(raw, keep_br=True)
        return out.replace("\r", "").replace("\n", "")

    return RE_HTML_TABLE.sub(repl, md_text)


def _ref_line(ref: str, det: Optional[str] = None) -> str:
    if not det:
        return f"<|ref|>{ref}<|/ref|>"
    return f"<|ref|>{ref}<|/ref|><|det|>{det}<|/det|>"


def extract_docx_media(docx_path: Path) -> List[DocxMediaImage]:
    out: List[DocxMediaImage] = []
    with zipfile.ZipFile(docx_path, "r") as zf:
        names = [n for n in zf.namelist() if n.startswith("word/media/")]
        for name in sorted(names):
            try:
                b = zf.read(name)
            except Exception:
                continue

            ext = Path(name).suffix.lower().lstrip(".") or "bin"
            out.append(DocxMediaImage(name=name, bytes=b, ext=ext))
    return out


def _sha1_bytes16(b: bytes) -> str:
    import hashlib
    return hashlib.sha1(b).hexdigest()[:16]


def _save_docx_media_image_as_png(image_bytes: bytes, out_png: Path) -> None:
    from PIL import Image
    from io import BytesIO

    img = Image.open(BytesIO(image_bytes))
    img.save(out_png, format="PNG")


def save_all_docx_media_images(
    *,
    doc_id: str,
    docx_media: List[DocxMediaImage],
    images_dir: Path,
    images_subdir_name: str = "images",
) -> List[ExportedImage]:
    images_dir.mkdir(parents=True, exist_ok=True)

    exported: List[ExportedImage] = []
    for idx, m in enumerate(docx_media, start=1):
        key = f"{idx}:{m.name}:{_sha1_bytes16(m.bytes)}"
        image_id = make_docx_image_id(doc_id=doc_id, page_no=None, key=key)

        safe_name = image_id.replace(":", "_")
        out_png = images_dir / f"{safe_name}.png"
        out_bin = images_dir / f"{safe_name}.bin"

        if not out_png.exists() and not out_bin.exists():
            try:
                _save_docx_media_image_as_png(m.bytes, out_png)
            except Exception:
                out_bin.write_bytes(m.bytes)

        if out_png.exists():
            rel_path = f"{images_subdir_name}/{out_png.name}"
        else:
            rel_path = f"{images_subdir_name}/{out_bin.name}"

        exported.append(
            ExportedImage(
                image_id=image_id,
                rel_path=rel_path,
                page_no=None,
                bbox=None,
                location_tokens=None,
            )
        )
    return exported


_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _open_docx_document_xml(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path, "r") as zf:
        return zf.read("word/document.xml").decode("utf-8", errors="replace")


def _get_para_text(p: ET.Element) -> str:
    parts: List[str] = []
    for node in p.iter():
        tag = node.tag
        if tag.endswith("}t") and node.text:
            parts.append(node.text)
        elif tag.endswith("}tab"):
            parts.append("\t")
        elif tag.endswith("}br"):
            parts.append("\n")
    s = "".join(parts)
    return s.strip()


def _cell_text(tc: ET.Element) -> str:
    paras = tc.findall(".//w:p", _W_NS)
    lines = []
    for p in paras:
        t = _get_para_text(p)
        if t:
            lines.append(t)
    return "\n".join(lines).strip()


def _table_to_html_from_xml(tbl: ET.Element) -> str:
    rows: List[List[str]] = []

    for tr in tbl.findall(".//w:tr", _W_NS):
        row: List[str] = []
        tcs = tr.findall("./w:tc", _W_NS)
        for tc in tcs:
            txt = _cell_text(tc)
            if txt:
                txt = "<br/>".join([x.strip() for x in txt.split("\n") if x.strip()])
            row.append(txt)
        if row:
            rows.append(row)

    max_cols = max((len(r) for r in rows), default=0)
    if max_cols <= 0:
        return "<table></table>"

    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    parts: List[str] = ["<table>"]
    for r in rows:
        tds = [f"<td>{c}</td>" for c in r]
        parts.append("<tr>" + "".join(tds) + "</tr>")
    parts.append("</table>")

    return normalize_table_html_to_tr_td("".join(parts), keep_br=True).replace("\r", "").replace("\n", "")


def export_grounded_md_from_docx_xml(
    *,
    docx_path: Path,
    doc_id: str,
    provenance_out: Optional[List[ProvenanceEntry]] = None,
) -> str:
    xml = _open_docx_document_xml(docx_path)
    root = ET.fromstring(xml)

    body = root.find(".//w:body", _W_NS)
    if body is None:
        return ""

    lines: List[str] = []
    table_idx = 0

    for child in list(body):
        tag = child.tag
        if tag.endswith("}p"):
            txt = _get_para_text(child)
            if not txt:
                continue

            lines.append(_ref_line("text"))
            lines.append(txt)
            lines.append("")

            if provenance_out is not None:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="text",
                        page_no=None,
                        bbox=None,
                        loc_tokens=None,
                        ref="text",
                        text=txt,
                    )
                )

        elif tag.endswith("}tbl"):
            table_idx += 1
            html = _table_to_html_from_xml(child)
            table_id = f"docx_table_{table_idx:04d}"

            lines.append(_ref_line("table", table_id))
            lines.append(html)
            lines.append("")

            if provenance_out is not None:
                provenance_out.append(
                    ProvenanceEntry(
                        kind="table",
                        page_no=None,
                        bbox=None,
                        loc_tokens=None,
                        ref="table",
                        text=None,
                    )
                )
        else:
            # 기타 요소는 일단 무시
            continue

    return "\n".join(lines).strip()


def ingest_docx(
    *,
    config_path: Optional[Path] = None,
    input_docx: Optional[Path] = None,
) -> ParseDocxResult:
    cfg_path = config_path or Path("config/docx_config.yml")
    cfg = load_config(cfg_path)

    data_folder = Path(get_value(cfg, "paths.data_folder", "."))
    output_dir = Path(get_value(cfg, "paths.output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    input_doc_path: Optional[Path] = input_docx
    if input_doc_path is None:
        input_docx_name = str(get_value(cfg, "paths.input_docx", "")).strip()
        if input_docx_name:
            input_doc_path = data_folder / input_docx_name

    if input_doc_path is None:
        raise ValueError("input_docx is required. (arg input_docx or paths.input_docx in docx_config.yml)")
    if not input_doc_path.exists():
        raise FileNotFoundError(f"DOCX not found: {input_doc_path}")
    if input_doc_path.suffix.lower() != ".docx":
        raise ValueError(f"Only .docx is supported. got={input_doc_path.suffix}")

    source_uri = str(input_doc_path)
    doc_title = input_doc_path.stem
    doc_sha = sha256_file(input_doc_path)
    doc_id = doc_sha

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    md_cache_path = output_dir / f"{doc_title}.{doc_id[:12]}.md"
    prov_path = output_dir / f"{doc_title}.{doc_id[:12]}.provenance.json"

    _log.info("Start DOCX -> MD(XML-ordered). doc_id=%s source=%s", doc_id, source_uri)

    provenance: List[ProvenanceEntry] = []

    md_text = export_grounded_md_from_docx_xml(
        docx_path=input_doc_path,
        doc_id=doc_id,
        provenance_out=provenance,
    )
    _log.info("XML-ordered grounded MD built. chars=%d", len(md_text))

    md_text = restore_lists(md_text)

    md_text = normalize_tables_in_md(md_text)

    docx_media = extract_docx_media(input_doc_path)
    _log.info("Extracted DOCX media images: %d", len(docx_media))
    exported_images = save_all_docx_media_images(
        doc_id=doc_id,
        docx_media=docx_media,
        images_dir=images_dir,
        images_subdir_name="images",
    )

    md_cache_path.write_text(md_text, encoding="utf-8")

    prov_payload = {
        "doc_id": doc_id,
        "doc_title": doc_title,
        "source_uri": source_uri,
        "exported_images": [asdict(e) for e in exported_images],
        "items": [p.to_dict() for p in provenance],
    }
    prov_path.write_text(json.dumps(prov_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _log.info(
        "Done DOCX -> MD. md_path=%s prov_path=%s images_dir=%s prov_items=%d exported_images=%d",
        str(md_cache_path),
        str(prov_path),
        str(images_dir),
        len(provenance),
        len(exported_images),
    )

    return ParseDocxResult(
        doc_id=doc_id,
        doc_sha256=doc_sha,
        doc_title=doc_title or doc_id,
        source_uri=source_uri,
        output_dir=str(output_dir),
        md_path=str(md_cache_path),
        provenance_path=str(prov_path),
        images_dir=str(images_dir),
    )
