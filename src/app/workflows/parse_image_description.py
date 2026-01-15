# ==============================================================================
# 목적 : PDF에서 OCR 후 MD/Artifacts를 생성하는 코드
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import os, logging
from tqdm import tqdm
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.common.config import load_config
from app.common.hash import sha256_file
from app.common.jsonl import append_jsonl, count_jsonl_lines
from app.common.parser import get_value, pdf_to_page_pngs, build_md_from_pages
from app.parsing.image_extractor import load_desc_cache, extract_and_store_images_from_page
from app.parsing.ocr import ocr_page
from app.parsing.pdf import coerce_page_no_and_payload, materialize_png_payload

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParseArtifactsResult:
    doc_id: str
    doc_sha256: str
    doc_title: str
    source_uri: str

    output_dir: str
    assets_root: str
    pages_dir: str

    md_path: str
    image_map_path: str
    image_desc_path: str

    page_count: int
    image_count: int
    desc_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

def parse_image_description(
    *,
    config_path: Optional[Path] = None,
    input_pdf: Optional[Path] = None,
) -> ParseArtifactsResult:
    cfg_path = config_path or Path("config/config.yaml")
    cfg = load_config(cfg_path)

    data_folder = Path(get_value(cfg, "paths.data_folder", "."))
    output_dir = Path(get_value(cfg, "paths.output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_pdf is None:
        input_pdf_name = get_value(cfg, "paths.input_pdf", "")
        if not input_pdf_name:
            raise ValueError("paths.input_pdf is required when input_pdf is not provided.")
        input_doc_path = data_folder / input_pdf_name
    else:
        input_doc_path = input_pdf

    if not input_doc_path.exists():
        raise FileNotFoundError(f"PDF not found: {input_doc_path}")
    
    source_uri = str(input_doc_path)
    doc_title = input_doc_path.stem
    doc_sha = sha256_file(input_doc_path)
    doc_id = doc_sha

    assets_root = output_dir / "assets" / doc_id
    assets_root.mkdir(parents=True, exist_ok=True)

    pages_dir = assets_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    image_map_path = assets_root / "image_map.jsonl"
    image_desc_path = assets_root / "image_desc.jsonl"

    vlm_url = get_value(cfg, "vlm.url", "")
    vlm_model = get_value(cfg, "vlm.model", "")
    vlm_api_key = get_value(cfg, "vlm.api_key", "")
    timeout_sec = int(get_value(cfg, "vlm.timeout_sec", 3600))

    prompt = get_value(cfg, "prompt_ocr", "")
    max_tokens = int(get_value(cfg, "generation.max_tokens", 2048))
    temperature = float(get_value(cfg, "generation.temperature", 0.0))

    do_image_desc = bool(get_value(cfg, "image_desc.enabled", False))
    img_desc_prompt = get_value(cfg, "prompt_img_desc", "")
    img_desc_max_tokens = int(get_value(cfg, "image_desc.max_tokens", 256))
    img_desc_temperature = float(get_value(cfg, "image_desc.temperature", 0.0))

    pad_ratio = float(get_value(cfg, "image_crop.pad_ratio", 0.01))
    debug_draw_bboxes = bool(get_value(cfg, "debug.draw_bboxes", False))

    render_scale = float(get_value(cfg, "render.scale", 1.0))
    use_md_cache = bool(get_value(cfg, "cache.use_md_cache", False))

    filter_enabled = bool(get_value(cfg, "image_filter.enabled", False))
    stddev_min = float(get_value(cfg, "image_filter.stddev_min", 0.0))
    stddev_mode = str(get_value(cfg, "image_filter.mode", "grayscale"))

    desc_cache = load_desc_cache(image_desc_path)

    md_cache_path = output_dir / f"{doc_title}.{doc_sha[:12]}.md"
    if use_md_cache and md_cache_path.exists():
        _log.info("MD cache found: %s", md_cache_path)
        md_text = md_cache_path.read_text(encoding="utf-8")
        page_count = 0
        image_count = count_jsonl_lines(image_map_path)
        desc_count = count_jsonl_lines(image_desc_path)
        return ParseArtifactsResult(
            doc_id=doc_id,
            doc_sha256=doc_sha,
            doc_title=doc_title,
            source_uri=source_uri,
            output_dir=str(output_dir),
            assets_root=str(assets_root),
            pages_dir=str(pages_dir),
            md_path=str(md_cache_path),
            image_map_path=str(image_map_path),
            image_desc_path=str(image_desc_path),
            page_count=page_count,
            image_count=image_count,
            desc_count=desc_count,
        )
    
    _log.info("MD cache not found or disable. Start OCR + Image extraction.")
    page_pngs = pdf_to_page_pngs(input_doc_path, scale=render_scale)

    page_texts: List[str] = []
    total_images = 0
    total_desc = 0
    page_count = 0

    for idx, item in enumerate(tqdm(page_pngs, total=len(page_pngs), desc="OCR + IMG", unit="page"), start=1):
        page_no, payload = coerce_page_no_and_payload(item, fallback_page_no=idx)
        png_path = materialize_png_payload(payload, out_dir=pages_dir, page_no=page_no)
        page_count += 1

        txt = ocr_page(
            png_path.read_bytes(),
            vlm_url, vlm_model, vlm_api_key,
            prompt, max_tokens, temperature, timeout_sec
        )

        new_txt, img_records, desc_records = extract_and_store_images_from_page(
            page_png_path=png_path,
            ocr_text=txt,
            assets_root=assets_root,
            doc_id=doc_id,
            doc_title=doc_title,
            source_uri=source_uri,
            sha256=doc_sha,
            page_no=page_no,
            pad_ratio=pad_ratio,
            debug_draw_bboxes=debug_draw_bboxes,
            do_image_desc=do_image_desc,
            img_desc_prompt=img_desc_prompt,
            img_desc_max_tokens=img_desc_max_tokens,
            img_desc_temperature=img_desc_temperature,
            vlm_url=vlm_url,
            vlm_model=vlm_model,
            vlm_api_key=vlm_api_key,
            vlm_timeout_sec=timeout_sec,
            desc_cache=desc_cache,
            filter_enabled=filter_enabled,
            stddev_min=stddev_min,
            stddev_mode=stddev_mode,
        )

        append_jsonl(img_records, image_map_path)
        append_jsonl(desc_records, image_desc_path)

        total_images += len(img_records)
        total_desc += len(desc_records)
        page_texts.append(new_txt)

    md_text = build_md_from_pages(page_texts)
    md_cache_path.write_text(md_text, encoding="utf-8")
    _log.info("Wrote MD cache: %s (pages=%d images=%d desc=%d)", md_cache_path, page_count, total_images, total_desc)

    return ParseArtifactsResult(
        doc_id=doc_id,
        doc_sha256=doc_sha,
        doc_title=doc_title,
        source_uri=source_uri,
        output_dir=str(output_dir),
        assets_root=str(assets_root),
        pages_dir=str(pages_dir),
        md_path=str(md_cache_path),
        image_map_path=str(image_map_path),
        image_desc_path=str(image_desc_path),
        page_count=page_count,
        image_count=total_images,
        desc_count=total_desc,
    )