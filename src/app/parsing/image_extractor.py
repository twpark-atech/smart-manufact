# ==============================================================================
# 목적 : 이미지 추출 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import json, hashlib, logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw

from app.common.hash import sha256_bytes
from app.parsing.ocr import ocr_page
from app.parsing.regex import _RE_CAP_BOX, _RE_IMG_BOX

_log = logging.getLogger(__name__)


def _make_image_id(doc_id: str, page_no: int, bbox: Tuple[int, int, int, int]) -> str:
    key = f"{doc_id}|p{page_no}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:p{page_no}:img{h}"


def _infer_bbox_scale(boxes: List[Tuple[int, int, int, int]], W: int, H: int) -> Tuple[float, float]:
    if not boxes:
        return 1.0, 1.0
    
    max_x2 = max(b[2] for b in boxes)
    max_y2 = max(b[3] for b in boxes)

    if max_x2 <= 1.5 and max_y2 <= 1.5:
        return float(W), float(H)
    
    if max_x2 <= 1100 and max_y2 <= 1100 and (W > 1200 or H > 1200):
        return W / 1000.0, H / 1000.0
    
    sx = 1.0
    sy = 1.0
    if max_x2 > 0 and (max_x2 < W * 0.85 or max_x2 > W * 1.15):
        sx = W / float(max_x2)
    if max_y2 > 0 and (max_y2 < H * 0.85 or max_y2 > H * 1.15):
        sy = H / float(max_y2)

    return sx, sy


def _transform_bbox(
    x1: int, y1: int, x2: int, y2: int,
    *, sx: float, sy: float, W: int, H: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    fx1 = int(round(x1 * sx))
    fy1 = int(round(y1 * sy))
    fx2 = int(round(x2 * sx))
    fy2 = int(round(y2 * sy))

    bw = max(1, fx2 - fx1)
    bh = max(1, fy2 - fy1)
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))

    fx1 -= px
    fy1 -= py
    fx2 += px
    fy2 += py

    fx1 = max(0, min(fx1, W))
    fx2 = max(0, min(fx2, W))
    fy1 = max(0, min(fy1, H))
    fy2 = max(0, min(fy2, H))
    return fx1, fy1, fx2, fy2


def describe_image_short(
    image_bytes: bytes,
    *,
    vlm_url: str,
    vlm_model: str,
    vlm_api_key: str,
    timeout_sec: int,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    txt = ocr_page(
        image_bytes,
        vlm_url, vlm_model, vlm_api_key,
        prompt,
        max_tokens, temperature, timeout_sec,
    )
    if not txt:
        return ""
    s = " ".join(txt.split())
    if len(s) > 240:
        s = s[:240].rstrip() + "…"
    return s


def load_desc_cache(desc_jsonl: Path) -> Dict[str, Dict[str, str]]:
    cache: Dict[str, Dict[str, str]] = {}
    if not desc_jsonl.exists():
        return cache
    
    try:
        with desc_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                img_sha = r.get("image_sha256")
                desc = r.get("description")
                if img_sha and desc:
                    cache[img_sha] = {"description": desc, "image_id": r.get("image_id", "")}
    except Exception as e:
        _log.warning("Failed to load image desc cache: %s", e)
    
    return cache


def extract_and_store_images_from_page(
    page_png_path: Path,
    ocr_text: str,
    *,
    assets_root: Path,
    doc_id: str,
    doc_title: str,
    source_uri: str,
    sha256: str,
    page_no: int,
    pad_ratio: float,
    debug_draw_bboxes: bool,
    do_image_desc: bool,
    img_desc_prompt: str,
    img_desc_max_tokens: int,
    img_desc_temperature: float,
    vlm_url: str,
    vlm_model: str,
    vlm_api_key: str,
    vlm_timeout_sec: int,
    desc_cache: Dict[str, Dict[str, str]],
    filter_enabled: bool = False,
    stddev_min: float = 0.0,
    stddev_mode: str = "grayscale",
) -> Tuple[str, List[Dict[str, Any]]]:
    if not ocr_text:
        return ocr_text, []
    
    img_dir = assets_root / "images" / f"p{page_no:04d}"
    img_dir.mkdir(parents=True, exist_ok=True)

    page_img = Image.open(page_png_path).convert("RGB")
    W, H = page_img.size

    lines = ocr_text.splitlines()
    cap_candidates: List[Tuple[Tuple[int, int, int, int], str]] = []
    for i, ln in enumerate(lines):
        mcap = _RE_CAP_BOX.search(ln)
        if not mcap:
            continue
        cx1, cy1, cx2, cy2 = map(int, mcap.groups())
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j  += 1
        cap_text = lines[j].strip() if j < len(lines) else ""
        cap_candidates.append(((cx1, cy1, cx2, cy2), cap_text))

    matches = list(_RE_IMG_BOX.finditer(ocr_text))
    raw_boxes = [tuple(map(int, m.groups())) for m in matches]
    sx, sy = _infer_bbox_scale(raw_boxes, W, H)

    debug_img = page_img.copy()
    debug_draw = ImageDraw.Draw(debug_img) if debug_draw_bboxes else None

    img_records: List[Dict[str, Any]] = []
    desc_records: List[Dict[str, Any]] = []
    new_text = ocr_text

    for m in matches:
        x1, y1, x2, y2 = map(int, m.groups())
        x1, y1, x2, y2 = _transform_bbox(x1, y1, x2, y2, sx=sx, sy=sy, W=W, H=H, pad_ratio=pad_ratio)
        if x2 <= x1 or y2 <= y1:
            continue

        bbox = (x1, y1, x2, y2)
        image_id = _make_image_id(doc_id, page_no, bbox)

        safe_name = image_id.replace(":", "_")
        img_path = img_dir / f"{safe_name}.png"
        meta_path = img_dir / f"{safe_name}.json"

        if not img_path.exists():
            crop = page_img.crop((x1, y1, x2, y2))

            should_skip, stddev = should_skip_crop_by_sttdev(
                crop,
                enabled=filter_enabled,
                stddev_min=stddev_min,
                mode=stddev_mode,
            )
            if should_skip:
                _log.debug(
                    "Skip crop by stddev: page=%d bbox=%s stddev=%.3f (min=%.3f mode=%s)",
                    page_no, bbox, stddev, stddev_min, stddev_mode
                )
                continue

            crop.save(img_path, format="PNG")

        cap_text_best = ""
        if cap_candidates:
            ix, iy = (x1 + x2) / 2, (y1 + y2) / 2
            best_d = None
            for (cx1, cy1, cx2, cy2), cap_text in cap_candidates:
                cx, cy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
                d = (ix - cx) ** 2 + (iy - cy) ** 2
                if best_d is None or d < best_d:
                    best_d = d
                    cap_text_best = cap_text

        record = {
            "image_id": image_id,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "source_uri": source_uri,
            "sha256": sha256,
            "page_no": page_no,
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "bbox_transform": {"sx": sx, "sy": sy, "pad_ratio": pad_ratio},
            "image_path": str(img_path),
            "caption": cap_text_best,
        }
        meta_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        img_records.append(record)

        if do_image_desc:
            b = img_path.read_bytes()
            image_sha = sha256_bytes(b)
            desc = desc_cache.get(image_sha)
            if desc is None:
                desc = describe_image_short(
                    b,
                    vlm_url=vlm_url,
                    vlm_model=vlm_model,
                    vlm_api_key=vlm_api_key,
                    timeout_sec=vlm_timeout_sec,
                    prompt=img_desc_prompt,
                    max_tokens=img_desc_max_tokens,
                    temperature=img_desc_temperature,
                )
                if desc:
                    desc_cache[image_sha] = desc

            if desc:
                desc_records.append({
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "doc_id": doc_id,
                    "page_no": page_no,
                    "image_path": str(img_path),
                    "caption": cap_text_best,
                    "description": desc,
                })
        
        new_text = new_text.replace(m.group(0), f"[[IMG:{image_id}]]", 1)

        if debug_draw is not None:
            debug_draw.rectangle([x1, y1, x2, y2], width=3)

    if debug_draw_bboxes:
        debug_path = assets_root / "pages" / f"p{page_no:04d}.bboxes.png"
        debug_img.save(debug_path, format="PNG")

    return new_text, img_records, desc_records


def should_skip_crop_by_sttdev(
    crop: Image.Image,
    *,
    enabled: bool,
    stddev_min: float,
    mode: str = "grayscale",
) -> tuple[bool, float]:
    if not enabled or stddev_min <= 0:
        return False, 0.0
        
    if mode not in ("grayscale", "rgb"):
        raise ValueError(f"mode must be 'grayscale' or 'rgb'. got={mode}")
    
    if mode == "grayscale":
        arr = np.asarray(crop.convert("L"), dtype=np.float32)
        stddev = float(arr.std())
        return stddev < stddev_min, stddev
    
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    stddev = float(arr.std())
    return stddev < stddev_min, stddev