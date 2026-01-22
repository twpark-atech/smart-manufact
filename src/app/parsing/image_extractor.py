# ==============================================================================
# 목적 : 이미지 추출 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import io, json, hashlib, logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw

from app.common.hash import sha256_bytes
from app.parsing.ocr import ocr_page
from app.parsing.regex import RE_CAP_BOX, RE_REF_DET, RE_SENT_END

_log = logging.getLogger(__name__)


def _make_image_id(doc_id: str, page_no: int, bbox: Tuple[int, int, int, int]) -> str:
    """문서/페이지/바운딩박스 정보를 기반으로 결정적 이미지 ID를 생성합니다.

    doc_id, page_no, bbox(x1,y1,x2,y2)를 문자열로 결합한 뒤 SHA-1 해시(16 hex)로 축약하여 충돌 가능성을 낮춘 ID를 만듭니다.

    Args:
        doc_id: 문서 식별자.
        page_no: 페이지 번호.
        bbox: (x1, y1, x2, y2) 픽셀 좌표 바운딩박스.

    Returns:
        결정적 이미지 ID 문자열.
    """
    key = f"{doc_id}|p{page_no}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:p{page_no}:img{h}"


def _infer_bbox_scale(boxes: List[Tuple[float, float, float, float]], W: int, H: int) -> Tuple[float, float]:
    """탐지된 bbox 좌표계가 페이지 픽셀 좌표로 변환되도록 스케일(sx, sy)을 추정합니다.
    
    OCR/VLM 출력의 bbox는 다음과 같은 형태로 섞여 들어올 수 있어, box들의 최대값을 기준으로 좌표계를 추정합니다.

    Args:
        boxes: (x1, y1, x2, y2) 형태의 bbox 리스트.
        W: 페이지 이미지 폭(px).
        H: 페이지 이미지 높이(py).

    Returns:
        (sx, sy) 스케일 팩터.
    """
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
    x1: float, y1: float, x2: float, y2: float,
    *, sx: float, sy: float, W: int, H: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    """bbox 좌표를 스케일링/패딩/클램핑하여 픽셀 정수 좌표로 변환합니다.
    
    입력 좌표에 (sx, sy) 스케일을 적용하고 반올림하여 정수 픽셀 좌표로 변환합니다.
    bbox 폭/높이(bw, bh)에 pad_ratio를 곱한 패딩(px, py)을 사방으로 확장합니다.
    최종 좌표를 이미지 경계 [0..W], [0..H]로 클램핑합니다.

    Args:
        x1, y1, x2, y2: 원본 bbox 좌표(실수).
        sx, sy: 좌표계 변환 스케일.
        W, H: 대상 이미지 크기(px).
        pad_ratio: bbox 크기 대비 패딩 비율.

    Returns:
        (fx1, fy1, fx2, fy2) 형태의 변환된 bbox 정수 좌표.
    """
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


def _cut_by_sentence_end(s: str, max_chars: int) -> str:
    """설명 텍스트의 길이를 제한합니다.
    
    describe 결과를 총 "max_chars"자 이내로 제한합니다.
    "max_chars"자가 넘어갈 경우 "max_chars"자 이내 마지막 마침표를 기준으로 잘라냅니다.
    마침표가 없을 경우 "max_chars"자에서 잘라냅니다.
    
    Args:
        s: 잘라낼 문자열.
        max_chars: 최대 길이

    Returns:
        최대 "max_chars"자로 제한된 설명.
    """
    if len(s) <= max_chars:
        return s
    
    prefix = s[:max_chars]

    last_end = -1
    for m in RE_SENT_END.finditer(prefix):
        last_end = m.end()

    if last_end > 0:
        return prefix[:last_end].rstrip()
    
    return prefix.rstrip()


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
    """이미지에 대한 짧은 설명 텍스트를 생성하고 길이를 제한합니다.
    
    내부적으로 ocr_page(...)를 호출해 이미지 설명/텍스트를 생성합니다.
    결과 텍스트는 공백을 정규화하고, 240자를 초과하면 말줄임표(…)로 절단합니다.
    생성된 텍스트가 비어있으면 빈 문자열을 반환합니다.

    Args:
        image_bytes: 입력 이미지 바이트.
        vlm_url: VLM/OCR API 엔드포인트.
        vlm_model: 사용할 모델 식별자.
        vlm_api_key: API 키.
        timeout_sec: 요청 타임아웃(초).
        prompt: 설명 생성을 위한 프롬프트.
        max_tokens: 생성 최대 토큰 수.
        temperature: 생성 온도.

    Returns:
        최대 240자로 제한된 단문 설명. 생성 실패/빈 응답이면 "".
    """
    try:
        txt = ocr_page(
            image_bytes,
            vlm_url, vlm_model, vlm_api_key,
            prompt,
            max_tokens, temperature, timeout_sec,
        )

        if not txt:
            return ""

        s = " ".join(txt.split()).strip()
        if not s:
            return ""

        s = _cut_by_sentence_end(s, max_chars=1200)
        
        return s

    except Exception as e:
        _log.exception("IMG_DESC failed: %s", e)

    return ""


def load_desc_cache(desc_jsonl: Path) -> Dict[str, Dict[str, str]]:
    """이미지 설명 캐시(JSONL)를 로드하여 image_sha256 -> {description, image_id} 맵을 구성합니다.
    
    desc_jsonl 파일을 한 줄씩 읽어 JSON으로 파싱합니다.
    {"image_sha256": ..., "description": ..., "image_id": ...} 형태 레코드를 캐시에 적재합니다.
    파일이 없으면 빈 dict를 반환합니다.
    파싱 중 오류가 발생하면 warning 로그를 남기고 가능한 범위까지 로드한 결과를 반환합니다.

    Args:
        desc_jsonl: 설명 캐시 jsonl 파일 경로.

    Returns:
        image_sha256를 키로 하는 캐시 dict.
    """
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
                    cache[img_sha] = {"description": str(desc), "image_id": r.get("image_id", "")}
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
    skip_fullpage: bool = True,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """페이지 OCR 텍스트에서 이미지 bbox를 추출해 crop 저장하고, 필요 시 설명을 생성합니다.

    1) ocr_text에서 RE_REF_DET 정규식으로 kind="image" 탐지 결과를 찾아 bbox 후보를 수집합니다.
    2) _infer_bbox_scale로 (sx, sy)를 추정한 뒤 _transform_bboxx로 픽셀 bbox를 얻습니다.
    3) bbox로 페이지 이미지를 crop하여 assets_root/images 아래에 PNG로 저장하고, 동일 메타데이터를 json으로도 저장합니다.
    4) do_image_desc=True이면 이미지 sha256 기반 캐시를 조회해 설명을 재사용하고, 없으면 describe_image_short로 생성합니다.
    5) 원본 ocr_text 내 해당 탐지 토큰을 [[IMG:{image_id}]]로 1회 치환하여 new_text로 반환합니다.
    6) debug_draw_bboxes=True이면 bbox가 그려진 디버그 이미지를 assets_root/pages에 저장합니다.

    Args:
        page_png_path: 페이지 PNG 파일 경로.
        ocr_text: 페이지 OCR 결과 텍스트.
        assets_root: 산출물 저장 루트 디렉토리.
        doc_id/doc_title/source_uri/sha256: 문서 메타데이터.
        page_no: 페이지 번호.
        pad_ratio: bbox 확장 패딩 비율.
        debug_draw_bboxes: bbox 디버그 이미지 저장 여부.
        do_image_desc: 이미지 설명 생성/캐시 사용 여부.
        img_desc_prompt/img_desc_max_tokens/img_desc_temperature: 설명 생성 파라미터.
        vlm_url/vlm_model/vlm_api_key/vlm_timeout_sec: VLM/OCR 호출 파라미터.
        desc_cache: image_sha256 -> {"description", "image_id"} 캐시.
        filter_enabled/stddev_min/stddev_mode: crop 품질 필터 설정.
        skip_fullpage: 전체 페이지 bbox 스킵 여부.

    Returns:
        (new_text, img_records, desc_records)

    Raises:
        ValueError: stddev_mode가 잘못된 경우.
        OSError: 이미지 파일 저장/읽기 등 파일 I/O 과정에서 잘못된 경우.
        PIL.UnidentifiedImageError: page_png_path가 이미지로 열리지 않을 경우.  
    """
    if not ocr_text:
        return ocr_text, [], []
    
    img_dir = assets_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    page_img = Image.open(page_png_path).convert("RGB")
    W, H = page_img.size

    lines = ocr_text.splitlines()
    cap_candidates: List[Tuple[Tuple[float, float, float, float], str]] = []
    for i, ln in enumerate(lines):
        mcap = RE_CAP_BOX.search(ln)
        if not mcap:
            continue
        try:
            cx1, cy1, cx2, cy2 = map(float, mcap.groups())
        except Exception:
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j  += 1
        cap_text = lines[j].strip() if j < len(lines) else ""
        cap_candidates.append(((cx1, cy1, cx2, cy2), cap_text))

    det_matches = list(RE_REF_DET.finditer(ocr_text))
    img_matches = [m for m in det_matches if (m.group("kind") or "").strip() == "image"]

    if not img_matches:
        return ocr_text, [], []
    
    raw_boxes: List[Tuple[float, float, float, float]] = []
    for m in img_matches:
        try:
            raw_boxes.append(
                (
                    float(m.group("x1")),
                    float(m.group("y1")),
                    float(m.group("x2")),
                    float(m.group("y2")),
                )
            )
        except Exception:
            continue

    if not raw_boxes:
        _log.info("Image bbox matched but parse failed. page_no=%s", page_no)
        return ocr_text, [], []

    sx, sy = _infer_bbox_scale(raw_boxes, W, H)

    debug_img = page_img.copy()
    debug_draw = ImageDraw.Draw(debug_img) if debug_draw_bboxes else None

    img_records: List[Dict[str, Any]] = []
    desc_records: List[Dict[str, Any]] = []
    new_text = ocr_text

    for m in img_matches:
        try:
            x1 = float(m.group("x1"))
            y1 = float(m.group("y1"))
            x2 = float(m.group("x2"))
            y2 = float(m.group("y2"))
        except Exception:
            continue

        if skip_fullpage:
            if abs(x1) < 1e-6 and abs(y1) < 1e-6 and x2 >= 0.99 and y2 >= 0.99:
                continue
            if abs(x1) < 1e-6 and abs(y1) < 1e-6 and x2 >= 999 and y2 >= 999:
                continue

        x1i, y1i, x2i, y2i = _transform_bbox(x1, y1, x2, y2, sx=sx, sy=sy, W=W, H=H, pad_ratio=pad_ratio)
        if x2i <= x1i or y2i <= y1i:
            continue

        bbox = (x1i, y1i, x2i, y2i)
        image_id = _make_image_id(doc_id, page_no, bbox)

        safe_name = image_id.replace(":", "_")
        img_path = img_dir / f"{safe_name}.png"
        meta_path = img_dir / f"{safe_name}.json"

        if not img_path.exists():
            crop = page_img.crop((x1i, y1i, x2i, y2i))

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

        with Image.open(img_path) as _ci:
            cw, ch = _ci.size

        cap_text_best = ""
        if cap_candidates:
            ix, iy = (x1i + x2i) / 2, (y1i + y2i) / 2
            best_d = None
            for (cx1, cy1, cx2, cy2), cap_text in cap_candidates:
                cx1i, cy1i, cx2i, cy2i = _transform_bbox(cx1, cy1, cx2, cy2, sx=sx, sy=sy, W=W, H=H, pad_ratio=0.0)
                cx, cy = (cx1i + cx2i) / 2, (cy1i + cy2i) / 2
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
            "bbox": {"x1": x1i, "y1": y1i, "x2": x2i, "y2": y2i},
            "bbox_transform": {"sx": sx, "sy": sy, "pad_ratio": pad_ratio},
            "image_path": str(img_path),
            "image_mime": "image/png",
            "width": int(cw),
            "height": int(ch),
            "caption": cap_text_best,
        }
        meta_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        img_records.append(record)

        if do_image_desc:
            b = img_path.read_bytes()
            image_sha = sha256_bytes(b)
            desc_obj = desc_cache.get(image_sha)
            desc_text = (desc_obj or {}).get("description", "") if isinstance(desc_obj, dict) else ""
            
            if not desc_text:
                desc_text = describe_image_short(
                    b,
                    vlm_url=vlm_url,
                    vlm_model=vlm_model,
                    vlm_api_key=vlm_api_key,
                    timeout_sec=vlm_timeout_sec,
                    prompt=img_desc_prompt,
                    max_tokens=img_desc_max_tokens,
                    temperature=img_desc_temperature,
                )
                if desc_text:
                    desc_cache[image_sha] = {"description": desc_text, "image_id": image_id}
            
            if desc_text:
                desc_records.append({
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "doc_id": doc_id,
                    "page_no": page_no,
                    "image_path": str(img_path),
                    "image_mime": "image/png",
                    "width": int(cw),
                    "height": int(ch),
                    "caption": cap_text_best,
                    "description": desc_text,
                })

        new_text = new_text.replace(m.group(0), f"[[IMG:{image_id}]]", 1)

        if debug_draw is not None:
            debug_draw.rectangle([x1i, y1i, x2i, y2i], width=3)

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
    """crop 이미지의 표준편차를 기준으로 스킵 여부를 판단합니다.
    
    배경/여백처럼 정보량이 낮은 crop을 걸러내기 위한 간단한 휴리스틱입니다.
    enabled가 False이거나 stddev_min <= 0이면 항상 스킵하지 않습니다.
    mode에 따라 grayscale 또는 RGB로 변환한 뒤 픽셀 값 표준편차를 계산합니다.

    Args:
        crop: PIL 이미지 객체.
        enabled: 필터 활성화 여부.
        stddev_min: 표준편차 임계값.
        mode: "grayscale" 또는 "rgb".

    Returns:
        (should_skip, stddev)

    Raises:
        ValueError: mode가 "grayscale" 또는 "rgb"가 아닐 경우.
    """
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


def get_image_size_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    """이미지 바이트에서 (width, height)를 안전하게 추출합니다.
    
    image_bytes를 PIL로 열어 이미지 크기를 반환합니다.
    파싱에 실패하면 예외를 전파하지 않고 (0, 0)을 반환합니다.
    
    Args:
        image_bytes: PNG/JPEG 등 이미지 바이트.
        
    Returns:
        (width, height).        
    """
    try:
        im = Image.open(io.BytesIO(image_bytes))
        return int(im.size[0]), int(im.size[1])
    except Exception:
        return 0, 0