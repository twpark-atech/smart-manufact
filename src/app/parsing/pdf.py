# ==============================================================================
# 목적 : PDF 추출 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path
from typing import Union, Tuple, Any


def coerce_page_no_and_payload(
    item: Union[str, Path, bytes, bytearray, Tuple[int, Any]],
    fallback_page_no: int,
) -> Tuple[int, Any]:
    if isinstance(item, tuple) and len(item) == 2:
        return int(item[0]), item[1]
    return fallback_page_no, item


def materialize_png_payload(payload: Any, *, out_dir: Path, page_no: int) -> Path:
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