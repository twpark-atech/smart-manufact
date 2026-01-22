# ==============================================================================
# 목적 : 정규식 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import re

RE_IMG_BOX = re.compile(r"image\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]")
RE_CAP_BOX = re.compile(r"image_caption\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]")
RE_REF_DET = re.compile(
    r"<\|ref\|>\s*(?P<kind>[^<]+?)\s*<\|/ref\|>\s*"
    r"<\|det\|>\s*\[\[\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)"
    r"\s*\]\]\s*<\|/det\|>",
    re.MULTILINE,
)
RE_IMG_TOKEN = re.compile(r"\[\[IMG:([^\]]+)\]\]")
RE_PAGE_HEADER = re.compile(r"(?m)^\s*##\s*Page\s+(\d+)\s*$")
RE_HTML_TABLE = re.compile(r"(<table\b[^>]*>.*?</table>)", re.IGNORECASE | re.DOTALL)
RE_SENT_END = re.compile(r"[.!?](?=\s|$)")

_HNSW_4096 = {
    "type": "knn_vector",
    "dimension": 4096,
    "method": {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil",
        "parameters": {"m": 48, "ef_construction": 256}
    },
}

_HNSW_1024 = {
    "type": "knn_vector",
    "dimension": 1024,
    "method": {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil",
        "parameters": {"m": 48, "ef_construction": 256}
    },
}