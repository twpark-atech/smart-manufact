# ==============================================================================
# 목적 : OCR 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import base64

import requests


def ocr_page(
    png_bytes: bytes, 
    url: str, 
    model: str, 
    api_key: str, 
    prompt: str, 
    max_tokens: int, 
    temperature: float, 
    timeout: int = 3600
) -> str:
    r"""페이지 이미지(PNG bytes)를 LVM(OpenAI 호환 API)으로 OCR하여 텍스트를 반환합니다.
    
    Args:
        png_bytes: PNG bytes (한 페이지) 
        url: VLM API endpoint
        model: 사용할 모델명
        api_key: Authorization Bearer 토큰
        prompt: OCR용 프롬프트
        max_tokens: 응답 최대 토큰
        temperature: 샘플링 온도
        timeout: HTTP timeout (초)

    Returns:
        OCR 결과 문자열

    Raises:
        requests.exceptions.RequestException: 네트워크/HTTP 요청 실패 
        requests.HTTPError: 4xx/5xx 응답

    Examples:
        >>> text = ocr_page(png_bytes, "http://localhost:11434/v1/chat/completions", "gpt-oss:20b", "", "OCR 엔진으로서 이미지의 모든 텍스트를 추출해줘", 2048, 0.1, 3600)
        sub_title[[435, 111, 559, 135]]
        소재 개론
    """
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""