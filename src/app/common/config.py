# ==============================================================================
# 목적 : 공통 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    """YAML 설정 파일을 로드합니다.
    
    Args:
        config_path: config.yaml 경로

    Returns:
        YAML을 dict로 파싱한 결과. 비어있으면 {} 반환.

    Raises:
        FileNotFoundError: config 파일이 존재하지 않을 때 
        yaml.YAMLError: YAML 파싱에 실패했을 때 

    Examples:
        >>> cfg = load_config(Path("config.yaml"))
        >>> isinstance(cfg, dict)
        True
    """
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}