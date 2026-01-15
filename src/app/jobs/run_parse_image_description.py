# ==============================================================================
# 목적 : PDF에서 OCR 후 MD/Artifacts를 실행하는 코드
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import json, logging
from pathlib import Path

from app.workflows.parse_image_description import parse_image_description

_log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    config_path = Path("config/config.yaml")

    result = parse_image_description(config_path=config_path)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    _log.info("Done. md_path=%s", result.md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())