# ==============================================================================
# 목적 : PDF에서 OCR 후 데이터를 DB에 적재를 실행하는 코드
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-21
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import json, logging
from pathlib import Path

from app.application.usecases.ingest_pdf import ingest_pdf

_log = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    logging.getLogger("opensearch").setLevel(logging.WARNING)
    logging.getLogger("opensearchpy").setLevel(logging.WARNING)
    logging.getLogger("opensearchpy.transport").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("app.infra.storage.opensearch").setLevel(logging.WARNING)


def main() -> int:
    setup_logging()

    config_path = Path("config/config.yml")

    try:
        result = ingest_pdf(config_path=config_path)
    except Exception:
        _log.exception("Workflow failed")
        return 1
    
    payload = result.to_dict()
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    _log.info("Done. md_path=%s doc_id=%s chunks=%s images_indexed=%s", 
              payload.get("md_path"), payload.get("doc_id"),
              payload.get("indexed_chunk_count"), payload.get("indexed_image_count"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
