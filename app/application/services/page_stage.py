from typing import Any, Dict, List, Tuple

from app.adapters.indexing.opensearch_docs import load_pages_from_staging as os_load_pages_from_staging
from app.infra.storage.opensearch import OpenSearchWriter


def load_pages_from_staging(
    *,
    os_pages_staging: OpenSearchWriter,
    doc_id: str,
) -> Tuple[List[str], int, int, int, Dict[str, Any]]:
    return os_load_pages_from_staging(os_pages_staging=os_pages_staging, doc_id=doc_id)
