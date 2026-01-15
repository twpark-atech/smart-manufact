# ==============================================================================
# 목적 : Batch 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from typing import Sequence, Any, Iterable, Tuple


def iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Tuple[int, int, Sequence[Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    n = len(items)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end, items[start:end]