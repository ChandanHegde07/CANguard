
from __future__ import annotations

import io
import time
from contextlib import contextmanager
from typing import Any, Iterator

import joblib

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def peak_rss_mb() -> float:
    """Current process RSS in megabytes (0.0 if psutil unavailable)."""
    if psutil is None:
        return 0.0
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """Context manager that records wall-clock seconds and RSS delta."""
    out: dict[str, float] = {}
    rss0 = peak_rss_mb()
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = float(time.perf_counter() - t0)
        out["rss_mb_end"] = peak_rss_mb()
        out["rss_mb_delta"] = out["rss_mb_end"] - rss0


def measure_model_size_bytes(obj: Any) -> int:
    """Serialize ``obj`` with joblib and return byte length."""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return int(buf.getbuffer().nbytes)
