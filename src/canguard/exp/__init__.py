"""Phase A experiment infrastructure: seeds, cache, logging, tracking."""

from .cache import FeatureCache
from .config import load_config
from .logging_utils import setup_logging
from .resources import measure_model_size_bytes, peak_rss_mb, timed
from .seeds import set_global_seed
from .tracking import ExperimentRun, save_csv, save_json

__all__ = [
    "ExperimentRun",
    "FeatureCache",
    "load_config",
    "measure_model_size_bytes",
    "peak_rss_mb",
    "save_csv",
    "save_json",
    "set_global_seed",
    "setup_logging",
    "timed",
]

# matrix helpers imported lazily by runners to avoid circular imports
