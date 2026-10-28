"""Video retrieval facade; load service dependencies only when requested."""

from importlib import import_module
from .config import SearchConfig

__all__ = ["SearchConfig", "VideoSearchResult", "index_local", "search_local", "search_internet", "retrieve_video"]


def __getattr__(name):
    if name in __all__ and name != "SearchConfig":
        return getattr(import_module(".service", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
