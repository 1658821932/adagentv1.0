"""Stable AdAgent entry points; native retrieval remains framework-independent."""
from mcp_servers.runtime import runtime_task, current_run

def retrieve_video(input_data: dict) -> dict:
    from .client import retrieve_video as call
    return call(input_data)

@runtime_task("sentry_video_search", "search_local")
def search_local(requirement, config=None):
    from ..core.config import SearchConfig
    from ..core.service import search_local as search
    from ..core.workspace import bind_run
    with bind_run(current_run("sentry_video_search")):
        return search(requirement, config or SearchConfig())

@runtime_task("sentry_video_search", "index_local")
def index_local(path, config=None):
    from ..core.config import SearchConfig
    from ..core.service import index_local as index
    from ..core.workspace import bind_run
    with bind_run(current_run("sentry_video_search")):
        return index(path, config or SearchConfig())

@runtime_task("sentry_video_search", "search_internet")
def search_internet(requirement, config=None):
    from ..core.config import SearchConfig
    from ..core.service import search_internet as search
    from ..core.workspace import bind_run
    with bind_run(current_run("sentry_video_search")):
        return search(requirement, config or SearchConfig())
