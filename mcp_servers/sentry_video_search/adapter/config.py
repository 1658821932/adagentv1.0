"""AdAgent video retrieval environment and native search configuration."""
from pathlib import Path
from mcp_servers.configuration import runtime_path
from ..core.config import SearchConfig, LIBRARIES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_RETRIEVER_PYTHON = runtime_path(Path(__file__).resolve().parents[1], "python", "VIDEO_RETRIEVER_PYTHON", "environments/video_retriever/.venv/bin/python")
