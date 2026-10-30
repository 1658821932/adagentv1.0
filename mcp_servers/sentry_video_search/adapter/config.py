"""AdAgent video retrieval environment and native search configuration."""
from pathlib import Path
from ..core.config import SearchConfig, LIBRARIES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEO_RETRIEVER_PYTHON = PROJECT_ROOT / "environments" / "video_retriever" / ".venv" / "bin" / "python"
