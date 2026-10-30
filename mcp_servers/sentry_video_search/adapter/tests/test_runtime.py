"""Task isolation and adapter compatibility without loading GPU models."""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_servers.runtime import current_run, runtime_task




def test_retrieval_exports_into_task_and_preserves_index(tmp_path):
    from mcp_servers.sentry_video_search.core.config import SearchConfig
    from mcp_servers.sentry_video_search.adapter.service import search_local
    config = SearchConfig(runtime_home=tmp_path / "shared-index")
    hit = {"candidate_index": 0, "source_file": "/video.mp4", "similarity_score": 0.9}
    def export(details, requirement, cfg, logs):
        assert cfg.runtime_home == config.runtime_home
        path = cfg.output_dir / "clip.mp4"; path.write_bytes(b"clip")
        return path
    with patch("mcp_servers.sentry_video_search.core.service._ensure_dashscope_key"), patch("mcp_servers.sentry_video_search.core.service._search_index", return_value=[hit]), patch("mcp_servers.sentry_video_search.core.reranker.rerank_hits", return_value=(hit, {})), patch("mcp_servers.sentry_video_search.core.service._export_reranked_hits", side_effect=export):
        result = search_local("front car", config)
    assert Path(result.video_path).is_relative_to(Path(result.run_dir) / "output")
    assert (Path(result.run_dir) / "logs/local_search.json").is_file()
    assert config.runtime_home == tmp_path / "shared-index"
