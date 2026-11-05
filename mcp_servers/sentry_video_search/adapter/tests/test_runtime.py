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
    with patch("mcp_servers.sentry_video_search.adapter.scene_binding.bind_scene", return_value={"scene_token": "test"}), patch("mcp_servers.sentry_video_search.core.service._ensure_dashscope_key"), patch("mcp_servers.sentry_video_search.core.service._search_index", return_value=[hit]), patch("mcp_servers.sentry_video_search.core.reranker.rerank_hits", return_value=(hit, {})), patch("mcp_servers.sentry_video_search.core.service._export_reranked_hits", side_effect=export):
        result = search_local("front car", config)
    assert Path(result.video_path).is_relative_to(Path(result.run_dir) / "output")
    assert (Path(result.run_dir) / "logs/local_search.json").is_file()
    assert config.runtime_home == tmp_path / "shared-index"


def test_long_chinese_retrieval_description_exports_valid_filename(tmp_path):
    from mcp_servers.sentry_video_search.core.config import SearchConfig
    from mcp_servers.sentry_video_search.core.service import _trim_hit
    requirement = "城市多车道直道，同车道前方有正常行驶车辆，邻道有少量社会车辆。" * 20
    def trim(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"video")
    with patch("mcp_servers.sentry_video_search.core.vendor.sentrysearch.chunker._get_video_duration", return_value=12), patch("mcp_servers.sentry_video_search.core.vendor.sentrysearch.trimmer.trim_clip", side_effect=trim):
        output = _trim_hit({"source_file":"/source.mp4", "start_time":0}, requirement, SearchConfig(output_dir=tmp_path), label="rank01_score0.900000_compressed")
    assert output.is_file()
    assert len(output.name.encode("utf-8")) <= 255
