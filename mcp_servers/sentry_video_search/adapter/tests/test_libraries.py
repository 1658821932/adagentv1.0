"""Routing and indexing behavior for the separate video libraries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp_servers.sentry_video_search.core.workspace import runtime_task
from mcp_servers.sentry_video_search.core.config import SearchConfig
from mcp_servers.sentry_video_search.core.service import _export_reranked_hits, _trim_hit, index_local
from tools.tool04_video_retriever.real import run as retrieve_video


class VideoLibraryTests(unittest.TestCase):
    def test_exports_all_reranked_candidates_in_score_order(self):
        details = {
            "candidates": [{"candidate_index": i, "source_file": f"/{i}.mp4"} for i in range(4)],
            "ranked_results": [
                {"candidate_index": i, "relevance_score": score}
                for i, score in enumerate((0.7, 0.9, 0.5, 0.6))
            ],
        }
        with patch("mcp_servers.sentry_video_search.core.service._trim_hit") as trim_hit:
            trim_hit.side_effect = [Path(f"/output/{i}.mp4") for i in range(5)]
            winner = runtime_task("sentry_video_search", "export")(_export_reranked_hits)(details, "cut-in", SearchConfig(), Path("/logs/run_test"))

        self.assertEqual(winner, Path("/output/1.mp4"))
        self.assertEqual([item["candidate_index"] for item in details["ranked_results"]], [1, 0, 3, 2])
        self.assertEqual([item["video_path"] for item in details["ranked_results"]], [f"/output/{i}.mp4" for i in (0, 2, 3, 4)])
        self.assertEqual(details["ranked_results"][0]["lossless_video_path"], "/output/1.mp4")
        self.assertEqual(trim_hit.call_count, 5)
        self.assertEqual(trim_hit.call_args_list[0].args[0]["candidate_index"], 1)
        self.assertEqual(trim_hit.call_args_list[0].args[2].output_dir.name, "output")
        self.assertTrue(trim_hit.call_args_list[0].args[2].output_dir.parent.name.startswith("run_"))
        self.assertEqual(trim_hit.call_args_list[0].kwargs["label"], "rank01_score0.900000_compressed")
        self.assertEqual(trim_hit.call_args_list[0].kwargs["crf"], 28)
        self.assertEqual(trim_hit.call_args_list[1].kwargs["label"], "rank01_score0.900000_lossless")

    def test_final_clip_is_ten_seconds_without_padding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SearchConfig.for_library("waymo")
            config.output_dir = Path(temp_dir)
            hit = {"source_file": "/video.mp4", "start_time": 10.0, "end_time": 19.8}
            with patch("mcp_servers.sentry_video_search.core.vendor.sentrysearch.chunker._get_video_duration", return_value=19.8), patch(
                "mcp_servers.sentry_video_search.core.vendor.sentrysearch.trimmer.trim_clip"
            ) as trim_clip:
                _trim_hit(hit, "cut-in", config)

            self.assertEqual(trim_clip.call_args.kwargs["start_time"], 9.8)
            self.assertEqual(trim_clip.call_args.kwargs["end_time"], 19.8)
            self.assertEqual(trim_clip.call_args.kwargs["padding"], 0.0)
            self.assertTrue(trim_clip.call_args.kwargs["accurate"])
            self.assertEqual(trim_clip.call_args.kwargs["crf"], 0)

    def test_waymo_uses_independent_paths_and_ten_second_chunks(self):
        default = SearchConfig.for_library("default")
        waymo = SearchConfig.for_library("waymo")

        self.assertNotEqual(default.runtime_home, waymo.runtime_home)
        self.assertEqual(default.library_dir, waymo.library_dir)
        self.assertEqual(default.logs_dir, waymo.logs_dir)
        self.assertEqual(default.output_dir, waymo.output_dir)
        self.assertEqual(waymo.chunk_duration, 10)
        self.assertTrue(waymo.preprocess_index_chunks)
        self.assertFalse(waymo.import_source_videos)
        self.assertFalse(default.preprocess_index_chunks)
        with self.assertRaisesRegex(ValueError, "Unknown video library"):
            SearchConfig.for_library("unknown")

    def test_waymo_index_references_existing_video_without_copying(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "videos" / "front.mp4"
            video.parent.mkdir()
            video.write_bytes(b"test video")
            config = SearchConfig.for_library("waymo")
            config.library_dir = root / "common-data"
            config.runtime_home = root / "runtime"

            with patch("mcp_servers.sentry_video_search.core.service._ensure_dashscope_key"), patch(
                "mcp_servers.sentry_video_search.core.service.index_segments", return_value=([str(video)], 0.1)
            ) as index_segments:
                result = index_local(video, config)

            self.assertEqual(result["processed_files"], [str(video.resolve())])
            self.assertEqual(list(video.parent.iterdir()), [video])
            self.assertFalse(config.library_dir.exists())
            args, _ = index_segments.call_args
            self.assertEqual(args[0], [{"segment_video": str(video.resolve())}])
            self.assertEqual(args[1].chunk_duration, 10)
            self.assertTrue(args[1].preprocess_chunks)

    def test_agent_forwards_library_selection(self):
        response = {"status": "success", "result": {"video_path": "/tmp/match.mp4"}}
        completed = type("Completed", (), {
            "returncode": 0,
            "stdout": json.dumps(response),
            "stderr": "",
        })()
        with patch("mcp_servers.sentry_video_search.adapter.client.run_logged", return_value=completed) as subprocess_run:
            result = retrieve_video({"requirement": "highway", "library": "waymo"})

        self.assertEqual(result["result"]["video_path"], response["result"]["video_path"])
        self.assertIn("run_dir", result["result"])
        self.assertEqual(json.loads(subprocess_run.call_args.kwargs["input"])["library"], "waymo")
        with self.assertRaisesRegex(ValueError, "library must be"):
            retrieve_video({"requirement": "highway", "library": "unknown"})


if __name__ == "__main__":
    unittest.main()
