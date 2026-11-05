"""Scene provenance must survive import, reranking and clip export."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_servers.sentry_video_search.adapter.scene_binding import bind_scene, register_scene_video
from mcp_servers.sentry_video_search.core.pipeline.postfilter import import_source_video

CAMERAS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]

@pytest.fixture
def scene(tmp_path):
    sample = tmp_path / "dataset/samples/anchor"
    (sample / "metadata").mkdir(parents=True)
    (sample / "scenario").mkdir()
    (sample / "scenario/all_scenarios.pkl").write_bytes(b"not loaded")
    (sample / "manifest.json").write_text(json.dumps({"scene_id": "log-anchor", "anchor_token": "anchor"}))
    frames = []
    for i in range(3):
        cameras = {}
        for name in CAMERAS:
            relative = f"log/{name}/{i}.jpg"
            p = sample / "original_images" / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"image")
            cameras[name] = {"data_path": relative}
        frames.append({"token": str(i), "timestamp": 1_000_000 + i * 500_000, "cams": cameras})
    (sample / "metadata/frames.json").write_text(json.dumps(frames))
    (sample / "metadata/cameras.json").write_text(json.dumps({name: {"channel": name} for name in CAMERAS}))
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in CAMERAS:
        video = videos / (name + ".mp4")
        video.write_bytes(b"video")
        register_scene_video(video, sample, name)
    return sample, videos

@pytest.mark.parametrize("camera", CAMERAS)
def test_any_camera_resolves_same_complete_scene(scene, tmp_path, camera):
    sample, videos = scene
    result = bind_scene(videos / (camera + ".mp4"), tmp_path / "output")
    assert result["scene_token"] == "anchor"
    assert result["matched_camera"] == camera
    assert result["sample_root"] == str(sample)
    assert result["fps"] == 2
    assert result["images_complete"] is True
    assert set(result["cameras"]) == set(CAMERAS)
    full = json.loads(Path(result["manifest_path"]).read_text())
    assert full["timestamps_us"] == [1_000_000, 1_500_000, 2_000_000]
    for name in CAMERAS:
        assert result["cameras"][name]["available_images"] == 3
        assert Path(result["cameras"][name]["video_path"]).is_file()
        assert [x["timestamp_us"] for x in full["cameras"][name]["frames"]] == full["timestamps_us"]


def test_missing_image_is_reported_without_inventing_availability(scene, tmp_path):
    sample, videos = scene
    (sample / "original_images/log/CAM_L0/1.jpg").unlink()
    result = bind_scene(videos / "CAM_F0.mp4", tmp_path / "output")
    assert result["images_complete"] is False
    assert result["cameras"]["CAM_L0"]["available_images"] == 2


def test_unbound_video_and_missing_source_are_errors(tmp_path):
    with pytest.raises(ValueError, match="No scene binding"):
        bind_scene(tmp_path / "unknown.mp4", tmp_path)
    with pytest.raises(ValueError, match="source_file"):
        bind_scene(None, tmp_path)


def test_wrong_scene_token_and_camera_are_errors(scene, tmp_path):
    sample, videos = scene
    sidecar = videos / "CAM_F0.scene.json"
    record = json.loads(sidecar.read_text())
    record["scene_token"] = "another-scene"
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="disagree"):
        bind_scene(videos / "CAM_F0.mp4", tmp_path)
    with pytest.raises(ValueError, match="Unknown scene camera"):
        register_scene_video(videos / "CAM_F0.mp4", sample, "CAM_UNKNOWN")


def test_import_keeps_binding_with_relative_sample_root(scene, tmp_path):
    sample, videos = scene
    sidecar = videos / "CAM_R0.scene.json"
    record = json.loads(sidecar.read_text())
    record["sample_root"] = "../dataset/samples/anchor"
    sidecar.write_text(json.dumps(record))
    imported, _ = import_source_video(videos / "CAM_R0.mp4", library_dir=tmp_path / "library")
    result = bind_scene(imported, tmp_path / "output")
    assert result["sample_root"] == str(sample)
    assert result["matched_camera"] == "CAM_R0"
    assert json.loads(imported.with_suffix(".scene.json").read_text())["video_path"] == str(imported)


def test_retrieval_binds_rerank_winner_source_not_export_filename(scene, tmp_path):
    from mcp_servers.sentry_video_search.adapter.service import search_local
    from mcp_servers.sentry_video_search.core.config import SearchConfig
    sample, videos = scene
    hit = {"source_file": str(videos / "CAM_B0.mp4"), "start_time": 0.0, "end_time": 1.5, "similarity_score": 0.9}
    def export(details, requirement, config, logs):
        p = config.output_dir / "renamed_search_clip.mp4"
        p.write_bytes(b"clip")
        return p
    with patch("mcp_servers.sentry_video_search.core.service._ensure_dashscope_key"), patch("mcp_servers.sentry_video_search.core.service._search_index", return_value=[hit]), patch("mcp_servers.sentry_video_search.core.reranker.rerank_hits", return_value=(hit, {})), patch("mcp_servers.sentry_video_search.core.service._export_reranked_hits", side_effect=export):
        result = search_local("rear view", SearchConfig(runtime_home=tmp_path / "index"))
    assert result.scene_ref["scene_token"] == "anchor"
    assert result.scene_ref["matched_camera"] == "CAM_B0"
    assert result.source_file == hit["source_file"]
    assert result.source_segment == {"start_time": 0.0, "end_time": 1.5}
    assert result.video_path.endswith("renamed_search_clip.mp4")
    saved = json.loads((Path(result.run_dir) / "output/result.json").read_text())
    assert saved["scene_ref"]["matched_camera"] == "CAM_B0"


def test_adapter_rejects_unbound_retrieval(tmp_path):
    from mcp_servers.sentry_video_search.adapter.service import search_local
    from mcp_servers.sentry_video_search.core.config import SearchConfig
    from mcp_servers.sentry_video_search.core.schemas import VideoSearchResult
    raw = VideoSearchResult("test", "/clip.mp4", "/run", "local_hit", 0.9, {}, source_file="/unbound.mp4")
    with patch("mcp_servers.sentry_video_search.core.service.search_local", return_value=raw), pytest.raises(ValueError, match="No scene binding"):
        search_local("test", SearchConfig(runtime_home=tmp_path / "index"))


def test_client_rejects_legacy_unbound_subprocess_payload():
    from mcp_servers.sentry_video_search.adapter.client import retrieve_video
    result = type("Completed", (), {"returncode": 0, "stderr": "", "stdout": json.dumps({"status": "success", "result": {"video_path": "/clip.mp4"}})})()
    with patch("mcp_servers.sentry_video_search.adapter.client.run_logged", return_value=result), pytest.raises(RuntimeError, match="bound multi-camera scene"):
        retrieve_video({"requirement": "test"})
