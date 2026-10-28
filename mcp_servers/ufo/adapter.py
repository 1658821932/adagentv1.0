"""Deterministic UFO tool adapters; DGGT keeps its own adapters unchanged."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from .service import reconstruct as reconstruct_ufo, edit_metric, edit_lane_change, calibrate_metric_scene, edit_metric_control
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_FRAME_COUNT = 40

def reconstruct_video(input_data: dict) -> dict:
    if input_data.get("num_frames", AGENT_FRAME_COUNT) != AGENT_FRAME_COUNT:
        raise ValueError("Agent reconstruction requires num_frames=40")
    video_path = input_data.get("video_path")
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("reconstruction requires a non-empty 'video_path' string")
    candidate = Path(video_path).expanduser().resolve()
    if not candidate.is_file():
        compact = "".join(candidate.name.split())
        for sibling in candidate.parent.iterdir() if candidate.parent.is_dir() else []:
            if "".join(sibling.name.split()) == compact:
                candidate = sibling
                break
    video_path = str(candidate)
    if input_data.get("export_render_state"):
        raise ValueError("UFO reconstruction does not use DGGT export_render_state")
    return reconstruct_ufo(
        video_path, scene_name=input_data.get("scene_name"), gpu=input_data.get("gpu", 0),
        entry_root=input_data.get("entry_root"), output_root=input_data.get("output_root"),
        resume=input_data.get('resume', False),
    )

def edit_scene(input_data: dict) -> dict:
    if input_data.get("num_frames", AGENT_FRAME_COUNT) != AGENT_FRAME_COUNT:
        raise ValueError("Agent scene_editor requires num_frames=40")
    if input_data.get("backend") == "ufo":
        operation = input_data.get('operation')
        if input_data.get('edit_config_path'):
            declared = json.loads(Path(input_data['edit_config_path']).expanduser().read_text()).get('operation')
            if declared in {'lane_change', 'metric_edit'}:
                if operation is not None and operation != declared:
                    raise ValueError('Editor operation differs from saved configuration')
                operation = declared
        if operation == 'calibrate_metric':
            result = calibrate_metric_scene(input_data.get('ufo_reconstruction') or {},
                output_path=input_data.get('output_path'), reference_path=input_data.get('reference_path'),
                road_plane_path=input_data.get('road_plane_path'), gpu=input_data.get('gpu', 0),
                method=input_data.get('calibration_method','depth_fit'),road_model=input_data.get('road_model','local_depth'))
            return result
        if operation == 'metric_edit':
            result = edit_metric_control(input_data.get('ufo_reconstruction') or {},
                edit_config_path=input_data['edit_config_path'], target_selection_path=input_data.get('target_selection_path'),
                output_root=input_data.get('output_root'), gpu=input_data.get('gpu', 0),
                assets_only=input_data.get('assets_only', False))
            return result
        if operation in (None, 'lane_change'):
            options = dict(input_data)
            if input_data.get('edit_config_path'):
                from mcp_servers.ufo.lane_config import validate
                config = validate(json.loads(Path(input_data['edit_config_path']).expanduser().read_text()))
                if input_data.get('target_selection_path') and Path(input_data['target_selection_path']).resolve() != Path(config['target_selection_path']).resolve():
                    raise ValueError('Editor target differs from generated config')
                options.update(config)
            result = edit_lane_change(
                input_data.get('ufo_reconstruction') or {}, track_id=options.get('track_id'),
                direction=options.get('direction', 'right'),
                lateral_distance_native=options.get('lateral_distance_native'),
                target_selection_path=options.get('target_selection_path'),
                start_fraction=options.get('start_fraction', .1), end_fraction=options.get('end_fraction', .9),
                vehicle_widths=options.get('vehicle_widths', 1.2),
                gpu=input_data.get('gpu', 0), targets_only=input_data.get('targets_only', False),
            )
            return result
        if input_data.get('operation') != 'metric':
            raise ValueError('UFO supports operation=lane_change or explicit experimental metric')
        result = edit_metric(
            input_data.get("ufo_reconstruction") or {},
            edit_config_path=input_data.get("edit_config_path"),
            target_selection_path=input_data.get("target_selection_path"),
            gpu=input_data.get("gpu", 0),
            output_asset_root=input_data.get("output_asset_root"),
        )
        return result
    if input_data.get("backend", "dggt") != "dggt":
        raise ValueError("scene_editor backend must be dggt or ufo")
    raise ValueError('UFO editor requires backend=ufo')

def discover_instances(input_data: dict) -> dict:
    if input_data.get('sequence_length', 40) != 40:
        raise ValueError('Agent instance_discovery requires sequence_length=40')
    backend = input_data.get('backend')
    if backend == 'ufo':
        from mcp_servers.ufo.runtime import python_path, offline_env
        reconstruction = input_data.get('ufo_reconstruction') or {}
        if reconstruction.get('backend') != 'ufo_worldengine_time_v2':
            raise ValueError('UFO discovery requires the successful ufo_reconstruction result')
        output = Path(input_data['output_path']).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(python_path()), '-m', 'mcp_servers.ufo.candidates',
                        '--workspace', reconstruction['workspace'], '--output', str(output)],
                       cwd=PROJECT_ROOT, env=offline_env(), check=True)
        result = json.loads(output.read_text())
        return {**result, 'instance_info_path': str(output), 'candidate_count': len(result['instances'])}
    raise ValueError('UFO discovery requires backend=ufo')

def generate_edit_config(input_data: dict) -> dict:
    if input_data.get('operation') == 'metric_edit':
        from .metric_config import generate
    else:
        from .lane_config import generate
    return generate(input_data)
