"""Exercise Agent reconstruction and scene_editor tools without starting the Agent."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess

from .runtime import REMOTE_PROJECT, offline_env, python_path
from .offline_pipeline import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=REMOTE_PROJECT / 'mydata/高速第一视角-去水印版.mp4')
    parser.add_argument('--name', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output-root', type=Path, default=REMOTE_PROJECT / 'outputs/ufo_tool_tests')
    parser.add_argument('--track-id', type=int)
    parser.add_argument('--direction', choices=['left', 'right'], default='right')
    parser.add_argument('--lateral-distance-native', type=float)
    parser.add_argument('--start-fraction', type=float, default=.1)
    parser.add_argument('--end-fraction', type=float, default=.9)
    parser.add_argument('--vehicle-widths', type=float, default=1.2)
    parser.add_argument('--target-selection-path')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--check-imports', action='store_true', help='Run exhaustive import diagnostics and exit')
    parser.add_argument('--skip-reconstruction', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse completed stages after adapter repairs')
    parser.add_argument('--reconstruction-only', action='store_true')
    parser.add_argument('--targets-only', action='store_true')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    from .service import SCENE_NAME
    if not SCENE_NAME.fullmatch(args.name):
        parser.error('--name accepts only ASCII letters, digits, underscore and dash')
    os.environ.update(offline_env(args.gpu))
    if args.check_only or args.check_imports:
        command = [str(python_path()), '-m', 'mcp_servers.ufo.offline_pipeline',
                   '--video', str(args.video.resolve()), '--check-only']
        if args.check_imports:
            command.append('--check-imports')
        subprocess.run(command, env=offline_env(args.gpu), check=True)
        return
    # Reconstruction performs its own lightweight path check. Do not start a
    # second Python process just to import the entire model stack beforehand.
    from tools.tool06_reconstruction.real import run as reconstruct_tool
    from tools.tool09_scene_editor.real import run as editor_tool
    workspace = args.output_root.resolve() / args.name
    saved = workspace / 'tool_reconstruction.json'
    if args.skip_reconstruction:
        response = json.loads(saved.read_text())
        if Path(response['result']['source_video_path']).resolve() != args.video.resolve():
            raise ValueError('Saved reconstruction belongs to a different video')
    else:
        print('CALL reconstruction backend=ufo, num_frames=40', flush=True)
        response = reconstruct_tool({'backend': 'ufo', 'video_path': str(args.video.resolve()),
                                     'scene_name': args.name, 'gpu': args.gpu, 'num_frames': 40,
                                     'output_root': str(args.output_root.resolve()), 'resume': args.resume})
        write_json(saved, response)
    if args.reconstruction_only:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return
    print('CALL scene_editor backend=ufo, operation=lane_change', flush=True)
    edited = editor_tool({'backend': 'ufo', 'operation': 'lane_change', 'ufo_reconstruction': response['result'],
                          'gpu': args.gpu, 'track_id': args.track_id, 'direction': args.direction,
                          'lateral_distance_native': args.lateral_distance_native, 'targets_only': args.targets_only,
                          'start_fraction': args.start_fraction, 'end_fraction': args.end_fraction,
                          'vehicle_widths': args.vehicle_widths, 'target_selection_path': args.target_selection_path})
    write_json(workspace / 'tool_scene_editor_v2.json', edited)
    result = edited['result']
    print(json.dumps({key: result.get(key) for key in ('track_id', 'direction', 'units', 'lateral_distance_native',
          'rendered_video', 'comparison_video', 'target_preview', 'asset_manifest', 'targets')},
          ensure_ascii=False, indent=2))
    print('TOOL_TEST_COMPLETE', workspace, flush=True)


if __name__ == '__main__':
    main()
