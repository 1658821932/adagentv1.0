"""Compose corrected MTGS front/left/right candidate views with synchronized geometry."""
import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
import numpy as np

from experiment_runtime import atomic_json
from render_ego_mtgs_three_view import CAMERAS
from run_ego_reference import DEFAULT_TARGET_TRACK, EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon, scene_from_frames


def require_three_view_report(report, frame_count):
    if report.get('status') != 'passed':
        raise ValueError('Require a passed three-view render')
    rendered = report.get('rendered_frame_ids')
    if report.get('full_window') is not True:
        # trimmed window: allowed only as a contiguous prefix (e.g. the asset replay of the
        # attack vehicle ends early); every delivered frame must still be individually validated
        if not rendered or rendered != list(range(len(rendered))) or len(rendered) >= frame_count:
            raise ValueError('Trimmed render must be a contiguous prefix of the scene')
    if not rendered or len(rendered) > frame_count:
        raise ValueError('Three-view render does not cover every recorded frame')
    if tuple(report.get('cameras', [])) != CAMERAS:
        raise ValueError('Require CAM_F0, CAM_L0 and CAM_R0 in the frozen order')
    for camera_id in CAMERAS:
        if report.get('decoded_videos', {}).get(f'new-camera-{camera_id}.mp4') != len(rendered):
            raise ValueError(f'Missing complete edited video for {camera_id}')
    if report.get('camera_readback_max_abs_error', 1.) > 2e-4:
        raise ValueError('Camera readback gate failed')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--target-track', required=True)
    args = parser.parse_args()
    target_track = args.target_track
    run = args.run.resolve(); output = run/'manual-explained.mp4'
    if output.exists():
        raise FileExistsError('Refusing to overwrite manual-explained.mp4')
    config = json.loads((run/'config.json').read_text())
    report = json.loads((run/'render-report.json').read_text())
    frames = json.loads((args.input/'metadata/frames.json').read_text())
    require_three_view_report(report, len(frames))
    candidate_id = config['candidate_id']
    with (run/'input/candidate-trajectory.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    times, baseline, yaw, objects = scene_from_frames(frames, target_track)
    if len(rows) != len(times) or any(abs(float(row['time_s'])-t) > 1e-6 for row, t in zip(rows, times)):
        raise ValueError('Candidate and source time grids differ')
    candidate = np.array([[float(row['ego_x_m']), float(row['ego_y_m'])] for row in rows])
    angles = np.array([float(row['ego_yaw_rad']) for row in rows])
    target = [next(actor for actor in frame if actor['track'] == target_track) for frame in objects]
    target_xy = np.array([[actor['x'], actor['y']] for actor in target])
    limits = np.concatenate([baseline, candidate, target_xy]); lo = limits.min(axis=0)-8; hi = limits.max(axis=0)+8
    shift = (EGO_FRONT-EGO_REAR)/2
    def clearance(xy, heading, observed_target):
        center = xy+shift*np.array([np.cos(heading), np.sin(heading)])
        ego = polygon(*center, heading, EGO_FRONT+EGO_REAR, EGO_WIDTH)
        return float(ego.distance(polygon(observed_target['x'], observed_target['y'], observed_target['yaw'],
            observed_target['length'], observed_target['width'])))
    baseline_clearance = np.array([clearance(xy, heading, actor) for xy, heading, actor in zip(baseline, yaw, target)])
    candidate_clearance = np.array([clearance(xy, heading, actor) for xy, heading, actor in zip(candidate, angles, target)])
    baseline_min = float(baseline_clearance.min()); candidate_min = float(candidate_clearance.min())
    reference_distance_reached = candidate_min >= .5
    readers = {camera_id:imageio.get_reader(run/f'new-camera-{camera_id}.mp4') for camera_id in CAMERAS}
    writer = imageio.get_writer(output, fps=2., codec='libx264', macro_block_size=2)
    def car(ax, xy, heading, length, width, color, label):
        geometry = polygon(*xy, heading, length, width)
        ax.add_patch(Patch(np.asarray(geometry.exterior.coords), closed=True, fill=False,
            edgecolor=color, linewidth=1.8))
        ax.text(xy[0], xy[1], label, color=color, fontsize=7)
    try:
        for i in range(len(times)):
            fig = plt.figure(figsize=(18, 9), dpi=100)
            fig.suptitle('MANUAL EXPLAINED | CORRECTED MTGS REFERENCE CAMERAS | CARLA OFF', fontsize=13)
            positions = [[.02, .52, .29, .34], [.32, .52, .29, .34], [.02, .15, .29, .34]]
            titles = ['Front CAM_F0 | edited ego', 'Left-front CAM_L0 | edited ego',
                'Right-front CAM_R0 | edited ego']
            for camera_id, position, title in zip(CAMERAS, positions, titles):
                ax = fig.add_axes(position); ax.imshow(readers[camera_id].get_data(i)); ax.set_axis_off()
                ax.set_title(title, fontsize=10)
            info = fig.add_axes([.32, .15, .29, .34]); info.set_axis_off()
            info.text(.02, .88, 'WorldEngine MTGS reference cameras', fontsize=12, weight='bold')
            info.text(.02, .72, f'Candidate: {candidate_id}', fontsize=11)
            info.text(.02, .56, 'Same candidate ego and timestamp in all 3 views', fontsize=10)
            info.text(.02, .40, 'Recorded z / pitch / roll retained', fontsize=10)
            info.text(.02, .24, 'MTGS actor log replay; no CARLA execution', fontsize=10)
            ax = fig.add_axes([.64, .15, .34, .71])
            ax.plot(baseline[:, 0], baseline[:, 1], color='gray', alpha=.7, label='recorded ego path')
            ax.plot(candidate[:, 0], candidate[:, 1], color='blue', label='candidate ego path')
            ax.plot(target_xy[:, 0], target_xy[:, 1], color='orange', label='recorded target path')
            center = candidate[i]+shift*np.array([np.cos(angles[i]), np.sin(angles[i])])
            car(ax, center, angles[i], EGO_FRONT+EGO_REAR, EGO_WIDTH, 'blue', 'ego ref')
            actor = target[i]
            car(ax, [actor['x'], actor['y']], actor['yaw'], actor['length'], actor['width'], 'orange', 'target')
            others = sorted([item for item in objects[i] if item['track'] != target_track],
                key=lambda item:np.hypot(item['x']-center[0], item['y']-center[1]))[:8]
            for item in others:
                car(ax, [item['x'], item['y']], item['yaw'], item['length'], item['width'],
                    'purple', item['track'][:6])
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect('equal')
            ax.grid(alpha=.2); ax.legend(loc='upper right', fontsize=7)
            ax.set_title('Synchronized reference geometry | observed actors only', fontsize=10)
            ax.set_xlabel('local x (m)'); ax.set_ylabel('local y (m)')
            fig.text(.02, .10, f't={times[i]:.2f}s | target clearance={candidate_clearance[i]:.3f}m | candidate={candidate_id} | observed 2Hz', fontsize=11)
            fig.text(.02, .065, f'Minimum clearance: recorded ego={baseline_min:.3f}m | candidate={candidate_min:.3f}m | change={candidate_min-baseline_min:+.3f}m', fontsize=10)
            threshold_text = 'reference only; not an acceptance gate'
            fig.text(.02, .03, f'All views use the corrected candidate camera transform; reference only, no physical tracking; 0.5m {threshold_text}.',
                fontsize=10, color='darkgreen' if reference_distance_reached else 'darkred')
            fig.canvas.draw(); data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(data); plt.close(fig)
    finally:
        for reader in readers.values():
            reader.close()
        writer.close()
    reader = imageio.get_reader(output); metadata = reader.get_meta_data()
    decoded = sum(1 for _ in reader); reader.close()
    if decoded != len(times) or abs(float(metadata['fps'])-2.) > 1e-6:
        raise ValueError('Three-view manual decode or fps mismatch')
    atomic_json(run/'manual-explained-validation.json', dict(status='passed', video=output.name,
        frames=decoded, fps=float(metadata['fps']), candidate=candidate_id,
        left_sources={camera_id:f'new-camera-{camera_id}.mp4' for camera_id in CAMERAS},
        right_source='Synchronized candidate and recorded reference geometry',
        render_report='render-report.json', sample_rate='Observed 2 Hz; no continuous-time certification',
        baseline_min_target_clearance_m=baseline_min, candidate_min_target_clearance_m=candidate_min,
        clearance_change_m=candidate_min-baseline_min, clearance_reference_m=.5,
        clearance_reference_reached=reference_distance_reached, physical_execution='not_run', carla_executed=False,
        original_camera_used_as_edited_view=False, corrected_first_ego_transform=True))
    print('Three-view manual validated:', decoded, 'frames')


if __name__ == '__main__':
    main()
