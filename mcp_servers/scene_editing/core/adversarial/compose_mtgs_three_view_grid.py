"""Compose three MTGS cameras on top and three synchronized demonstrations below."""
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

from compose_mtgs_three_view import require_three_view_report
from experiment_runtime import atomic_json
from render_ego_mtgs_three_view import CAMERAS
from run_ego_reference import DEFAULT_TARGET_TRACK, EGO_FRONT, EGO_REAR, EGO_WIDTH, polygon, scene_from_frames

DISPLAY_CAMERAS = ('CAM_L0', 'CAM_F0', 'CAM_R0')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--render-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-note', default='')
    parser.add_argument('--target-track', required=True)
    args = parser.parse_args()
    target_track = args.target_track
    render_run = args.render_run.resolve(); output = args.output.resolve()
    if (output/'manual-explained.mp4').exists():
        raise FileExistsError('Refusing to overwrite an existing grid composition')
    output.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((render_run/'config.json').read_text())
    report = json.loads((render_run/'render-report.json').read_text())
    frames = json.loads((args.input/'metadata/frames.json').read_text())
    require_three_view_report(report, len(frames))
    candidate_id = source_config['candidate_id']
    atomic_json(output/'config.json', dict(version='target_ego_mtgs_three_view_grid.v1',
        render_run=str(render_run), candidate_id=candidate_id,
        layout='CAM_L0/CAM_F0/CAM_R0 on top; global trajectory/current conflict/clearance history below',
        physical_execution='not_run', diagnostic_note=args.diagnostic_note))
    with (render_run/'input/candidate-trajectory.csv').open(newline='') as stream:
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
    def center(xy, heading):
        return xy+shift*np.array([np.cos(heading), np.sin(heading)])
    def clearance(xy, heading, observed_target):
        ego = polygon(*center(xy, heading), heading, EGO_FRONT+EGO_REAR, EGO_WIDTH)
        return float(ego.distance(polygon(observed_target['x'], observed_target['y'], observed_target['yaw'],
            observed_target['length'], observed_target['width'])))
    baseline_clearance = np.array([clearance(xy, heading, actor) for xy, heading, actor in zip(baseline, yaw, target)])
    candidate_clearance = np.array([clearance(xy, heading, actor) for xy, heading, actor in zip(candidate, angles, target)])
    baseline_min = float(baseline_clearance.min()); candidate_min = float(candidate_clearance.min())
    readers = {camera_id:imageio.get_reader(render_run/f'new-camera-{camera_id}.mp4') for camera_id in CAMERAS}
    rendered_ids = report['rendered_frame_ids']
    video_path = output/'manual-explained.mp4'
    writer = imageio.get_writer(video_path, fps=2., codec='libx264', macro_block_size=2)
    def car(ax, xy, heading, length, width, color, label):
        geometry = polygon(*xy, heading, length, width)
        ax.add_patch(Patch(np.asarray(geometry.exterior.coords), closed=True, fill=False,
            edgecolor=color, linewidth=1.8))
        ax.text(xy[0], xy[1], label, color=color, fontsize=7, clip_on=True)
    try:
        for video_index, i in enumerate(rendered_ids):
            fig = plt.figure(figsize=(18, 10), dpi=100)
            title = 'MANUAL EXPLAINED | THREE MTGS REFERENCE CAMERAS + SYNCHRONIZED DEMOS | CARLA OFF'
            if args.diagnostic_note:
                title += ' | '+args.diagnostic_note
            fig.suptitle(title, fontsize=11 if args.diagnostic_note else 13)
            xs = [.01, .34, .67]
            titles = ['Left-front CAM_L0', 'Front CAM_F0', 'Right-front CAM_R0']
            for x, camera_id, title in zip(xs, DISPLAY_CAMERAS, titles):
                ax = fig.add_axes([x, .54, .32, .36]); ax.imshow(readers[camera_id].get_data(video_index))
                ax.set_axis_off(); ax.set_title(f'{title} | edited ego reference camera', fontsize=10)
            global_ax = fig.add_axes([.045, .17, .285, .32])
            global_ax.plot(baseline[:, 0], baseline[:, 1], color='gray', alpha=.7, label='recorded ego')
            global_ax.plot(candidate[:, 0], candidate[:, 1], color='blue', label='candidate ego')
            global_ax.plot(target_xy[:, 0], target_xy[:, 1], color='orange', label='target (scene trajectory)')
            car(global_ax, center(candidate[i], angles[i]), angles[i], EGO_FRONT+EGO_REAR,
                EGO_WIDTH, 'blue', 'ego ref')
            actor = target[i]
            car(global_ax, [actor['x'], actor['y']], actor['yaw'], actor['length'], actor['width'], 'orange', 'target')
            global_ax.set_xlim(lo[0], hi[0]); global_ax.set_ylim(lo[1], hi[1]); global_ax.set_aspect('equal')
            global_ax.grid(alpha=.2); global_ax.legend(loc='upper right', fontsize=7)
            global_ax.set_title('Global trajectory | current bodies only', fontsize=10)
            global_ax.set_xlabel('local x (m)'); global_ax.set_ylabel('local y (m)')
            conflict_ax = fig.add_axes([xs[1], .17, .32, .32])
            nearby = np.arange(max(0, i-5), min(len(times), i+6))
            conflict_ax.plot(candidate[nearby, 0], candidate[nearby, 1], color='blue', label='candidate ±5 frames')
            conflict_ax.plot(target_xy[nearby, 0], target_xy[nearby, 1], color='orange', label='target ±5 frames')
            car(conflict_ax, center(candidate[i], angles[i]), angles[i], EGO_FRONT+EGO_REAR,
                EGO_WIDTH, 'blue', 'ego ref')
            car(conflict_ax, center(baseline[i], yaw[i]), yaw[i], EGO_FRONT+EGO_REAR,
                EGO_WIDTH, 'gray', 'ego recorded')
            car(conflict_ax, [actor['x'], actor['y']], actor['yaw'], actor['length'], actor['width'], 'orange', 'target')
            others = sorted([item for item in objects[i] if item['track'] != target_track],
                key=lambda item:np.hypot(item['x']-candidate[i, 0], item['y']-candidate[i, 1]))[:8]
            for item in others:
                car(conflict_ax, [item['x'], item['y']], item['yaw'], item['length'], item['width'],
                    'purple', item['track'][:6])
            conflict_center = (center(candidate[i], angles[i])+np.array([actor['x'], actor['y']]))/2
            separation = np.abs(center(candidate[i], angles[i])-np.array([actor['x'], actor['y']]))
            span = max(9., float(separation.max())/2+8.)
            conflict_ax.set_xlim(conflict_center[0]-span, conflict_center[0]+span)
            conflict_ax.set_ylim(conflict_center[1]-span, conflict_center[1]+span)
            conflict_ax.set_aspect('equal'); conflict_ax.grid(alpha=.2)
            conflict_ax.legend(loc='upper right', fontsize=7)
            conflict_ax.set_title(f'Current conflict | target gap={candidate_clearance[i]:.3f}m', fontsize=10)
            conflict_ax.set_xlabel('local x (m)'); conflict_ax.set_ylabel('local y (m)')
            history_ax = fig.add_axes([xs[2], .17, .32, .32])
            history_ax.plot(times, baseline_clearance, color='gray', alpha=.75, label='recorded ego')
            history_ax.plot(times, candidate_clearance, color='blue', label='candidate ego')
            history_ax.axhline(.5, color='darkred', linestyle='--', linewidth=1.2, label='0.5m reference only')
            history_ax.axvline(times[i], color='black', linestyle=':', linewidth=1.2)
            history_ax.scatter([times[i]], [candidate_clearance[i]], color='blue', s=22, zorder=5)
            # Keep the close-range view for near misses; show the full candidate curve otherwise.
            history_limit = (6. if candidate_min <= 6. else
                             float(np.ceil(np.max(candidate_clearance) * 1.05)))
            history_ax.set_xlim(times[0], times[-1]); history_ax.set_ylim(0, history_limit)
            history_ax.grid(alpha=.2); history_ax.legend(loc='upper right', fontsize=7)
            history_ax.set_title(f'Target clearance history | zoom 0–{history_limit:g}m', fontsize=10)
            history_ax.set_xlabel('time (s)'); history_ax.set_ylabel('clearance (m)')
            fig.text(.01, .105, f't={times[i]:.2f}s | candidate={candidate_id} | observed 2Hz | all panels synchronized', fontsize=11)
            fig.text(.01, .07, f'Minimum clearance: recorded ego={baseline_min:.3f}m | candidate={candidate_min:.3f}m | change={candidate_min-baseline_min:+.3f}m', fontsize=10)
            fig.text(.01, .035, 'MTGS reference cameras; recorded actors; no physical tracking. 0.5m is a visual reference, not an acceptance threshold.',
                fontsize=10, color='darkred')
            fig.canvas.draw(); data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(data); plt.close(fig)
    finally:
        for reader in readers.values():
            reader.close()
        writer.close()
    reader = imageio.get_reader(video_path); metadata = reader.get_meta_data()
    decoded = sum(1 for _ in reader); reader.close()
    if decoded != len(rendered_ids) or abs(float(metadata['fps'])-2.) > 1e-6:
        raise ValueError('Grid manual decode or fps mismatch')
    atomic_json(output/'manual-explained-validation.json', dict(status='passed', video=video_path.name,
        frames=decoded, fps=float(metadata['fps']), candidate=candidate_id,
        rendered_frame_ids=rendered_ids,
        trimmed_window=None if len(rendered_ids) == len(times) else
            'video covers scene frames %d..%d of %d; later frames excluded because the attack '
            'vehicle asset replay ends early (see render report)' % (rendered_ids[0], rendered_ids[-1], len(times)),
        layout='Top: CAM_L0/CAM_F0/CAM_R0. Bottom: global trajectory/current conflict/clearance history',
        render_run=str(render_run), render_report='render-report.json',
        sample_rate='Observed 2 Hz; no continuous-time certification',
        baseline_min_target_clearance_m=baseline_min, candidate_min_target_clearance_m=candidate_min,
        clearance_change_m=candidate_min-baseline_min, clearance_reference_m=.5,
        clearance_reference_is_acceptance_threshold=False, physical_execution='not_run', carla_executed=False,
        original_camera_used_as_edited_view=False, corrected_first_ego_transform=True))
    print('Grid manual validated:', decoded, 'frames')


if __name__ == '__main__':
    main()
