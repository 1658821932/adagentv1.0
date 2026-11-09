"""Create a labelled road and candidate-trajectory inspection for the ego agent."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch
import numpy as np

from ego_road_checks import ego_body_polygon
from ego_edit_evidence import geometry_evidence
from ego_trajectory_edit import start_state
from run_ego_reference import polygon


def create_inspection(editor, candidate_id, path):
    """Draw an archived candidate and its parent at the candidate's closest frame."""
    if candidate_id not in editor.nodes:
        raise ValueError('Unknown candidate_id')
    path = Path(path)
    if path.exists():
        raise FileExistsError('Refusing to overwrite inspection')
    node = editor.nodes[candidate_id]
    parent_id = node['parent_id'] or candidate_id
    parent = editor.nodes[parent_id]
    unranked=getattr(editor,'unranked_candidates',False)
    eligible=np.flatnonzero(editor.times>editor.effective_policy['observation_end_s']) if unranked else np.arange(len(editor.times))
    frame_index = int(eligible[np.argmin(node['analysis']['clearance'][eligible])])
    target = next(actor for actor in editor.actors[frame_index] if actor['track'] == editor.target_track)
    if unranked:
        fig, panels=plt.subplots(2,2,figsize=(13,9),dpi=100)
        axes=panels[0]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    geometries = getattr(editor.road.geometry, 'geoms', [editor.road.geometry])
    for ax in axes:
        for shape in geometries:
            if shape.geom_type != 'Polygon':
                continue
            ax.add_patch(Patch(np.asarray(shape.exterior.coords), facecolor='#d9ead3', edgecolor='#9cbca0', alpha=.8))
            for ring in shape.interiors:
                ax.add_patch(Patch(np.asarray(ring.coords), facecolor='white', edgecolor='#9cbca0'))
        ax.plot(parent['xy'][:, 0], parent['xy'][:, 1], '--', color='gray', label=f'parent {parent_id}')
        ax.plot(node['xy'][:, 0], node['xy'][:, 1], color='blue', label=f'candidate {candidate_id}')
        positions = [next(a for a in frame if a['track'] == editor.target_track) for frame in editor.actors]
        ax.plot([a['x'] for a in positions], [a['y'] for a in positions], color='orange', label='fixed attack trajectory (moving vehicle)' if unranked else 'frozen attack target')
        for candidate, color, label in [(parent, 'gray', 'parent ego'), (node, 'blue', 'candidate ego')]:
            body = ego_body_polygon(*candidate['xy'][frame_index], candidate['yaw'][frame_index])
            ax.add_patch(Patch(np.asarray(body.exterior.coords), fill=False, edgecolor=color, linewidth=2, label=label))
        body = polygon(target['x'], target['y'], target['yaw'], target['length'], target['width'])
        ax.add_patch(Patch(np.asarray(body.exterior.coords), fill=False, edgecolor='orange', linewidth=2, label='target at frame'))
        ax.set_aspect('equal'); ax.grid(alpha=.2); ax.set_xlabel('local x (m)'); ax.set_ylabel('local y (m)')
    all_xy = np.vstack([node['xy'], parent['xy'], [[a['x'], a['y']] for a in positions]])
    axes[0].set_xlim(all_xy[:, 0].min()-5, all_xy[:, 0].max()+5)
    axes[0].set_ylim(all_xy[:, 1].min()-5, all_xy[:, 1].max()+5)
    failed = node['result']['road']['failing_samples']
    if failed:
        worst = min(failed, key=lambda sample:sample['margin_m'])
        axes[0].scatter([worst['x']], [worst['y']], marker='x', s=70, color='red',
            label=f"worst road margin at {worst['time_s']:.2f}s", zorder=10)
    axes[0].set_title('Full trajectories'); axes[0].legend(fontsize=6, loc='best')
    center = (node['xy'][frame_index]+np.array([target['x'], target['y']]))/2
    span = max(9., float(np.max(np.abs(node['xy'][frame_index]-[target['x'], target['y']])))/2+8.)
    axes[1].set_xlim(center[0]-span, center[0]+span); axes[1].set_ylim(center[1]-span, center[1]+span)
    road = node['result']['road']
    pictured_gap=float(node['analysis']['clearance'][frame_index])
    focus='Closest editable sample' if unranked else 'Closest observed frame'
    axes[1].set_title(f"{focus} {frame_index} | t={editor.times[frame_index]:.2f}s\n"
        f"gap={pictured_gap:.3f}m | full-window min road margin={road['min_margin_m']:.3f}m")
    if unranked:
        boundary=editor.effective_policy['observation_end_s']
        gap_ax,speed_ax=panels[1]
        gap_ax.plot(editor.times,editor.baseline_analysis['clearance'],'--',color='gray',label='recorded ego vs fixed attack')
        gap_ax.plot(editor.times,node['analysis']['clearance'],color='blue',label='candidate vs fixed attack')
        gap_ax.scatter([editor.times[frame_index]],[pictured_gap],color='blue')
        gap_ax.set_ylabel('Body clearance (m)');gap_ax.set_title('Clearance over time; fixed history shaded')
        dt=np.diff(editor.times)
        for points,color,label in [(node['xy'],'blue','candidate ego'),
                (editor.baseline_xy,'gray','recorded ego'),
                (np.array([[a['x'],a['y']] for a in positions]),'orange','attack vehicle')]:
            speed_ax.step(editor.times[:-1],np.linalg.norm(np.diff(points,axis=0),axis=1)/dt,
                          where='post',color=color,label=label)
        speed_ax.set_ylabel('Sampled interval speed (m/s)');speed_ax.set_title('Speed over time')
        for ax in (gap_ax,speed_ax):
            ax.axvspan(editor.times[0],boundary,color='gray',alpha=.15)
            ax.set_xlabel('Time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Map lane area and reference geometry | not physical execution')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100); plt.close(fig)
    return {'candidate_id':candidate_id, 'parent_candidate_id':parent_id, 'frame_index':frame_index,
        'time_s':float(editor.times[frame_index]), 'path':str(path), 'source':'generated reference diagram',
        'min_clearance_m':pictured_gap, 'full_horizon_min_clearance_m':node['result']['min_target_clearance_m'],
        'inspection_scope':'editable-period nearest native sample' if unranked else 'full-horizon nearest native sample',
        'min_road_margin_m':road['min_margin_m'],
        'road_failing_obs_frames':road['failing_obs_frames'],
        'geometry_at_pictured_frame':geometry_evidence(
            editor.times,node['xy'],node['yaw'],editor.actors,editor.target_track,float(editor.times[frame_index])),
        'actual_start_state':(node['result'].get('generation_curve',{}).get('actual_start_state') or start_state(
            editor.times,node['xy'],node['yaw'],editor.effective_policy['observation_end_s'])),
        'note':'Road margin can occur at a different time than pictured; checks are authoritative.'}
