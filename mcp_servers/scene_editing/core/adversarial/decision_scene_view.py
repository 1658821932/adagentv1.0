"""Small decision views and BEV evidence; authoritative geometry stays untouched."""
from pathlib import Path
import numpy as np


GUIDANCE = ('Use the BEV for road/actor layout and the candidate diagram for behavior. '
    'Numbers below are rounded decision summaries, not replacement geometry. '
    'All actors and native timestamps remain in deterministic checks. '
    'Recorded ego is a reference, not a tested VADv2 rollout. '
    'Original cameras predate editing; a generated BEV describes the supplied scene version.')


def prepare_attack_scene(editor, folder):
    from experiment_runtime import atomic_json
    folder=Path(folder)
    path=render_overview(editor.times,editor.ego,editor.ego_yaw,editor.actors,editor.road,
        folder/'scene-overview.png',editor.target['track_id'],editor.seed_xy,
        event_window_s=editor.constraints.get('event_window_s'))
    atomic_json(folder/'scene-evidence.json',dict(times_s=editor.times.tolist(),actors=editor.actors,
        ego_xy_m=editor.ego.tolist(),target_seed_xy_m=editor.seed_xy.tolist(),scope='Full precision scene evidence used by checks'))
    return dict(candidate_id='seed_000',path=str(path),purpose='scene_layout')


def actor_view(times, ego, actors, target_track=None, target_xy=None, limit=12):
    """Select by closest approach over the entire horizon, not just frame zero."""
    tracks = {}
    for index, frame in enumerate(actors):
        for actor in frame:
            track = actor['track']
            xy = np.array([actor['x'], actor['y']], dtype=float)
            distance = float(np.linalg.norm(xy-ego[index]))
            if target_xy is not None:
                distance = min(distance, float(np.linalg.norm(xy-target_xy[index])))
            entry = tracks.setdefault(track, dict(min_distance=float('inf'), samples=[]))
            entry['samples'].append((index, actor))
            entry['min_distance'] = min(entry['min_distance'], distance)
    ordered = sorted(tracks, key=lambda key: (key != target_track, tracks[key]['min_distance'], key))
    chosen = ordered[:limit]
    rows = []
    for track in chosen:
        samples = tracks[track]['samples']
        nearest = min(samples, key=lambda item: np.linalg.norm(
            np.array([item[1]['x'],item[1]['y']])-ego[item[0]]))
        selected = dict((index, actor) for index, actor in (samples[0], nearest, samples[-1]))
        facts = []
        for index, actor in sorted(selected.items()):
            facts.append(dict(time_s=round(float(times[index]),3),
                center_xy_m=[round(float(actor['x']),2),round(float(actor['y']),2)],
                heading_rad=round(float(actor.get('yaw',0)),3)))
        rows.append(dict(track_id=track, diagram_label=track[:6],
            closest_reference_center_distance_m=round(tracks[track]['min_distance'],2),
            key_states=facts))
    return dict(actors=rows,total_actor_count=len(tracks),omitted_actor_count=max(0,len(tracks)-len(chosen)),
        selection='target plus nearest approaches over the full horizon; spatial relevance, not candidate quality ranking',
        distance_kind='reference center distance, not bumper clearance',precision='display only; full precision is used by tools')


def render_overview(times, ego, yaw, actors, road, path, target_track=None, target_xy=None, event_window_s=None, route_time_labels=False, spatial_hint=None):
    """Two readable scene snapshots, without any WE/GPU/video execution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from run_ego_reference import polygon, EGO_FRONT, EGO_REAR, EGO_WIDTH
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    ego=np.asarray(ego); yaw=np.asarray(yaw)
    if target_xy is None and target_track:
        target_xy=np.array([[next((a['x'] for a in frame if a['track']==target_track),np.nan),
            next((a['y'] for a in frame if a['track']==target_track),np.nan)] for frame in actors])
    usable_target=target_xy is not None and np.isfinite(target_xy).all()
    import acceptance_policy as policy
    window=event_window_s or [policy.ATTACK_WINDOW_START_S,policy.ATTACK_WINDOW_END_S]
    relevant=np.flatnonzero((np.asarray(times)>=window[0]-.001)&(np.asarray(times)<=window[1]+.001))
    if not len(relevant): relevant=np.arange(len(times))
    focus=int(relevant[np.argmin(np.linalg.norm(np.asarray(target_xy)[relevant]-ego[relevant],axis=1))]) if usable_target else int(relevant[len(relevant)//2])
    figure, axes=plt.subplots(1,2,figsize=(12,5),dpi=110)
    roster=actor_view(times,ego,actors,target_track,target_xy if usable_target else None)
    visible={row['track_id'] for row in roster['actors']}
    extent=np.vstack([ego,np.asarray(target_xy)]) if usable_target else ego
    for axis,index in zip(axes,(0,focus)):
        for geometry in getattr(road,'polygons',[]):
            axis.add_patch(Patch(np.asarray(geometry.exterior.coords),facecolor='#edf3e8',edgecolor='#bac8b0',linewidth=.5))
        axis.plot(ego[:,0],ego[:,1],color='#1565c0',label='recorded ego reference')
        if route_time_labels:
            queries=np.unique(np.clip(np.r_[1.5,np.linspace(window[0],window[1],5)],times[0],times[-1]))
            points=np.column_stack([np.interp(queries,times,ego[:,a]) for a in (0,1)])
            axis.scatter(points[:,0],points[:,1],s=12,color='#1565c0')
            for t,p in zip(queries,points):axis.annotate(f'{t:.1f}s',p,fontsize=7,xytext=(3,4),textcoords='offset points')
        center=ego[index]+(EGO_FRONT-EGO_REAR)/2*np.array([np.cos(yaw[index]),np.sin(yaw[index])])
        body=polygon(*center,yaw[index],EGO_FRONT+EGO_REAR,EGO_WIDTH)
        axis.add_patch(Patch(np.asarray(body.exterior.coords),fill=False,edgecolor='#1565c0',linewidth=2))
        axis.text(*center,'EGO',fontsize=8,color='#1565c0')
        axis.arrow(*center,1.5*np.cos(yaw[index]),1.5*np.sin(yaw[index]),width=.06,color='#1565c0',length_includes_head=True)
        for actor in actors[index]:
            if actor['track'] not in visible: continue
            color='#d84315' if actor['track']==target_track else '#777777'
            body=polygon(actor['x'],actor['y'],actor.get('yaw',0),actor.get('length',4.5),actor.get('width',1.8))
            axis.add_patch(Patch(np.asarray(body.exterior.coords),facecolor=color,edgecolor=color,alpha=.35))
            axis.text(actor['x'],actor['y'],actor['track'][:6],fontsize=7,color=color,clip_on=True)
        if usable_target:
            axis.plot(np.asarray(target_xy)[:,0],np.asarray(target_xy)[:,1],color='#d84315',label='target scene trajectory')
        if isinstance(spatial_hint, dict):
            cands_st = spatial_hint.get('attack_start_candidates_xy_m') or (
                [spatial_hint['attack_start_xy_m']] if spatial_hint.get('attack_start_xy_m') else []
            )
            for idx_s, st in enumerate(cands_st, 1):
                if isinstance(st, (list, tuple)) and len(st) == 2:
                    axis.scatter([st[0]], [st[1]], s=110, marker='*', color='#2e7d32', zorder=12,
                                 label='Tool04 start hints (S1..S3)' if idx_s == 1 else None)
                    axis.annotate(f'S{idx_s}', st, fontsize=7.5, fontweight='bold', color='#1b5e20', xytext=(4, 4), textcoords='offset points')
            cz = spatial_hint.get('conflict_zone_xy_m')
            if isinstance(cz, (list, tuple)) and len(cz) == 2:
                axis.scatter([cz[0]], [cz[1]], s=70, marker='x', color='#ff1744', zorder=12, label='Tool04 conflict hint')
                axis.annotate('conflict_hint', cz, fontsize=7, color='#b71c1c', xytext=(4, -8), textcoords='offset points')
        axis.set_xlim(extent[:,0].min()-15,extent[:,0].max()+15)
        axis.set_ylim(extent[:,1].min()-12,extent[:,1].max()+12)
        axis.set_aspect('equal');axis.grid(alpha=.2)
        axis.set_xlabel('scene x / initial ego forward (m)');axis.set_ylabel('scene y / initial ego left (m)')
        axis.set_title(f'Scene layout at t={float(times[index]):.3f}s')
    figure.suptitle('BEV geometry of supplied scene; not camera rendering or policy simulation',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    figure.tight_layout(rect=[0,.06,1,.95]);figure.savefig(path);plt.close(figure)
    return path
