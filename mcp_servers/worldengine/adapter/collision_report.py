"""Observed closed-loop body contact, independent of PDM fault and road scores.

Positions are WE's actual body centres in its local world axes. No distance
buffer is used to turn a close pass into a collision. Keyframe contact and
interpolated possible contact are reported separately; neither certifies physics.
"""
import json
import math
from pathlib import Path

from mcp_servers.scene_editing.core.adversarial.box_geometry import body_contact, corners, point_segment


def interpolate(a,b,fraction):
    yaw=math.atan2(math.sin(b[-1]-a[-1]),math.cos(b[-1]-a[-1]))
    return [a[i]+fraction*(b[i]-a[i]) for i in range(4)]+[a[-1]+fraction*yaw]


def summarize_frames(frames,target_track,takeover_step):
    if not frames:raise ValueError('No actual closed-loop states were recorded')
    scenes={frame['scene_id'] for frame in frames}
    if len(scenes)!=1:raise ValueError('Target collision check requires exactly one scene')
    frames=sorted(frames,key=lambda row:row['step'])
    steps=[f['step'] for f in frames]
    if steps!=list(range(len(frames))):raise ValueError('Actual collision state coverage has gaps or duplicate steps')
    eligible=[f for f in frames if f['step']>takeover_step]
    if not eligible:raise ValueError('No actual state after policy takeover')
    if any('ego' not in f['actors'] for f in frames):raise ValueError('Missing actual ego body')
    target_frames=[f for f in eligible if target_track in f['actors']]
    events=[];history_events=[];closest=None
    for frame in frames:
        ego=frame['actors']['ego']['box']
        for track,actor in frame['actors'].items():
            if track=='ego':continue
            contact,gap=body_contact(ego,actor['box'])
            if contact:
                event=dict(track_id=track,time_s=frame['time_s'],step=frame['step'])
                (events if frame['step']>takeover_step else history_events).append(event)
            if track==target_track and frame['step']>takeover_step and (closest is None or gap<closest['clearance_m']):
                ev=frame['actors']['ego'].get('velocity');tv=actor.get('velocity')
                heading=ego[-1]
                relative=None if ev is None or tv is None else (ev[0]-tv[0])*math.cos(heading)+(ev[1]-tv[1])*math.sin(heading)
                dx,dy=actor['box'][0]-ego[0],actor['box'][1]-ego[1]
                closest=dict(time_s=frame['time_s'],step=frame['step'],clearance_m=gap,
                    target_forward_m=dx*math.cos(heading)+dy*math.sin(heading),
                    target_left_m=-dx*math.sin(heading)+dy*math.cos(heading),closing_speed_mps=relative)
    target_events=[event for event in events if event['track_id']==target_track]
    history_target=[event for event in history_events if event['track_id']==target_track]
    possible=[]
    for a,b in zip(frames,frames[1:]):
        if b['step']<=takeover_step or target_track not in a['actors'] or target_track not in b['actors']:continue
        count=max(1,math.ceil((b['time_s']-a['time_s'])/0.05))
        for i in range(1,count):
            fraction=i/count
            ego=interpolate(a['actors']['ego']['box'],b['actors']['ego']['box'],fraction)
            target=interpolate(a['actors'][target_track]['box'],b['actors'][target_track]['box'],fraction)
            if body_contact(ego,target)[0]:
                possible.append(a['time_s']+fraction*(b['time_s']-a['time_s']))
                break
    complete=len(target_frames)==len(eligible)
    status='completed' if complete else 'incomplete_target_coverage'
    collision=True if target_events else (False if complete else None)
    takeover_time=next(f['time_s'] for f in frames if f['step']==takeover_step)
    return dict(schema_version='native_collision.v1',status=status,scene_id=frames[0]['scene_id'],
        target_track=target_track,policy_takeover_step=takeover_step,policy_takeover_time_s=takeover_time,
        frame_count=len(frames),eligible_frame_count=len(eligible),target_observed_frame_count=len(target_frames),
        target_collision_detected=collision,any_collision_detected=bool(events),
        first_target_collision_time_s=target_events[0]['time_s'] if target_events else None,
        first_collision=events[0] if events else None,target_contact_sample_count=len(target_events),
        initial_history_target_contact=bool(history_target),
        min_target_clearance_m=None if closest is None else closest['clearance_m'],closest_approach=closest,
        possible_interpolated_target_contact=bool(possible),
        first_interpolated_target_contact_time_s=min(possible) if possible else None,
        max_observed_step_s=max((b['time_s']-a['time_s'] for a,b in zip(frames,frames[1:])),default=0),
        contact_events=events[:8],contact_events_omitted=max(0,len(events)-8),
        detection='Unbuffered oriented body rectangles including boundary contact at actual WE states; no fault attribution.',
        scope='Observed closed-loop geometry; interpolated possible contact is separate. No physical-impact or continuous-time certification.')


def collision_evaluation_scope(report, scene_ids, history, frames):
    """A collision probe verifies actual coverage without claiming an official PDM score."""
    if (report.get('status')!='completed' or report.get('target_collision_detected') is None
            or report.get('frame_count')!=frames or report.get('policy_takeover_step')!=history-1
            or scene_ids!={report.get('scene_id')}
            or abs(report.get('policy_takeover_time_s',-1)-(history-1)*.5)>1e-6):
        raise RuntimeError('Collision-only coverage does not match the completed rollout')
    return dict(scope='native_target_collision',official_pdm_score=False,status='completed',
        start_time_s=(history-1)*.5,end_time_s=(frames-1)*.5,
        num_history=history,max_frame_per_scene=frames,scene_id=report['scene_id'])


def install_recorder(path):
    from worldengine.manager.render_manager import RenderManager
    path=Path(path)
    if path.exists():raise FileExistsError('Preserve actual collision states: '+str(path))
    original=RenderManager.render
    def render(manager):
        engine=manager.engine
        actors={}
        for key,agent in engine.agent_manager.all_agents.items():
            if not agent.policy.is_current_step_valid:continue
            # WE positions can be 2D or 3D, so bounding_box has six or seven
            # entries. Use the named body properties rather than guessing z.
            position=agent.current_position
            box=[float(position[0]),float(position[1]),float(agent.length),
                 float(agent.width),float(agent.current_heading)]
            velocity=getattr(agent,'current_velocity',None)
            actors[str(key)]=dict(box=box,
                velocity=None if velocity is None else [float(velocity[0]),float(velocity[1])])
        frame=dict(scene_id=str(engine.current_scene['id']),step=int(engine.episode_step),
            time_s=float(engine.episode_step*engine.sim_time_interval),actors=actors)
        # Render first: a failed render is not a successful observed policy frame.
        result=original(manager)
        with path.open('a') as stream:stream.write(json.dumps(frame,allow_nan=False)+'\n')
        return result
    RenderManager.render=render


def write_report(states_path,report_path,target_track,takeover_step):
    frames=[json.loads(line) for line in Path(states_path).read_text().splitlines() if line]
    result=summarize_frames(frames,target_track,takeover_step)
    result.update(states_path=str(Path(states_path).resolve()),report_path=str(Path(report_path).resolve()))
    Path(report_path).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
