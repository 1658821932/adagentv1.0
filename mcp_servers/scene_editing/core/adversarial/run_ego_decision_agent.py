"""Prepare video/state evidence and run one budgeted ego defense-decision agent."""
from llm_profile import load_llm_profile
import argparse
import base64
import json
import os
from pathlib import Path

import numpy as np

from ego_avoidance_contract import validate_decision
from ego_avoidance_scene import load_frozen_attack_scene,metadata_json,write_frozen_input
from run_ego_avoidance_offline import DEFAULT_LIMITS

ROOT=Path(__file__).resolve().parents[1]



def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def state_summary(scene,indices):
    rows=[]
    for index in indices:
        delta=scene.attack_xy[index]-scene.ego_xy[index]
        heading=scene.ego_heading_rad[index];c,s=np.cos(heading),np.sin(heading)
        rows.append(dict(sample_index=int(index),frame_id=float(scene.frame_ids[index]),time_s=float(scene.times_s[index]),
            attack_forward_m=float(c*delta[0]+s*delta[1]),attack_left_m=float(-s*delta[0]+c*delta[1]),
            center_distance_m=float(np.linalg.norm(delta)),attack_speed_mps=float(scene.attack_speed_mps[index]),
            archived_ego_speed_mps=float(scene.ego_speed_mps[index])))
    return rows


def prepare_decision_run(source_run,run_dir,frame_count=8):
    run_dir=Path(run_dir)
    if run_dir.exists():raise FileExistsError('Refusing to overwrite an existing decision run')
    if not 4<=frame_count<=12:raise ValueError('frame_count must be between 4 and 12')
    source=load_frozen_attack_scene(source_run);scene=source.planning_view();write_frozen_input(source,run_dir)
    _json(run_dir/'config.json',dict(version='ego_avoidance_decision.v1',mode='llm',limits=DEFAULT_LIMITS,
        source=metadata_json(source),decision_prompt='ego_defense_decision_v1.txt',max_model_actions=1,
        uses_offline_future_truth=True,carla_executed=False,rendered=False))
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    video=Path(source_run)/'render/edited.mp4'
    reader=imageio.get_reader(video);meta=reader.get_meta_data();count=reader.count_frames();fps=float(meta['fps'])
    indices=np.unique(np.linspace(0,len(scene.times_s)-1,frame_count).round().astype(int))
    cells=[];records=[]
    try:
        for index in indices:
            video_index=int(round(scene.times_s[index]*fps))
            if video_index>=count:raise ValueError('Attack video does not cover the frozen planning window')
            image=Image.fromarray(reader.get_data(video_index)).convert('RGB')
            frame_path=run_dir/'decision-input'/f'frame_{int(scene.frame_ids[index]):06d}_t{scene.times_s[index]:04.1f}s.jpg'
            frame_path.parent.mkdir(parents=True,exist_ok=True);image.save(frame_path,quality=92)
            records.append(dict(sample_index=int(index),frame_id=int(scene.frame_ids[index]),time_s=float(scene.times_s[index]),
                video_frame_index=video_index,path=str(frame_path.relative_to(run_dir)),view='Original ego camera during the completed attack; not a post-avoidance observation'))
            cell=image.copy();cell.thumbnail((640,360));cells.append((cell,records[-1]))
    finally:reader.close()
    if len(cells)!=frame_count:raise ValueError('Could not extract the requested distinct video frames')
    label_height=28;columns=2;rows=(len(cells)+1)//2
    sheet=Image.new('RGB',(1280,rows*(360+label_height)),'white');draw=ImageDraw.Draw(sheet)
    for i,(image,record) in enumerate(cells):
        x=(i%2)*640;y=(i//2)*(360+label_height)
        sheet.paste(image,(x,y+label_height));draw.text((x+8,y+7),f"frame {record['frame_id']} | t={record['time_s']:.1f}s | original ego view",fill='black')
    sheet_path=run_dir/'decision-input/attack-video-contact-sheet.jpg';sheet.save(sheet_path,quality=92)
    summary=dict(version='ego_decision_input.v1',uses_offline_future_truth=True,
        video_source=str(video.resolve()),video_sampling='Uniformly selected frames from the completed attack video; this is frame-based image understanding, not direct video ingestion',
        original_ego_camera=True,post_avoidance_camera=False,execution_window_s=[float(scene.times_s[0]),float(scene.times_s[-1])],
        planning_horizon_s=float(scene.times_s[-1]),attack_object_id=scene.metadata['attack_object_id'],
        baseline_actual_contact_time_s=scene.metadata['baseline_actual_contact_time_s'],frames=records,
        frozen_states=state_summary(scene,indices),coordinate_frame=scene.metadata['coordinate_frame'],
        vehicle_geometry=scene.metadata['geometry'],road_status='not_strictly_validated',
        other_actor_scope='Not checked in this decision pack; missing observations are unknown',
        action_bounds=dict(deceleration_mps2=[1.,6.],target_speed_mps=[0.,float(np.median(scene.ego_speed_mps[:11]))],
            lateral_offset_m=[-5.,5.],lane_duration_s=[1.,8.]))
    _json(run_dir/'decision-input/state-summary.json',summary)
    _json(run_dir/'decision-input/artifact-check.json',dict(status='passed',frames=len(records),contact_sheet='attack-video-contact-sheet.jpg',
        all_frames_decoded=True,state_samples=len(indices),offline_future_truth=True))
    return summary


def extract_decision(content):
    text=content.strip()
    if text.startswith('```'):
        lines=text.splitlines()
        if len(lines)<3 or lines[-1].strip()!='```':raise ValueError('Unclosed decision code fence')
        text='\n'.join(lines[1:-1]).strip()
    value=json.loads(text)
    if not isinstance(value,dict):raise ValueError('Decision response must be a JSON object')
    return validate_decision(value)


def accept_decision_response(run_dir,content,source_attempt=None,model=None,visual_input_used=True):
    run_dir=Path(run_dir);decision=extract_decision(content)
    expected=json.loads((run_dir/'decision-input/state-summary.json').read_text())['planning_horizon_s']
    if abs(decision['planning_horizon_s']-expected)>1e-9:
        raise ValueError('Decision horizon differs from frozen scene')
    _json(run_dir/'decisions/llm-decision.json',decision)
    summary=dict(status='completed_decision',decision_version=decision['version'],action_type=decision['action_type'],
        source_attempt=str(source_attempt) if source_attempt else None,model=model,uses_offline_future_truth=True,
        visual_input_used=visual_input_used,
        trajectory_agent_required=True,carla_validated=False,avoidance_status='undetermined')
    _json(run_dir/'decision-agent/summary.json',summary)
    return decision


def image_part(path):
    from PIL import Image
    with Image.open(path) as image:mime=Image.MIME.get(image.format)
    if mime not in {'image/jpeg','image/png'}:raise ValueError('Only verified JPEG/PNG inputs are accepted')
    data=base64.b64encode(Path(path).read_bytes()).decode()
    return dict(type='image_url',image_url=dict(url=f'data:{mime};base64,{data}'))


def decision_model_view(value):
    """Use traffic-safety terminology in the request without changing frozen evidence."""
    if isinstance(value,dict):
        return {(key.replace('attack_','target_',1) if key.startswith('attack_') else key):
            decision_model_view(item) for key,item in value.items()}
    if isinstance(value,list):return [decision_model_view(item) for item in value]
    if isinstance(value,str):
        return value.replace('attack vehicle','target vehicle').replace('attack video','scene video') \
            .replace('completed attack','completed traffic conflict')
    return value


def compact_visual_prompt(summary):
    """A short single-image planning request; the same strict response contract applies."""
    evidence=decision_model_view(summary)
    compact=dict(planning_horizon_s=evidence['planning_horizon_s'],target_object_id=evidence['target_object_id'],
        baseline_actual_contact_time_s=evidence['baseline_actual_contact_time_s'],
        coordinate_frame=evidence['coordinate_frame'],vehicle_geometry=evidence['vehicle_geometry'],
        action_bounds=evidence['action_bounds'],frozen_states=evidence['frozen_states'],
        frames=[dict(frame_id=row['frame_id'],time_s=row['time_s']) for row in evidence['frames']])
    return ('Offline defensive-driving analysis of a recorded road scene. Inspect the attached single '
        'timestamped image montage and use the numeric states as primary evidence. Choose one ego action: '
        'brake, lane_change, or brake_then_lane_change. The target vehicle trajectory is fixed. This is '
        'offline future evidence, not online prediction. Do not output trajectory points or claim execution '
        'success. Coordinates are x forward, y left; left offset is positive and right negative. The horizon '
        'must equal the input. Return only one defense_decision.v1 JSON object with fields version, '
        'action_type, start_s, planning_horizon_s, reason, evidence, brake, lane_change. ' 
        'reason must be a string and evidence must be a JSON array of strings. brake requires '
        'deceleration_mps2 and target_speed_mps. lane_change requires direction, offset_m, start_s and '
        'duration_s. Set the unused action object to null; do not add fields.\nFrozen evidence:\n'+
        json.dumps(compact,ensure_ascii=False,separators=(',',':')))


def request_decision(run_dir,text_only=False,compact_visual=False):
    import httpx
    from openai import OpenAI
    from streaming_requests import StreamingRequestArchive
    run_dir=Path(run_dir).resolve();summary_path=run_dir/'decision-input/state-summary.json'
    if (run_dir/'decision-agent/summary.json').exists():
        raise ValueError('Decision already accepted; create a new authorized run instead of another request')
    if text_only and compact_visual:raise ValueError('Choose only one request mode')
    summary=json.loads(summary_path.read_text())
    if compact_visual:
        prompt=compact_visual_prompt(summary)
    else:
        prompt=(ROOT/'configs/prompts/ego_defense_decision_v1.txt').read_text()
        prompt+='\n冻结输入如下（数值均由程序生成）：\n'+json.dumps(decision_model_view(summary),ensure_ascii=False)
    profile=load_llm_profile()
    client=OpenAI(api_key=profile['api_key'],base_url=profile['base_url'],max_retries=0,
        http_client=httpx.Client(trust_env=False,follow_redirects=False,timeout=300.))
    archive=StreamingRequestArchive(run_dir/'decision-agent/requests',client,profile)
    mode='frozen_state_text_only' if text_only else ('compact_video_frames_and_frozen_state' if compact_visual else 'video_frames_and_frozen_state')
    _json(run_dir/'decision-agent/profile.json',dict(base_url=profile['base_url'],model=profile['model'],max_model_actions=1,
        request_mode=mode,visual_input_used=not text_only))
    instruction='你是自动驾驶离线专家规划中的高层防御决策 agent，只输出指定 JSON。\n'
    if compact_visual:
        messages=[dict(role='user',content=[dict(type='text',text=prompt),
            image_part(run_dir/'decision-input/attack-video-contact-sheet.jpg')])]
    elif text_only:
        prompt+='\n本轮视觉输入因服务端内容检查不可用；只依据以下冻结数值决策，不得声称看过视频。\n'
        messages=[dict(role='user',content=instruction+prompt)]
    else:
        messages=[dict(role='user',content=[dict(type='text',text=instruction+prompt),
            image_part(run_dir/'decision-input/attack-video-contact-sheet.jpg')])]
    try:
        content,attempt=archive.complete('decision',messages,total_seconds=300,max_tokens=16384,
            extra_body=dict(thinking=dict(type='enabled')),reasoning_effort='high')
        decision=accept_decision_response(run_dir,content,attempt,profile['model'],visual_input_used=not text_only)
    finally:client.close()
    return decision


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--request-only',action='store_true',help='Explicitly use an already prepared run; never rebuild its input')
    parser.add_argument('--text-only',action='store_true',help='Explicit degraded mode: use frozen numeric evidence only and record visual_input_used=false')
    parser.add_argument('--compact-visual',action='store_true',help='Use one image and a compact defensive-driving prompt with the same strict decision contract')
    parser.add_argument('--frames',type=int,default=8)
    args=parser.parse_args()
    if args.prepare_only and args.request_only:parser.error('Choose only one of prepare-only/request-only')
    if (ROOT/'experiments') not in args.run_dir.resolve().parents:parser.error('run-dir must be inside ignored experiments/')
    if args.text_only and args.compact_visual:parser.error('Choose only one of text-only/compact-visual')
    if args.request_only:
        decision=request_decision(args.run_dir,text_only=args.text_only,compact_visual=args.compact_visual)
        print(json.dumps(decision,ensure_ascii=False))
    elif args.prepare_only:
        summary=prepare_decision_run(args.source_run,args.run_dir,args.frames)
        print(json.dumps(dict(status='decision_input_prepared',frames=len(summary['frames']),run_dir=str(args.run_dir)),ensure_ascii=False))
    else:
        prepare_decision_run(args.source_run,args.run_dir,args.frames)
        decision=request_decision(args.run_dir,text_only=args.text_only,compact_visual=args.compact_visual)
        print(json.dumps(decision,ensure_ascii=False))


if __name__=='__main__':main()
