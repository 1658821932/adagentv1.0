"""One-turn ego-side conflict understanding for a frozen attack scene (nuPlan 2 Hz chain).

Reads the frozen attack scene pack (freeze_attack_scene.py output) and asks the model
for up to three defense directions (e.g. braking, lateral shift, combination) with
evidence and uncertainty. The brief is analysis input for the downstream constructor /
FOCI / trajectory agent stages; it never gates or narrows their search space.

Optionally supplies original recorded camera images for lane understanding. These
images predate the attack edit and must not be described as the edited scene.
Offline counterfactual analysis, not online prediction.
"""
from llm_profile import load_llm_profile
from workflow_timing import timed_operation

import argparse
import acceptance_policy as policy
import csv
import json
import os
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI

from ego_road_checks import load_official_drivable, check_drivable_area_official
from experiment_runtime import ROOT, atomic_json, experiments_root
from run_ego_reference import EGO_FRONT, EGO_REAR, EGO_WIDTH, evaluate, polygon, scene_from_frames
from run_ego_decision_agent import image_part
from streaming_requests import StreamingRequestArchive
from model_context import compact_model_context
from ego_edit_evidence import TOOL_SEMANTICS, model_policy, braking_diagnostic

PROMPT = """你是自车侧冲突理解 agent。输入是一次已冻结的攻击编辑后的离线反事实场景（数值与时间线，不是新相机画面；攻击后的画面尚未渲染，不得声称看过攻击画面）。
任务：分析攻击车相对自车的运动、冲突时间与位置、道路空间和其他有观测对象，提出最多3个值得尝试的防御方向（例如制动、横向避让或组合），每个方向说明依据与不确定性。有几个写几个，不凑数；依据不足可为空。
按几何事实和任务比较有依据的防御方向，无默认动作。不得为了提高进度或净距越过禁止跨越的标线。线型不明时保持unknown，说明需要什么证据，不将unknown直接视为允许或禁止全部横向动作。
max_lateral_speed_mps是速度在车身横向轴的投影（侧滑诊断），不是道路横向移动速度；转弯时车身坐标轴随车头改变。±2米/±0.35弧度仅属于固定锚点偏移工具，不是全部编辑工具的共同可达范围。区分实际执行过检查的候选、解析估算和未验证设想。你不设计轨迹数值，不输出逐点轨迹；不要声称物理碰撞或安全性已成立。其他对象按原记录回放，不会避让。约2Hz采样，采样之间的接触不可排除。
读取actual_start_facts中的后轴参考点、车体尺寸、速度来源和制动估计。停车距离大于可用净距时，不声称纵向制动可停在障碍前；可作为减轻冲突的方向。首次重叠采样时间不是连续接触期限。局部近似不证明所有二维轨迹不可行。只输出JSON：{"directions":[{"rank":1,"type":"brake|lateral_left|lateral_right|combination|other","rationale":"一句话依据","target_window_s":[t0,t1],"uncertainty":"关键未知"}],"insufficient_basis":false}"""

VISUAL_PROMPT = """附带的是原记录的相机图片，仅用于理解道路、车道线和路缘；它不是攻击编辑后的画面，不包含新增车辆。攻击目标及其位置以数值时间线为准，不要声称从图片看到了新增目标。
请分别判断自车左、右侧可见的车道线或道路边界；看不清时明确写 unknown，不要猜测。报告具体图片编号、画面位置、线型和信心，并说明遮挡或适用范围。针对清楚可见的实线、双实线提示不得随意跨越；路缘、人行道不应作为普通绕行空间。图像不能证明候选轨迹是否压线，也不能把静态画面线型推广到看不到的全部路段。
在原JSON中增加字段 lane_observations: [{"side":"left|right","marking":"double_solid|solid|broken|curb|unknown","confidence":"high|medium|low","image_index":0,"evidence":"画面中的位置和证据","uncertainty":"遮挡及范围限制"}]，以及 lane_guidance: 一句用于后续编辑的建议。"""


def validate_brief(brief, end_s):
    if not isinstance(brief, dict) or not isinstance(brief.get('directions'), list) or len(brief['directions']) > 3:
        raise ValueError('Provide up to three directions')
    if type(brief.get('insufficient_basis')) is not bool or (not brief['directions'] and not brief['insufficient_basis']):
        raise ValueError('Empty directions require insufficient_basis')
    ranks = set()
    for item in brief['directions']:
        if not isinstance(item, dict) or type(item.get('rank')) is not int or item['rank'] < 1 or item['rank'] in ranks:
            raise ValueError('Direction ranks must be positive and unique')
        ranks.add(item['rank'])
        if item.get('type') not in {'brake','lateral_left','lateral_right','combination','other'}:
            raise ValueError('Unknown defense direction')
        window = item.get('target_window_s')
        if isinstance(window, list) and len(window) == 2:
            for index in (0, 1):
                if type(window[index]) in (int, float) and abs(window[index]-end_s) <= .001:
                    window[index] = float(end_s)
        if not isinstance(window, list) or len(window) != 2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in window) or not 0 <= window[0] < window[1] <= end_s:
            raise ValueError('Direction window outside scene')
        if any(not isinstance(item.get(k), str) or not item[k].strip() for k in ('rationale','uncertainty')):
            raise ValueError('Direction needs rationale and uncertainty')


def recorded_images(scene_dir, image_scene, frame_index, cameras):
    """Find original camera images and verify their frame identity against the frozen scene."""
    scene_dir, image_scene = Path(scene_dir), Path(image_scene)
    frozen = json.loads((scene_dir / 'metadata/frames.json').read_text())
    recorded = json.loads((image_scene / 'metadata/frames.json').read_text())
    if isinstance(frame_index, bool) or not 0 <= frame_index < len(frozen) or len(frozen) != len(recorded):
        raise ValueError('Image frame must exist in both matching scene timelines')
    if any(a['timestamp'] != b['timestamp'] or a.get('token') != b.get('token')
           for a, b in zip(frozen, recorded)):
        raise ValueError('Image scene timeline does not match the frozen scene')
    if not cameras or len(cameras) > 3 or len(set(cameras)) != len(cameras):
        raise ValueError('Choose one to three distinct camera channels')
    with (image_scene / 'metadata/image_manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = []
    root = (image_scene / 'original_images').resolve()
    for camera in cameras:
        matches = [row for row in rows if int(row['frame_index']) == frame_index and row['camera'] == camera]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one manifest image for {camera} frame {frame_index}')
        row = matches[0]
        if int(row['timestamp']) != int(recorded[frame_index]['timestamp']) or row['token'] != recorded[frame_index]['token']:
            raise ValueError('Image manifest identity does not match the scene frame')
        path = (root / row['data_path']).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f'Image missing or outside image root: {camera}')
        result.append(dict(camera=camera, frame_index=frame_index, timestamp=int(row['timestamp']),
                           path=str(path),
                           source='original_recording_not_attack_render'))
    return result


def build_context(scene_dir, attack_run, candidate_id, target_track, allow_safe_stop=True, target_effect=None):
    scene_dir = Path(scene_dir).resolve()
    frozen = json.loads((scene_dir/'freeze-info.json').read_text())
    if frozen.get('candidate_id') != candidate_id or frozen.get('target_track', frozen.get('track_id')) != target_track:
        raise ValueError('Frozen scene candidate or target does not match conflict inputs')
    if frozen.get('attack_run') and Path(frozen['attack_run']).resolve() != Path(attack_run).resolve():
        raise ValueError('Frozen scene attack source differs')
    frames = json.loads((scene_dir / 'metadata/frames.json').read_text())
    times, ego_xy, ego_yaw, objects = scene_from_frames(frames, target_track)
    baseline = evaluate(times, ego_xy, ego_yaw, objects, times, target_track)
    shift = (EGO_FRONT - EGO_REAR) / 2
    timeline = []
    for i in range(len(times)):
        frame = objects[i]
        target = next(o for o in frame if o['track'] == target_track)
        others = [o for o in frame if o['track'] != target_track]
        other_min = None
        if others:
            ego_poly = polygon(*(ego_xy[i] + shift * np.array([np.cos(ego_yaw[i]), np.sin(ego_yaw[i])])),
                               ego_yaw[i], EGO_FRONT + EGO_REAR, EGO_WIDTH)
            other_min = float(min(ego_poly.distance(polygon(o['x'], o['y'], o['yaw'], o['length'], o['width']))
                                  for o in others))
        timeline.append(dict(frame=i, time_s=round(float(times[i]), 3),
                             target_x_y=[round(float(target['x']), 2), round(float(target['y']), 2)],
                             target_yaw_rad=round(float(target['yaw']), 3),
                             ego_x_y=np.round(ego_xy[i], 2).tolist(),
                             ego_yaw_rad=round(float(ego_yaw[i]), 3),
                             clearance_m=round(baseline['target_clearance_m'][i], 3),
                             nearest_other_m=None if other_min is None else round(other_min, 2)))
    official_path = scene_dir / 'metadata/map_drivable_official.json'
    road = None
    if official_path.is_file():
        official = load_official_drivable(official_path)
        result = check_drivable_area_official(times, ego_xy, ego_yaw, official, min_margin_m=0.1)
        road = dict(recorded_ego_passed=result['passed'], min_margin_m=result.get('min_margin_m'),
                    gate='official polygons, 0.1 m positive margin, per-corner strict contains')
    return dict(
        task='Analyze the frozen attack scene and propose up to three defense directions for the ego.',
        scene=str(scene_dir), attack_run=str(attack_run), attack_candidate=candidate_id,
        target_track=target_track, target_effect=target_effect, tool_semantics=TOOL_SEMANTICS,
        scene_facts=dict(frames=len(times), duration_s=round(float(times[-1]), 3),
                         sample_rate='observed ~2 Hz native timestamps; interpolation between samples is an assumption',
                         min_target_clearance_m=baseline['min_target_clearance_m'],
                         target_overlap_frames=baseline['target_overlap_indices'],
                         recorded_ego_progress_m=baseline['recorded_progress_m'],
                         max_ego_speed_mps=baseline['max_speed_mps']),
        ego_geometry=dict(reference='rear_axle',front_m=EGO_FRONT,rear_m=EGO_REAR,width_m=EGO_WIDTH),
        official_road=road,
        acceptance_policy=model_policy(policy.resolve(dict(allow_safe_stop=allow_safe_stop))),
        defense_constraints='Read acceptance_policy for checks and limits; a null minimum_progress_fraction means progress is diagnostic only.',
        timeline=timeline)


@timed_operation('ego.understanding', 'run_dir')
def run(scene_dir, attack_run, candidate_id, target_track, run_dir, timeout_seconds=300, max_tokens=8192,
        allow_safe_stop=True, image_scene=None, visual_frame=0, cameras=None, acceptance=None, target_effect=None):
    run_dir = Path(run_dir).resolve()
    if (experiments_root()) not in run_dir.parents or run_dir.exists():
        raise ValueError('Run must be a new directory under experiments')
    images = recorded_images(scene_dir, image_scene, visual_frame, cameras or ['CAM_F0']) if image_scene else []
    run_dir.mkdir(parents=True)
    context = build_context(scene_dir, attack_run, candidate_id, target_track, allow_safe_stop, target_effect)
    if acceptance is not None:
        context['acceptance_policy'] = model_policy(policy.resolve(acceptance))
        context['defense_constraints'] = 'Use acceptance_policy as the frozen experiment settings. First observation_end_s seconds are immutable; progress is diagnostic when minimum_progress_fraction is null. Road/vehicle capability not provided remains unknown. Do not invent a progress gate or an earlier editing start.'
    context['acceptance_policy'] = model_policy(policy.resolve(context['acceptance_policy']))
    actual_frames = json.loads((Path(scene_dir)/'metadata/frames.json').read_text())
    actual_times,actual_xy,actual_yaw,actual_actors = scene_from_frames(actual_frames,target_track)
    context['actual_start_facts'] = braking_diagnostic(actual_times,actual_xy,actual_yaw,
        actual_actors,target_track,context['acceptance_policy'])
    context['defense_constraints'] = ('Use only acceptance_policy for required checks and progress rules. '
        'Geometry facts are reference evidence, not a global feasibility proof. '
        'Directions are advisory; distinguish avoidance, overlap reduction and missing evidence.')
    atomic_json(run_dir / 'config.json', dict(
        version='ego_conflict_brief.v3', scene_dir=str(Path(scene_dir).resolve()),
        attack_run=str(Path(attack_run).resolve()), attack_candidate=candidate_id,
        target_track=target_track, timeout_seconds=timeout_seconds, max_tokens=max_tokens,
        allow_safe_stop=allow_safe_stop, acceptance_policy=context['acceptance_policy'],
        visual_input=images,
        role='analysis input only; downstream constructor/FOCI/agent search is not gated by this brief'))
    atomic_json(run_dir / 'context.json', context)
    profile = load_llm_profile()
    client = OpenAI(api_key=profile['api_key'], base_url=profile['base_url'], max_retries=0,
                    http_client=httpx.Client(trust_env=False, follow_redirects=False,
                                             timeout=float(timeout_seconds)))
    archive = StreamingRequestArchive(run_dir / 'requests', client, profile)
    prompt = PROMPT + ('\n' + VISUAL_PROMPT if images else '') + '\nScene state:\n' + json.dumps(compact_model_context(context), ensure_ascii=False, separators=(',', ':'))
    content = prompt
    if images:
        content = [{'type': 'text', 'text': prompt}]
        for index, item in enumerate(images):
            content.extend([{'type': 'text', 'text': f"Image {index}: {item['camera']}, recorded frame {item['frame_index']}"}, image_part(item['path'])])
    messages = [{'role': 'user', 'content': content}]
    status = 'completed'
    brief = None
    response_received = False
    try:
        _, attempt = archive.complete('turn_01', messages, total_seconds=timeout_seconds,
                                      max_tokens=max_tokens,
                                      extra_body={'thinking': {'type': 'enabled'}},
                                      reasoning_effort='high')
        response_received = True
        message = json.loads((attempt / 'response.json').read_text())['choices'][0]['message']
        text = message.get('content') or ''
        start = text.find('{')
        if start < 0:
            status = 'no_json_in_response'
        else:
            # balanced-brace extraction tolerates ```json fences and trailing prose
            depth, end, in_string, escape = 0, -1, False, False
            for pos in range(start, len(text)):
                ch = text[pos]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = pos
                        break
            if end < 0:
                raise ValueError('unbalanced JSON in response')
            brief = json.loads(text[start:end + 1])
            validate_brief(brief, context['scene_facts']['duration_s'])
            if images:
                if not isinstance(brief.get('lane_observations'), list) or not isinstance(brief.get('lane_guidance'), str):
                    raise ValueError('Visual response requires lane_observations and lane_guidance')
                for observation in brief['lane_observations']:
                    if not isinstance(observation, dict) or observation.get('side') not in {'left', 'right'} \
                            or observation.get('marking') not in {'double_solid', 'solid', 'broken', 'curb', 'unknown'} \
                            or observation.get('confidence') not in {'high', 'medium', 'low'} \
                            or type(observation.get('image_index')) is not int \
                            or not 0 <= observation['image_index'] < len(images) \
                            or not isinstance(observation.get('evidence'), str) \
                            or not isinstance(observation.get('uncertainty'), str):
                        raise ValueError('Invalid visual lane observation')
                if {item['side'] for item in brief['lane_observations']} != {'left', 'right'}:
                    raise ValueError('Visual response must describe both sides, using unknown where needed')
                brief['visual_evidence_scope'] = 'Original recorded images, not the edited attack scene; no geometric lane-crossing verification.'
            atomic_json(run_dir / 'conflict-brief.json', brief)
    except Exception as exc:  # record failure honestly, never fake a brief
        brief = None
        status = f'{type(exc).__name__}: {exc}'
    atomic_json(run_dir / 'summary.json', dict(
        status=status, brief_written=brief is not None,
        directions=None if brief is None else len(brief['directions']),
        visual_input_used=bool(images and response_received), visual_images=images, physical_execution='not_run'))
    print(json.dumps(dict(status=status, directions=None if brief is None else len(brief['directions']))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scene', type=Path, required=True, help='frozen attack scene pack')
    parser.add_argument('--attack-run', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--target-track', required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=300)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--allow-stop', action=argparse.BooleanOptionalAction, default=True,
                        help='Allow a safe stop instead of 50% progress (default: enabled)')
    parser.add_argument('--image-scene', type=Path, help='Original recorded scene containing the image manifest')
    parser.add_argument('--visual-frame', type=int, default=0)
    parser.add_argument('--camera', action='append', dest='cameras', help='Recorded camera channel; repeat up to three times')
    args = parser.parse_args()
    run(args.scene, args.attack_run, args.candidate, args.target_track, args.run,
        timeout_seconds=args.timeout_seconds, max_tokens=args.max_tokens, allow_safe_stop=args.allow_stop,
        image_scene=args.image_scene, visual_frame=args.visual_frame, cameras=args.cameras)


if __name__ == '__main__':
    main()
