"""Two-stage attack editing: select/create an actor, then edit a generic trajectory."""
from llm_profile import load_llm_profile
import acceptance_policy as policy
from workflow_timing import timed_operation

import argparse
import copy
import json
import os
from pathlib import Path

import httpx
import numpy as np

from openai import OpenAI
from shapely.geometry import LineString

from attack_tool_contract import constructor_exhaustion, trajectory_tools
from model_context import compact_model_context
from attack_editor import AttackEditor, read_seed, write_trajectory
from ego_road_checks import check_drivable_area_official, load_official_drivable
from experiment_runtime import ROOT, atomic_json, experiments_root
from run_ego_decision_agent import image_part
from run_ego_editor_agent import TOOLS as EGO_TOOLS
from ego_edit_contract import normalize_edit_call
from run_ego_reference import (EGO_FRONT, EGO_REAR, EGO_WIDTH, ego_path_from_frames,
                               observed_objects_from_frames, polygon)
from streaming_requests import StreamingRequestArchive
from vehicle_trajectory import actor_observation_series, actor_pose_at_time, sample_xy
from vehicle_motion import stopping_distance


def function_tool(name, description, properties, required=None):
    return {'type':'function', 'function':dict(name=name, description=description,
        parameters=dict(type='object', properties=properties,
                        required=list(properties) if required is None else required,
                        additionalProperties=False))}


NUMBER = {'type':'number'}
REASON = {'type':'string'}
OBJECT_TOOLS = [
    function_tool('finish_search','Decline object selection with measured reasons when no suitable object or placement can be selected within the current tools and budget.',{'reason':REASON}),
    function_tool('select_vehicle', 'Choose one existing fully observed vehicle to edit.',
                  {'track_id':{'type':'string'}, 'reason':REASON}),
    function_tool('add_vehicle', 'Create a vehicle identity and straight initial-motion seed. Subsequent generic tools determine its final path.',
                  {'asset_id':{'type':'string'}, 'x_m':NUMBER, 'y_m':NUMBER,
                   'yaw_rad':NUMBER, 'speed_mps':NUMBER, 'reason':REASON}),
]
POINT = dict(type='object', properties={'time_s':NUMBER, 'x_m':NUMBER, 'y_m':NUMBER},
             required=['time_s', 'x_m', 'y_m'], additionalProperties=False)
TRAJECTORY_TOOLS = [function_tool('construct_trajectory',
    'Construct one target-center trajectory from 2..16 absolute scene-local control points. '
    'End at the final scene time. Start at t=0 with the seed position, or at/after the observation boundary; '
    'the seed initial point is then inserted automatically. The tool locks the seed '
    'through the observation boundary and inserts its exact boundary point. Supplied prefix '
    'points may differ by at most 2cm for rounding, then are replaced by the seed. C2 interpolation reuses '
    'the existing displacement constructor. No maneuver template is selected.',
    {'points':dict(type='array', items=POINT, minItems=2, maxItems=16),
     'end_velocity_mps':dict(type='array', items=NUMBER, minItems=2, maxItems=2), 'reason':REASON})]
INTERVAL = dict(type='array', items=NUMBER, minItems=2, maxItems=2)
KEYFRAME = dict(type='object', properties={
    'time_window_s':INTERVAL, 'hold_duration_s':NUMBER,
    'relative_position':dict(type='object', properties={'forward_m':INTERVAL, 'left_m':INTERVAL}, additionalProperties=False),
    'heading_difference_rad':INTERVAL, 'speed_mps':INTERVAL},
    required=['time_window_s', 'hold_duration_s'], additionalProperties=False)
TRAJECTORY_TOOLS.insert(0, function_tool('construct_attack_candidates',
    'Optimize a parent acceleration/steering control sequence with continuous official-road and observed-obstacle guidance. '
    'Declare body-center position intervals in the recorded ego BODY-center frame, heading and/or speed keyframes. '
    'Supports stationary/slow lead, merging, crossing or opposite-heading objectives without maneuver templates. '
    'New first calls use seed_000. Further calls truly start from parent controls. '
    'Optional edit_window_s limits changed control knots; preserve_outside_window additionally requires reconnection '
    '(outside position<=0.05m, yaw<=0.05rad, velocity<=0.1m/s); failure is reported, never spliced away. '
    'Near miss stays contact-free, >=0.15m clearance, <=2m event gap; contact requires first sampled overlap in window. '
    'Either accepts one of these two events within the window, never absence of interaction. '
    'reason is explanatory only: all numeric requirements must appear in objective. '
    'Returns fully checked candidates separately from optimization guidance. Preserve the first 3s and allow room/time for ego avoidance.',
    {'parent_candidate_id':{'type':'string'},
     'objective':dict(type='object', properties={
         'interaction':dict(type='string', enum=['near_miss','contact','either']),
         'interaction_window_s':INTERVAL,
         'keyframes':dict(type='array', items=KEYFRAME, minItems=0, maxItems=6)},
         required=['interaction'],additionalProperties=False),
     'edit_window_s':INTERVAL, 'preserve_outside_window':{'type':'boolean'},
     'max_candidates':dict(type='integer',minimum=1,maximum=3), 'reason':REASON},
    required=['parent_candidate_id','objective','max_candidates','reason']))
TRAJECTORY_TOOLS += [copy.deepcopy(tool) for tool in EGO_TOOLS if tool['function']['name'] in {
    'edit_speed_profile', 'edit_route', 'repair_candidate', 'submit_candidate', 'finish_search'}]
for tool in TRAJECTORY_TOOLS:
    if tool['function']['name']=='edit_speed_profile':
        tool['function']['description']+=' In this attack stage the unchanged bicycle policy rejects reverse travel. Generator support for signed speeds does not override that policy.'
TRAJECTORY_TOOLS += [function_tool('inspect_trajectory', 'View a candidate BEV and reference checks before submitting it.',
                                  {'candidate_id':{'type':'string'}})]

OBJECT_PROMPT = """Select the object for the requested scene edit. Obey the user's instruction: use an
existing vehicle when requested, or add a vehicle when requested. You choose between the available
tools; no stationary/cut-in mode is preselected. The next stage receives your object and edits its
trajectory using generic tools. For a new object choose a plausible initial center, yaw and speed;
this straight-motion seed must remain unchanged during the first 3 seconds. Choose it so it has
space on the official road and does not hit ego or observed others during those 3 seconds.
Coordinates are metres in the initial ego frame: x forward, y left, yaw counterclockwise radians.
Consider the full ego timeline and stopped traffic before selecting the initial state. A safe initial
3 seconds alone does not guarantee room to merge later. Use vehicle_planning_hints to compare
stopping/turning space with observed front traffic; choose a lower speed or smaller side offset
when space is tight. Avoid choosing a speed and side offset that
would require a late backward correction. If no suitable object or placement can be selected within current tools/budget, finish_search with measured reasons. Provide one native tool call per turn."""

EDIT_PROMPT = """If the selected object/seed leaves no feasible approach, request_object_reselection once with measured reasons. The controller creates an explicit new branch and carries spent search budget.
Only currently available tools may be called. If the cumulative constructor evaluation budget is exhausted and no candidate passes trajectory checks plus fixed task requirements, the controller ends attack search with a measured refusal. A candidate awaiting PDM remains available for checking and submission.
When constraints.task_contract is present it contains explicit task checks; read its provenance.
Do not invent user requirements from planning preferences. An unspecified window covers all time after
observation, and keyframes are optional. For stationary_target there are no trajectory degrees of freedom:
evaluate seed_000, inspect and submit if passing, or request a new placement branch. Never call a motion
constructor for a stationary target. Measured geometry overrides your mental estimate of contact time.
Road coverage does not certify lane identity; report lane evidence unknown unless verified.
Your objective keyframes are planning preferences only. Changing preferences never changes task acceptance.
Read construction_goal.task_contract evidence separately from preferences. Physical/map checks stay hard.
Use one or two broad keyframes and keep a useful parent's controls. Do not move a lead into another lane
or invent an ego overtake to satisfy near_miss. Contact and near_miss may both satisfy an 'either' contract.
When reference requirements pass, perform the configured PDM self-check if enabled, then inspect and submit; do not perfect optional intervals.
Choose the next tool from measured failures. Let the numerical constructor solve controls; avoid lengthy
manual trajectory calculations. Explain the next action briefly.
Edit only the selected target; ego and other actors replay recorded observations.
Do not confuse body center distance with bumper clearance. An aligned near_miss needs forward center
distance at least half the sum of body lengths plus clearance. If recorded ego later drives through a
stationary lead, near_miss is impossible without moving that lead out; contact mode can represent
this intended attack while later ego defense is a separate stage. Do not invent evasive motion merely
to satisfy near_miss when the instruction requests a stationary lead. Prefer one or two broad,
realistically achievable keyframes first; narrow multi-stage objectives make optimization harder.
For moving targets, first call construct_attack_candidates. Declare structured keyframes (relative body-center forward/left,
heading difference radians, speed m/s), each inside a post-observation time window with a hold duration.
Use goals appropriate to the instruction: a stationary lead needs speed~0 and forward relation, not a
side-entry predicate; crossing needs positions on both sides, opposite approach needs heading near pi.
All first 3 seconds are immutable. Three seconds is the edit start, not a requirement to crash immediately.
Timing is allowed strictly after 3 seconds. Do not invent a reaction delay or a narrower hard window. Contact is optional unless
explicitly required; near_miss must remain contact-free. Do not replace a requested behavior with mere proximity.
The constructor starts from parent controls, uses road/observed obstacles during numerical optimization,
and strictly checks finalists. On refinement keep the same objective unless deliberately revising your plan.
Set edit_window_s for local control changes. preserve_outside_window requires reconnection within stated
tolerances; a failed reconnection needs an explicitly larger window or a rejected edit. Read local_preservation.
reason is explanatory: only structured objective fields guide optimization. Body-relative objective positions
use EGO BODY center, while absolute control points use initial-ego scene coordinates.
Read separate reference_screen_passed and construction_goal, keyframe actual values, diagnostic measurements
and suggestions. A fallback is not completion. objective_cost is a mixed-unit ranking cost, not metres.
Use edit_speed_profile for timed signed speed targets and edit_route for smooth lateral offsets and return.
These edits preserve the chosen parent through start_time_s, including the first 3s. edit_route computes
heading/curvature; its default preserve_speed spends some forward progress on lateral motion, while
preserve_progress can increase total speed. A return removes the lateral offset, not necessarily same-time
position lag. Inspect actual continuous acceleration/normal component and inherited task-goal checks.
No implicit route extension, retracking or changed objective. Preview does not create a candidate.
Read decision_summary before raw edit evidence. recommended_previews contain complete conditional arguments to preview, not automatic goal changes; create_candidate_call repeats a passed preview with preview_only=false. When a recommendation matches the task and budget remains, preview it before deciding the modified behavior is unsuitable; selecting another known route still rechecks all geometry and task requirements.
A stationary car cannot move sideways or rotate in place; prepare a feasible moving parent first. Inspect and receive a promising candidate image before submitting. Submit an
existing passing candidate without unnecessary edits. Stop honestly with finish_search if the tools cannot
meet the objective. Use one native tool call per turn; reference geometry is not physical execution."""



STATIC_EDIT_PROMPT = """Validate the selected stationary placement. Only the selected target is added or
edited; ego and observed others replay their records. The target pose is fixed for the entire scene.
Read seed_000's reference checks, fixed task contract and geometry_feedback. Use measured body overlap
times, never mental center-arrival estimates. The observation interval includes t=3; positive-area overlap
is allowed only strictly after it. Do not invent a narrower hard window, hold duration or reaction delay.
Only explicit task requirements and named checking policy are hard; read provenance and uncertainties.
If pdm_self_check is enabled, passing geometry only makes the placement eligible for probe_pdm.
Test first; a safely defended placement needs reselection or an honest finish. A tested PDM-failing
placement must still use inspect_trajectory and receive the image before submit_candidate(seed_000).
When self-check is disabled, passing geometry plus trajectory inspection permits submission.
No motion construction, point editing or repair is available or needed. evaluate_candidate reads existing
measurements without optimization. A failed placement can request_object_reselection once, creating a
new scene branch with a new checked prefix; otherwise finish_search with measured reasons.
Road coverage does not establish lane identity. Check the BEV against the instruction; if lane evidence
remains unknown, preserve that uncertainty in your submission reason. Geometry is not physical execution.
Use one tool call per turn, and do not optimize optional scores after the fixed requirements pass."""


PDM_PROBE_PROMPT = '''PDM self-check is enabled for this run. Reference geometry acceptance is only eligibility
for testing, not attack completion. Before submitting, call probe_pdm for that exact candidate and
read the actual edited-scene PDM trajectory and supplied overlay. Original-scene PDM is not required. Recorded ego and PDM may follow
different paths; never infer an offset or evasive maneuver from a scalar clearance alone.
Only pdm_self_check.candidates contains tests of the current branch. revision_feedback and
pdm_reselection_feedback describe previous versions, even when their local candidate IDs match.
Run the current eligible candidate before treating an earlier safe outcome as its result.
When PDM defends safely, revise a moving candidate or request_object_reselection for a stationary
placement within the remaining budgets; keep the original instruction and every road/history check.
Do not satisfy the PDM challenge by violating same-lane requirements. Do not invent a narrower
contact window or clearance goal. If no tested PDM-failing candidate is found, finish honestly.
A passing PDM challenge still requires trajectory inspection before submission. Simulation/service
errors are execution failures, not proof of an extreme attack.'''

DEFENSE_FEEDBACK_PROMPT = '''Defense feedback is enabled. The final loop needs both PDM failure and an
accepted defense, not maximal immediacy or impact. probe_pdm returns defense_feasibility separately
from PDM. Read its measured edit-start clearance, braking estimate and bounded constructor results.
Prefer preserving a tested candidate with a passing reference defense witness while improving attack
challenge. If PDM fails but no witness was found, consider a local timing/position/control adjustment
that gives defense more opportunity, then test again; a missing witness is not global infeasibility.
A straight-line braking estimate is advisory, not a rule excluding steering alternatives. Never invent
minimum reaction time, a narrower collision window or a new minimum clearance. Unavailable diagnostic
or failed API requests are not evidence that defense is impossible. Static targets must remain static;
request position reselection as needed within the existing shared action/probe budgets. Final defense
still requires its own Agent submission and independent checks.'''

TRAJECTORY_TOOLS.append(function_tool('probe_pdm',
    'Test an eligible candidate with real PDM + LQR/kinematic closed loop. Returns actual trajectory '
    'samples, closest approach and an overlay image. Shared simulation budget; repeats reuse the same candidate result.',
    {'candidate_id':{'type':'string'}}))

TRAJECTORY_TOOLS.append(function_tool('evaluate_candidate',
    'Return measured full-horizon reference geometry and task checks for an existing candidate. No optimization.',
    {'candidate_id':{'type':'string'}}))
TRAJECTORY_TOOLS.append(function_tool('request_object_reselection', 'Request one bounded return to object selection when geometry fails, PDM defends safely, or PDM fails but bounded defense feedback found no witness; retain spent PDM/search budgets.', {'reason':REASON}))

def decode_call(attempt):
    response = json.loads((attempt / 'response.json').read_text())
    calls = response['choices'][0]['message'].get('tool_calls') or []
    if len(calls) != 1:
        raise ValueError('Exactly one native tool call is required')
    call = calls[0]
    arguments = json.loads(call['function']['arguments'])
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be an object')
    return call, call['function']['name'], arguments


def archive_exchange(folder, turn, call, result):
    path = folder / 'tool_calls' / f'turn_{turn:02d}'
    atomic_json(path / 'call.json', call)
    atomic_json(path / 'result.json', result)


def _road_sections(times, ego, yaw, road):
    sections = []
    for index in sorted(set(np.linspace(0, len(times) - 1, min(6, len(times)), dtype=int))):
        normal = np.array([-np.sin(yaw[index]), np.cos(yaw[index])])
        line = LineString([ego[index] - normal * 15, ego[index] + normal * 15])
        intersection = road.union.intersection(line)
        pieces = list(getattr(intersection, 'geoms', [intersection]))
        intervals = []
        for piece in pieces:
            if piece.geom_type == 'LineString' and not piece.is_empty:
                offsets = (np.asarray(piece.coords) - ego[index]) @ normal
                intervals.append([round(float(offsets.min()), 3), round(float(offsets.max()), 3)])
        sections.append(dict(time_s=float(times[index]), ego_xy_m=ego[index].tolist(),
                             road_lateral_intervals_m=intervals))
    return sections


def check_initial_relation(times,seed,angles,ego,ego_yaw,size,requirement):
    if requirement is None:
        return None
    time=float(requirement['time_s'])
    position=sample_xy(times,seed,[time])[0]
    origin=sample_xy(times,ego,[time])[0]
    heading=float(np.interp(time,times,ego_yaw))
    angle=float(np.interp(time,times,angles))-heading
    relative=position-origin
    forward=float(relative @ np.array([np.cos(heading),np.sin(heading)]))-(EGO_FRONT-EGO_REAR)/2
    left=float(relative @ np.array([-np.sin(heading),np.cos(heading)]))
    half_extent=abs(np.sin(angle))*size[0]/2+abs(np.cos(angle))*size[1]/2
    clearance=abs(left)-EGO_WIDTH/2-half_extent
    side_ok=left>0 if requirement['side']=='left' else left<0
    lo,hi=requirement['forward_to_ego_center_interval_m']
    gap_lo,gap_hi=requirement['side_clearance_interval_m']
    if not(side_ok and lo<=forward<=hi and gap_lo<=clearance<=gap_hi):
        raise ValueError(f'Initial boundary relation invalid: forward_to_ego_center={forward:.3f}m, side_clearance={clearance:.3f}m; required {requirement}')
    return dict(time_s=time,forward_to_ego_center_m=forward,side_clearance_m=clearance,left_m=left)


@timed_operation('attack.object_selection', 'folder')
def select_object(scene, folder, instruction, assets, constraints, archive,
                  timeout_seconds, max_tokens, required_operation=None, required_track=None):
    frames = json.loads((scene / 'metadata/frames.json').read_text())
    times, ego, yaw = ego_path_from_frames(frames)
    actors = observed_objects_from_frames(frames)
    road = load_official_drivable(scene / 'metadata/map_drivable_official.json')
    common = set(actor['track'] for actor in actors[0] if actor['label'] == 'vehicle')
    for frame in actors:
        common &= {actor['track'] for actor in frame if actor['label'] == 'vehicle'}
    catalog = []
    for track in sorted(common):
        sequence = [next(actor for actor in frame if actor['track'] == track) for frame in actors]
        catalog.append(dict(track_id=track,
            poses=[dict(time_s=float(times[index]), xy_m=[sequence[index]['x'], sequence[index]['y']],
                        yaw_rad=sequence[index]['yaw'])
                   for index in sorted(set([0, len(times) // 2, len(times) - 1]))]))
    ego_speeds = np.linalg.norm(np.diff(ego, axis=0), axis=1) / np.diff(times)
    context = dict(instruction=instruction, scene=str(scene), assets=assets,
        existing_vehicles=catalog, required_operation=required_operation, required_track=required_track,
        constraints=constraints,
        ego_reference_point='rear axle',ego_body_center_shift_m=(EGO_FRONT-EGO_REAR)/2,
        ego_body_dimensions_m=dict(length=EGO_FRONT+EGO_REAR,width=EGO_WIDTH,front_from_rear_axle=EGO_FRONT,rear_from_rear_axle=EGO_REAR),
        ego_timeline=[dict(time_s=float(time), xy_m=position.tolist(), yaw_rad=float(angle),
                          speed_to_next_mps=None if index == len(times) - 1 else float(ego_speeds[index]))
                      for index, (time, position, angle) in enumerate(zip(times, ego, yaw))],
        vehicle_planning_hints=([
            dict(asset_id=asset['asset_id'],wheelbase_assumption_m=.6*asset['size_m'][0],
                min_rear_axle_turn_radius_m=.6*asset['size_m'][0]/np.tan(.55),
                stopping_distance_by_initial_speed_m={str(speed):round(stopping_distance(speed),2)
                    for speed in (2.,3.,4.,5.,6.,7.)},
                note='Model estimates with immediate braking, no reaction delay; not a feasibility certificate')
            for asset in assets] if constraints.get('vehicle_motion_model')=='bicycle.v1' else []),
        road_sections=_road_sections(times, ego, yaw, road),
        actors_at_start=actors[0],
        role='Select an actor and initial state; trajectory behavior is decided in the next stage')
    folder.mkdir(parents=True, exist_ok=False)
    atomic_json(folder / 'input.json', context)
    atomic_json(folder / 'tools.json', OBJECT_TOOLS)
    feedback = None
    for turn in range(1, 5):
        prompt=OBJECT_PROMPT + '\n' + json.dumps(
            compact_model_context(dict(context=context, last_tool_result=feedback)), ensure_ascii=False, separators=(',', ':'))
        if constraints.get('pdm_defense_feedback'):
            prompt=DEFENSE_FEEDBACK_PROMPT+'\n'+prompt
        review=constraints.get('pdm_reselection_feedback')
        if review is None:
            evidence=constraints.get('revision_feedback',{}).get('evidence',{})
            if evidence.get('overlay_path'):
                review=evidence
        content=prompt if not review else [dict(type='text',text=prompt),image_part(review['overlay_path'])]
        messages = [{'role':'user', 'content':content}]
        print('Object selection turn', turn, flush=True)
        _, attempt = archive.complete(f'object_{turn:02d}', messages, total_seconds=timeout_seconds,
            max_tokens=max_tokens, extra_body={'thinking':{'type':'enabled'}}, reasoning_effort='high',
            tools=OBJECT_TOOLS, tool_choice='required', parallel_tool_calls=False, allow_tool_calls=True)
        call, name, args = decode_call(attempt)
        try:
            if not isinstance(args.get('reason'), str) or not args['reason'].strip():
                raise ValueError('A nonempty selection reason is required')
            if name == 'finish_search' and set(args) == {'reason'}:
                result=dict(status='not_submitted',stop_reason='finished_without_object_selection',
                            reason=args['reason'],search_stage='object_selection')
                atomic_json(folder/'output.json',result)
                archive_exchange(folder,turn,call,dict(status='finished',reason=args['reason']))
                return result
            if name == 'select_vehicle':
                if set(args) != {'track_id', 'reason'} or args['track_id'] not in common:
                    raise ValueError('Choose a fully observed vehicle from the catalog')
                if required_operation == 'add_vehicle' or (required_track and args['track_id'] != required_track):
                    raise ValueError('Object selection conflicts with the user-specified object requirement')
                track = args['track_id']
                sequence = [next(actor for actor in frame if actor['track'] == track) for frame in actors]
                seed = np.array([[actor['x'], actor['y']] for actor in sequence])
                angles = np.unwrap([actor['yaw'] for actor in sequence])
                raw_index = frames[0]['anns']['track_tokens'].index(track)
                size = frames[0]['anns']['gt_boxes'][raw_index][3:6]
                target = dict(operation='edit_existing', track_id=track, size_m=size, asset_path=None)
            elif name == 'add_vehicle':
                expected = {'asset_id', 'x_m', 'y_m', 'yaw_rad', 'speed_mps', 'reason'}
                if set(args) != expected or required_operation == 'edit_existing' or required_track is not None:
                    raise ValueError('Invalid add_vehicle selection arguments')
                asset = next(item for item in assets if item['asset_id'] == args['asset_id'])
                numbers = [args[key] for key in ('x_m', 'y_m', 'yaw_rad', 'speed_mps')]
                if any(type(value) not in (int, float) for value in numbers) or not np.isfinite(numbers).all() or not 0 <= args['speed_mps'] <= 12:
                    raise ValueError('New vehicle requires finite pose and speed in [0,12] m/s')
                track_ids = {actor['track'] for frame in actors for actor in frame}
                track_index = 1
                while f'synthetic_vehicle_{track_index:03d}' in track_ids:
                    track_index += 1
                track = f'synthetic_vehicle_{track_index:03d}'
                if constraints.get('stationary_target') and args['speed_mps'] != 0:
                    raise ValueError('User requires a stationary added vehicle: initial speed must be zero')
                direction = np.array([np.cos(args['yaw_rad']), np.sin(args['yaw_rad'])])
                seed = np.array([args['x_m'], args['y_m']]) + (times - times[0])[:, None] * args['speed_mps'] * direction
                angles = np.full(len(times), args['yaw_rad'])
                size = asset['size_m']
                target = dict(operation='add_vehicle', track_id=track, size_m=size,
                              asset_path=asset['path'], initial_speed_mps=args['speed_mps'],
                              initial_state=dict(x_m=args['x_m'], y_m=args['y_m'], yaw_rad=args['yaw_rad']))
                initial_road = check_drivable_area_official(times[:1], seed[:1], angles[:1], road,
                    front=size[0] / 2, rear=size[0] / 2, width=size[1],
                    min_margin_m=constraints['official_margin_m'])
                if not initial_road['passed']:
                    raise ValueError('New vehicle initial footprint is outside the official road margin')
            else:
                raise ValueError('Unknown object-selection tool')
            # Validate the immutable observation prefix before accepting the object.
            prefix_times = np.linspace(times[0], constraints['observation_end_s'], 31)
            prefix_xy = sample_xy(times, seed, prefix_times)
            prefix_yaw = np.interp(prefix_times, times, angles)
            prefix_road = check_drivable_area_official(prefix_times, prefix_xy, prefix_yaw, road,
                front=size[0] / 2, rear=size[0] / 2, width=size[1], min_margin_m=constraints['official_margin_m'])
            if not prefix_road['passed']:
                raise ValueError('Selected straight/recorded seed leaves the road during the immutable first 3 seconds')
            ego_prefix = sample_xy(times, ego, prefix_times)
            ego_angles = np.interp(prefix_times, times, yaw)
            series = actor_observation_series(times, actors, track)
            for time, position, angle, ego_position, ego_angle in zip(prefix_times, prefix_xy, prefix_yaw, ego_prefix, ego_angles):
                body = polygon(*position, angle, *size[:2])
                center = ego_position + (EGO_FRONT - EGO_REAR) / 2 * np.array([np.cos(ego_angle), np.sin(ego_angle)])
                if body.intersection(polygon(*center, ego_angle, EGO_FRONT + EGO_REAR, EGO_WIDTH)).area > 1e-9:
                    raise ValueError(f'Selected seed contacts ego during observation at {time:.2f}s')
                for other_track, observed, poses in series:
                    pose = actor_pose_at_time(observed, poses, time, 1.1 * np.median(np.diff(times)), scene_times=times)
                    if pose is not None and body.intersection(polygon(*pose[:2], pose[2], pose[3], pose[4])).area > 1e-9:
                        raise ValueError(f'Selected seed contacts observed actor {other_track} during observation at {time:.2f}s')
            if constraints.get('vehicle_motion_model') == 'bicycle.v1' and target['operation']=='add_vehicle':
                from road_height import RoadHeightMap,read_recon_translation
                surface=RoadHeightMap.load(scene)
                recon=read_recon_translation(scene)
                origin=np.asarray(frames[0]['ego2global'],float)
                yaw0=np.arctan2(origin[1,0],origin[0,0])
                rotation=np.array([[np.cos(yaw0),-np.sin(yaw0)],[np.sin(yaw0),np.cos(yaw0)]])
                support_xy = prefix_xy[:1] if constraints.get('stationary_target') else prefix_xy
                support_yaw = prefix_yaw[:1] if constraints.get('stationary_target') else prefix_yaw
                for position,angle in zip(support_xy@rotation.T+origin[:2,3]-recon[:2],support_yaw):
                    plane=surface.placement_plane(*position,*size[:2],yaw0+angle)
                    if plane['max_corner_residual_m']>.1:
                        raise ValueError('Initial prefix has non-planar road support; choose another initial state')
            initial_relation=check_initial_relation(times,seed,angles,ego,yaw,size,constraints.get('initial_relation'))
            seed_path = folder / 'placements' / f'attempt_{turn:02d}' / 'seed_trajectory.csv'
            write_trajectory(seed_path, times, seed, angles, size)
            selection = dict(schema_version='attack_object.v1', status='completed', scene=str(scene),
                instruction=instruction, target=target, seed_trajectory_path=str(seed_path),
                selection_reason=args['reason'],initial_relation=initial_relation, annotation_changes=dict(operation='none',
                    note='Identity and motion seed only; full scene is materialized after trajectory acceptance'))
            forecast_editor = AttackEditor(scene, selection, folder / f'placement_check_{turn:02d}', constraints)
            selection['geometry_feedback'] = forecast_editor.public('seed_000')
            atomic_json(folder / 'output.json', selection)
            feedback = dict(status='selected', target=target, geometry_feedback=selection['geometry_feedback'], output_path=str(folder / 'output.json'))
            archive_exchange(folder, turn, call, feedback)
            return selection
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            feedback = dict(status='invalid', error=str(error)[:350])
            archive_exchange(folder, turn, call, feedback)
    result=dict(status='not_submitted',stop_reason='object_selection_budget_exhausted',
                search_stage='object_selection',reason='No valid object selection or initial placement was found within the object-selection budget.')
    atomic_json(folder/'output.json',result)
    return result


@timed_operation('attack.agent_loop', None)
def edit_object(editor, selection, archive, timeout_seconds, max_tokens, max_turns):
    folder = editor.run_dir
    atomic_json(folder / 'input.json', dict(selection=selection, constraints=editor.constraints))
    if editor.constraints.get('pdm_self_check'):
        from attack_pdm_probe import PDMProbe
        editor.pdm_probe=PDMProbe(editor)
    available_tools = [tool for tool in TRAJECTORY_TOOLS if not (
        (editor.constraints.get('require_structured_objective') or editor.constraints.get('task_contract')) and tool['function']['name']=='construct_trajectory')]
    if editor.constraints.get('stationary_target'):
        allowed = {'evaluate_candidate', 'inspect_trajectory', 'submit_candidate',
                   'request_object_reselection', 'finish_search', 'probe_pdm'}
        available_tools = [tool for tool in available_tools if tool['function']['name'] in allowed]
    if editor.pdm_probe is None:
        available_tools=[tool for tool in available_tools if tool['function']['name']!='probe_pdm']
    atomic_json(folder / 'tools.json', available_tools)
    feedback = editor.public(editor.history[-1]['candidate_id']) if editor.history else editor.public('seed_000')
    pending = None
    imported_count = len(editor.history)
    deliveries = []
    evaluated_candidates = set()
    stop = 'action_budget_exhausted'
    termination = None
    try:
        for turn in range(1, max_turns + 1):
            context = dict(instruction=selection['instruction'], **editor.context(),
                           last_tool_result=feedback, turns_remaining=max_turns - turn + 1)
            termination = constructor_exhaustion(editor.nodes,context['constructor_budget'])
            if termination is not None:
                stop = termination['stop_reason']
                break
            context = compact_model_context(context)
            prompt = (STATIC_EDIT_PROMPT if editor.constraints.get('stationary_target') else EDIT_PROMPT) + '\n' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))
            if editor.pdm_probe is not None:
                prompt=PDM_PROBE_PROMPT+'\n'+prompt
                if editor.constraints.get('pdm_defense_feedback'):
                    prompt=DEFENSE_FEEDBACK_PROMPT+'\n'+prompt
            content = prompt if pending is None else [{'type':'text', 'text':prompt}, image_part(pending['path'])]
            probe_context = editor.pdm_probe.context() if editor.pdm_probe is not None else {}
            turn_tools = trajectory_tools(available_tools,editor.nodes,evaluated_candidates,
                set(probe_context.get('candidates',{})),probe_context.get('remaining',0),
                constructor_remaining=context.get('constructor_budget',{}).get('remaining_evaluations'),
                reselections_remaining=max(0,1-editor.constraints.get('object_reselection_count',0)))
            turn_names = {tool['function']['name'] for tool in turn_tools}
            choice = 'auto'
            if (turn == 1 and not imported_count and not editor.constraints.get('stationary_target')
                    and 'construct_attack_candidates' in turn_names):
                choice = {'type':'function', 'function':{'name':'construct_attack_candidates'}}
            elif turn == 1 and imported_count:
                choice = {'type':'function', 'function':{'name':'inspect_trajectory'}}
            print('Trajectory editing turn', turn, flush=True)
            _, attempt = archive.complete(f'edit_{turn:02d}', [{'role':'user', 'content':content}],
                total_seconds=timeout_seconds, max_tokens=max_tokens,
                extra_body={'thinking':{'type':'enabled'}}, reasoning_effort='high',
                tools=turn_tools, tool_choice=choice, parallel_tool_calls=False, allow_tool_calls=True)
            if pending is not None:
                if pending.get('purpose') != 'pdm_probe':
                    editor.inspected.add(pending['candidate_id'])
                deliveries.append(dict(candidate_id=pending['candidate_id'], request_path=str(attempt), turn=turn))
                pending = None
            call, name, args = decode_call(attempt)
            try:
                if name not in turn_names:
                    raise ValueError('Tool unavailable for the current candidate eligibility or resource state')
                if name in ('evaluate_candidate','probe_pdm'):
                    schema = next(tool['function']['parameters'] for tool in turn_tools if tool['function']['name']==name)
                    if args.get('candidate_id') not in schema['properties']['candidate_id']['enum']:
                        raise ValueError('Candidate is not eligible for this tool in the current state')
                if name not in ('inspect_trajectory', 'evaluate_candidate', 'probe_pdm') and (not isinstance(args.get('reason'), str) or not args['reason'].strip()):
                    raise ValueError('A nonempty action reason is required')
                if name == 'evaluate_candidate' and set(args) == {'candidate_id'}:
                    feedback = editor.evaluate_candidate(args['candidate_id'])
                    evaluated_candidates.add(args['candidate_id'])
                elif name == 'probe_pdm' and set(args)=={'candidate_id'}:
                    feedback=editor.pdm_probe.probe(args['candidate_id'])
                    if feedback.get('status')=='completed':
                        pending=dict(candidate_id=args['candidate_id'],path=feedback['overlay_path'],purpose='pdm_probe')
                elif name == 'construct_attack_candidates':
                    feedback = editor.construct_pool(args)
                elif name == 'construct_trajectory' and set(args) == {'points', 'end_velocity_mps', 'reason'}:
                    feedback = editor.construct(args)
                elif name in ('edit_speed_profile','edit_route'):
                    feedback = editor.edit_motion(normalize_edit_call(name,args))
                elif name == 'repair_candidate' and {'parent_candidate_id', 'target_candidate_id', 'ratio', 'reason'} <= set(args) <= {'parent_candidate_id', 'target_candidate_id', 'ratio', 'reason', 'window_s'}:
                    feedback = editor.repair(args)
                elif name == 'inspect_trajectory' and set(args) == {'candidate_id'}:
                    feedback = editor.inspect(args['candidate_id'])
                    pending = feedback
                elif name == 'submit_candidate' and set(args) == {'candidate_id', 'reason'}:
                    feedback = editor.submit(args['candidate_id'], args['reason'])
                    stop = 'candidate_submitted'
                elif name == 'request_object_reselection' and set(args) == {'reason'}:
                    if editor.pdm_probe is not None:
                        editor.pdm_probe.require_reselection_evidence()
                    if editor.constraints.get('object_reselection_count', 0) >= 1:
                        raise ValueError('Object reselection budget exhausted')
                    feedback = dict(status='object_reselection_requested', reason=args['reason'])
                    stop = 'object_reselection_requested'
                elif name == 'finish_search' and set(args) == {'reason'}:
                    feedback = dict(status='finished', reason=args['reason'])
                    stop = 'finished_without_submission'
                else:
                    raise ValueError('Unknown tool or invalid arguments')
            except (ValueError, KeyError, TypeError) as error:
                feedback = dict(status='invalid', error=str(error)[:350])
            archive_exchange(folder, turn, call, feedback)
            print('Tool', name, 'returned', feedback.get('status', feedback.get('reference_screen_passed')), flush=True)
            if editor.submitted or stop in ('finished_without_submission', 'object_reselection_requested'):
                break
            if name == 'construct_attack_candidates':
                termination = constructor_exhaustion(editor.nodes,editor.context()['constructor_budget'])
                if termination is not None:
                    stop = termination['stop_reason']
                    break
    except Exception:
        stop = 'request_or_execution_failed'
        raise
    finally:
        atomic_json(folder / 'output.json', dict(schema_version='attack_trajectory.v1',
            status='completed' if editor.submitted else 'not_submitted', stop_reason=stop,
            selected_candidate_id=editor.submitted, target_track=editor.target['track_id'],
            reason=None if termination is None else termination['reason'],
            termination_evidence=termination,
            candidate_count=len(editor.nodes), image_deliveries=deliveries,
            imported_candidate_count=imported_count, new_candidate_count=len(editor.history) - imported_count,
            constructor_budget=editor.context()['constructor_budget'],
            annotation_changes=dict(operation='export_target_trajectory', frames_json_modified=False),
            physical_collision='undetermined'))
    return editor.submitted


@timed_operation('attack.workflow', 'output')
def run(scene, output, instruction, assets=None, constraints=None, timeout_seconds=480,
        max_tokens=12288, max_turns=12, required_operation=None, required_track=None, continue_from=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if (experiments_root()).resolve() not in output.parents or output.exists():
        raise ValueError('Workflow output must be a new directory under experiments')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty user instruction is required')
    config = dict(observation_end_s=policy.OBSERVATION_END_S, official_margin_m=policy.OFFICIAL_MARGIN_M, max_ego_clearance_m=policy.ATTACK_EVENT_MAX_GAP_M, min_travel_m=0.)
    config.update(constraints or {})
    if config.get('pdm_self_check'):
        config.setdefault('pdm_probe_root',str(output/'pdm-probes'))
        if type(config.get('pdm_probe_budget',2)) is not int or config.get('pdm_probe_budget',2)<1:
            raise ValueError('PDM probe budget must be a positive integer')
    effective = policy.resolve(config.get('acceptance_policy'))
    for key in ('official_margin_m','observation_end_s'):
        if constraints and key in constraints:
            if constraints.get('acceptance_policy') and key in constraints['acceptance_policy'] and effective[key] != config[key]:
                raise ValueError('Conflicting task and acceptance setting: '+key)
            effective[key] = config[key]
        else:
            config[key] = effective[key]
    config['acceptance_policy'] = effective
    frames = json.loads((scene / 'metadata/frames.json').read_text())
    times, _, _ = ego_path_from_frames(frames)
    for key in ('observation_end_s', 'official_margin_m', 'max_ego_clearance_m', 'min_travel_m'):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not 3. <= config['observation_end_s'] < times[-1]:
        raise ValueError('Observation prefix must be at least 3 seconds and shorter than the scene')
    if required_operation not in {None, 'add_vehicle', 'edit_existing'}:
        raise ValueError('Unknown object operation requirement')
    if type(max_turns) is not int or max_turns < 1 or timeout_seconds <= 0:
        raise ValueError('Positive action and timeout budgets are required')
    if 'event_window_s' in config:
        window = np.asarray(config['event_window_s'], dtype=float)
        if window.shape != (2,) or not np.isfinite(window).all() or not times[0] <= window[0] < window[1] <= times[-1] or window[1] <= config['observation_end_s']:
            raise ValueError('Event window must be ordered, inside the scene and extend after observation')
    if config.get('task_contract') is not None:
        from attack_task_contract import validate_task_contract
        if continue_from is None and 'schema_version' not in config['task_contract']:
            raise ValueError('Task contracts require attack_task.v2 provenance')
        validate_task_contract(config['task_contract'], times, config['observation_end_s'])
    assets = assets or []
    if len({asset['asset_id'] for asset in assets}) != len(assets):
        raise ValueError('Asset IDs must be distinct')
    for asset in assets:
        size = np.asarray(asset['size_m'], dtype=float)
        if not Path(asset['path']).is_file() or size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError('Assets require an existing file and positive finite length/width/height')
    from attack_difficulty import resolve_difficulty,run_search
    difficulty=resolve_difficulty(config.get('attack_difficulty'))
    if 'attack_difficulty' in config: config['attack_difficulty']=difficulty
    if difficulty['mode']!='free':
        if continue_from is not None:
            raise ValueError('Difficulty search starts a new explicit branch; it cannot resume old Agent requests')
        return run_search(scene,output,instruction,assets,config,required_operation,required_track)
    output.mkdir(parents=True)
    atomic_json(output / 'input.json', dict(schema_version='attack_workflow.v1', scene=str(scene),
        instruction=instruction, assets=assets, constraints=config, attack_difficulty=difficulty, required_operation=required_operation,
        required_track=required_track, reasoning_effort='high', timeout_seconds=timeout_seconds,
        max_tokens=max_tokens, max_turns=max_turns,
        continued_from=None if continue_from is None else str(Path(continue_from).resolve())))
    profile = load_llm_profile()
    client = OpenAI(api_key=profile['api_key'], base_url=profile['base_url'], max_retries=0,
        http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=float(timeout_seconds)))
    archive = StreamingRequestArchive(output / 'requests', client, profile)
    summary = dict(schema_version='attack_workflow.v1', status='failed', physical_collision='undetermined',attack_difficulty=difficulty)
    stage = 'object_selection'
    try:
        if continue_from is None:
            selection = select_object(scene, output / 'object_selection', instruction, assets, config,
                archive, timeout_seconds, max_tokens, required_operation, required_track)
        else:
            import shutil
            previous = Path(continue_from).resolve()
            if previous == output or (experiments_root()).resolve() not in previous.parents:
                raise ValueError('Continuation source must be another experiments run')
            prior_input = json.loads((previous / 'input.json').read_text())
            if Path(prior_input['scene']).resolve() != scene or prior_input['constraints'] != config or prior_input['instruction'] != instruction or prior_input['assets'] != assets or prior_input.get('required_operation') != required_operation or prior_input.get('required_track') != required_track:
                raise ValueError('Continuation scene, constraints and instruction must match')
            selection = json.loads((previous / 'object_selection/output.json').read_text())
            if selection.get('status') != 'completed':
                raise ValueError('Continuation requires completed object selection')
            folder = output / 'object_selection'
            folder.mkdir()
            shutil.copyfile(selection['seed_trajectory_path'], folder / 'seed_trajectory.csv')
            selection.update(seed_trajectory_path=str(folder / 'seed_trajectory.csv'),
                             imported_from=str(previous / 'object_selection/output.json'))
            atomic_json(folder / 'input.json', dict(imported_from=str(previous / 'object_selection'),
                                                   note='Previously accepted selection data, not a new model response'))
            atomic_json(folder / 'output.json', selection)
        if selection.get('status') == 'not_submitted':
            summary.update(selection)
            return summary
        stage = 'trajectory_editing'
        editor = AttackEditor(scene, selection, output / 'trajectory_editing', config)
        if continue_from is not None:
            for name in ('constructor-budget.json', 'construction_pools'):
                source = previous / 'trajectory_editing' / name
                if source.is_dir():
                    shutil.copytree(source, editor.run_dir / name)
                elif source.is_file():
                    shutil.copyfile(source, editor.run_dir / name)
            for candidate in sorted((previous / 'trajectory_editing/candidates').glob('candidate_*')):
                operation = json.loads((candidate / 'operation.json').read_text())['operation']
                method = {'construct_trajectory':editor.construct,
                          'edit_speed_profile':editor.edit_motion,'edit_route':editor.edit_motion,
                          'repair_candidate':editor.repair, 'construct_from_offsets':editor.construct_offsets,
                          'construct_vehicle_motion':editor.construct_motion,
                          'construct_vehicle_controls':editor.construct_controls}[operation['tool']]
                if operation['tool'] in ('edit_speed_profile','edit_route'):
                    key,version = ('speed_schedule_version','state_continuity.v4') if operation['tool']=='edit_speed_profile' else ('route_edit_version','state_continuity.v2')
                    if operation['input'].get(key) != version:
                        raise ValueError('Archived editing versions are unsupported; start a new task')
                replayed = method(operation['input'])
                _, old_xy, old_yaw = read_seed(candidate / 'trajectory.csv')
                node = editor.nodes[replayed['candidate_id']]
                if replayed['candidate_id'] != candidate.name or not np.allclose(node['xy'], old_xy, atol=1e-8, rtol=0) or not np.allclose(node['yaw'], old_yaw, atol=1e-8, rtol=0):
                    raise ValueError('Archived candidate does not match its verified operation replay')
                atomic_json(editor.run_dir / 'candidates' / candidate.name / 'import-source.json',
                            dict(source_candidate=str(candidate), verified_operation_replay=True))
        inherited = config.get('inherited_constructor_budget')
        if inherited is not None and continue_from is None:
            atomic_json(editor.run_dir/'constructor-budget.json', inherited)
        selected = edit_object(editor, selection, archive, timeout_seconds, max_tokens, max_turns)
        if editor.pdm_probe is not None:
            summary['pdm_probe_feedback']=editor.pdm_probe.context()
            if selected is not None:
                summary['pdm_probe_result_path']=editor.pdm_probe.require_challenge(selected)['result_path']
        summary.update(status='not_submitted' if selected is None else 'reference_candidate_submitted',
            target=selection['target'], selected_candidate_id=selected,
            object_output=str(output / 'object_selection/output.json'),
            trajectory_output=str(output / 'trajectory_editing/output.json'))
        if selected is not None:
            target, node = selection['target'], editor.nodes[selected]
            if target['operation'] == 'add_vehicle':
                from add_vehicle_scene import add_vehicle
                detail = add_vehicle(scene, output / 'materialized', 0, target['track_id'], target['size_m'],
                    trajectory_xy=node['xy'], trajectory_yaw=node['yaw'], official_min_margin_m=config['official_margin_m'])
                atomic_json(output / 'materialized/asset.json', dict(asset_path=target['asset_path']))
                # Existing renderer reads asset_path from the added-object summary.
                detail['asset_path'] = target['asset_path']
                detail['candidate_id'] = selected
                atomic_json(output / 'materialized/summary.json', detail)
                atomic_json(output / 'materialized/frozen_scene/freeze-info.json', detail)
                frozen = output / 'materialized/frozen_scene'
            else:
                from freeze_attack_scene import freeze
                frozen = output / 'materialized/frozen_scene'
                freeze(scene, editor.run_dir, selected, target['track_id'], frozen)
            from ego_edit_evidence import defense_difficulty_or_unknown
            summary['defense_difficulty']=defense_difficulty_or_unknown(frozen,target['track_id'],config.get('acceptance_policy'))
            summary.update(frozen_scene_path=str(frozen),
                trajectory_path=str(editor.run_dir / 'candidates' / selected / 'trajectory.csv'),
                metrics=editor.public(selected),
                annotation_changes=dict(operation='append_actor' if target['operation'] == 'add_vehicle' else 'update_actor',
                    fields=['anns.gt_boxes', 'anns.gt_velocity_3d'] +
                    (['anns.gt_names', 'anns.instance_tokens', 'anns.track_tokens'] if target['operation'] == 'add_vehicle' else [])))
    except Exception as error:
        summary.update(status='failed', error_type=type(error).__name__, error=str(error)[:350])
        stage_output = output / stage / 'output.json'
        if not stage_output.exists():
            atomic_json(stage_output, dict(status='failed', error_type=type(error).__name__,
                                          error=str(error)[:350]))
        raise
    finally:
        client.close()
        atomic_json(output / 'output.json', summary)
    stage_result = json.loads((editor.run_dir/'output.json').read_text())
    if stage_result['stop_reason'] == 'object_reselection_requested':
        used = len(list((editor.run_dir/'tool_calls').glob('turn_*')))
        # Exchanges are stored as turn directories; no model or evaluation reset.
        remaining_turns = max_turns-used
        if remaining_turns < 1:
            summary.update(status='not_submitted', stop_reason='action_budget_exhausted_before_reselection')
        else:
            reason = json.loads(sorted((editor.run_dir/'tool_calls').glob('turn_*/result.json'))[-1].read_text())['reason']
            revised = dict(config, object_reselection_count=1, object_reselection_reason=reason,
                           inherited_constructor_budget=editor.context()['constructor_budget'])
            revised['inherited_constructor_budget'] = {k:revised['inherited_constructor_budget'][k] for k in ('calls','evaluations')}
            if editor.pdm_probe is not None:
                latest=editor.pdm_probe.context()['latest_feedback']
                if latest is not None:
                    revised['pdm_reselection_feedback']=latest
            followup = output/'object_revision_01'
            child = run(scene, followup, instruction, assets, revised, timeout_seconds,
                        max_tokens, remaining_turns, required_operation, required_track)
            summary.update(status='continued_in_object_revision', followup_run=str(followup),
                           followup_status=child['status'])
        atomic_json(output/'output.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--constraints', type=Path, help='JSON constraints including task_contract')
    parser.add_argument('--asset', type=Path)
    parser.add_argument('--size', type=float, nargs=3)
    parser.add_argument('--min-travel-m', type=float, default=0.)
    parser.add_argument('--max-ego-clearance-m', type=float, default=2.)
    parser.add_argument('--timeout-seconds', type=int, default=480)
    parser.add_argument('--max-turns', type=int, default=12)
    parser.add_argument('--continue-from', type=Path, help='Explicitly import and recheck object/candidate data into a new run')
    args = parser.parse_args()
    if bool(args.asset) != bool(args.size):
        parser.error('--asset and --size must be supplied together')
    assets = [] if not args.asset else [dict(asset_id='asset_001', path=str(args.asset.resolve()), size_m=args.size)]
    summary = run(args.scene, args.output, args.instruction, assets,
        dict(dict(require_structured_objective=True, vehicle_motion_model='bicycle.v1') |
             ({} if args.constraints is None else json.loads(args.constraints.read_text())),
             min_travel_m=args.min_travel_m, max_ego_clearance_m=args.max_ego_clearance_m),
        timeout_seconds=args.timeout_seconds, max_turns=args.max_turns, continue_from=args.continue_from)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
