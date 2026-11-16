"""Two-stage attack editing: select/create an actor, then edit a generic trajectory."""
from llm_profile import load_llm_profile,reasoning_effort
import acceptance_policy as policy
from workflow_timing import timed_operation

import argparse
import copy
import json
import os
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np

from openai import OpenAI
from shapely.geometry import LineString

from attack_tool_contract import candidate_eligibility, constructor_exhaustion, revision_evidence, trajectory_tools
from model_context import compact_model_context
from attack_editor import AttackEditor, read_seed, write_trajectory
from attack_configuration import normalize_constraints, same_scene_constraints
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
from initial_motion_precheck import initial_state_precheck
from attack_stop_policy import repetition
from action_validation import prevalidate_action, invalid_action_result
from agent_tool_view import compact_action_feedback
from attack_budget import BudgetedArchive, AttackBudgetExhausted, constructor_budget_path
from attack_plan_options import shared_auto_session


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
    'No interaction category or fixed proximity gate is required. Review measured geometry against the task. '
    'reason is explanatory only: all numeric requirements must appear in objective. '
    'Returns fully checked candidates separately from optimization guidance. Motion may be edited from edit_start_s (0s in new runs). No implicit 3-second event delay applies; respect explicit task windows and physical/map checks.',
    {'parent_candidate_id':{'type':'string'},
     'objective':dict(type='object', properties={
         'keyframes':dict(type='array', items=KEYFRAME, minItems=0, maxItems=6)},
         required=[],additionalProperties=False),
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
this straight-motion seed is an initial proposal that may be edited from the first editable time (0s in
current runs); only the t=0 pose is fixed. Choose a safe
initial pose with space for a valid early trajectory. Final candidates must pass their explicit task, road and motion checks.
When plan_options are supplied, compare them once and choose a promising existing or added actor;
option order is not a quality rank. You may adapt an option within the unchanged scene mechanism.
Use optional option_id and explain the actual choice/deviation in reason. Do not run all alternatives.
previous_object_feedback contains real failures: reconsider unsupported targets and use an added
vehicle when that resolves a placement or inherited-state problem. Only explicit required_operation
and required_track are hard object requirements. Existing t=0 poses cannot be relocated; a newly
placed stationary car should use add_vehicle with speed_mps=0 and a supported initial pose.
Coordinates are metres in the initial ego frame: x forward, y left, yaw counterclockwise radians.
Consider the full ego timeline and stopped traffic before selecting the initial state. A safe initial
pose alone does not guarantee room to merge later. Use vehicle_planning_hints to compare
stopping/turning space with observed front traffic; choose a lower speed or smaller side offset
when space is tight. Avoid choosing a speed and side offset that
would require a late backward correction. If no suitable object or placement can be selected within current tools/budget, finish_search with measured reasons. Provide one native tool call per turn."""

EDIT_PROMPT = """Edit the selected target vehicle using the scene BEV, candidate diagrams and measured feedback.
DECIDE: identify the conflict, compare current candidates, then choose one concrete tool action.
Do not reconstruct a whole scene from coordinates or repeat the numeric reports in your response.
INPUTS: scene_layout contains a small relevance summary; all actors and full-precision trajectories remain
in deterministic checks and scene-evidence.json. Pictures show geometry of this scene version, not a camera
render or tested policy rollout. Recorded ego is a reference. Original cameras are pre-edit observations.
GOALS: explicit caller task_contract and configured event window are hard; interaction quality is for your judgment. New attack events default to 1.5–7.5s;
editing/braking/steering may start at 0s. Agent-generated keyframes are optimization preferences unless an
explicit contract makes them required. Read acceptance_basis and actual measurements; explain whether a
candidate still meets the qualitative instruction when a preference misses. Do not change caller requirements.
TOOLS: first use construct_attack_candidates with achievable optional motion keyframes. No contact/near-miss/either class is used.
Only structured fields guide construction; reason is a brief explanation. Every hold_duration_s must fit its
own time_window_s. Use edit_speed_profile for signed speed targets; use edit_route for smooth lateral offsets
and return. Select an objective-bearing parent exposed by the tool enum. Preview checks do not create a candidate.
Read decision_summary and recommended_previews; an explicit goal revision is different from a numeric repair.
Body-relative goal positions use ego BODY center; scene coordinates use initial ego forward/left. Center
separation is not bumper clearance. Reference contact and clearance are measured facts, not mandatory event classes or proof that VADv2 will crash. Do not make a stopped car slide or rotate in place.
CHECK: physical/map checks and native-replay checks must pass. Generated and replayed paths can differ, so
use both reports. Speed limits here are a named research profile, not certified vehicle capabilities.
REVIEW: initial layout and newly constructed candidate pictures are supplied automatically. Inspect another
candidate when needed. The diagram must actually be delivered before submission; a policy-test overlay or
layout-only BEV does not satisfy that review. No unnecessary edit is required after the task is satisfied.
If an enabled policy self-test is configured, complete its exact-candidate test; PDM zero or road failure alone
is not target collision. When policy tests are disabled, report reference validity without claiming policy defeat.
BUDGET: read cumulative_api_budget, constructor_budget and stage_seconds_remaining. In finalization_only,
choose an existing qualified candidate for review/submission or finish_search; do not restart construction.
A new tool call cannot reset cumulative budgets. If no candidate can meet the task, finish honestly with the
measured cause. Use one exposed native tool call and a short reason per turn."""


STATIC_EDIT_PROMPT = """Validate the selected stationary placement. Only the selected target is added or
edited; ego and observed others replay their records. The target pose is fixed for the entire scene.
Read seed_000's reference checks, fixed task contract and geometry_feedback. Use measured body overlap
times, never mental center-arrival estimates. No implicit three-second event constraint applies; use the explicit task interval. Do not invent a narrower hard window, hold duration or reaction delay.
Only explicit task requirements and named checking policy are hard; read provenance and uncertainties.
If pdm_self_check is enabled, passing geometry only makes the placement eligible for probe_pdm.
Test first; a safely defended placement needs reselection or an honest finish. A tested PDM-failing
placement must still use inspect_trajectory and receive the image before submit_candidate(seed_000).
Only when no policy self-test is enabled do geometry plus trajectory inspection alone permit submission.
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

VADV2_PROBE_PROMPT = '''VADv2 self-test is enabled. Construct/edit a candidate, pass its trajectory
and task checks, call probe_vadv2, inspect actual VAD trajectory feedback, then revise if the selected
attack vehicle was not hit. Submission requires a successful test on the exact candidate. Geometry
or contact with recorded ego alone is not enough. Use min_target_clearance_m, closest_approach time,
policy_behavior_summary (stopped_detected, progress_ratio_vs_recorded, signed_yaw_change_deg),
forward/left/closing_speed and the plot to understand braking/early-stop or steering avoidance. Road violation,
PDM zero, history contact and interpolated-only contact do not establish VAD failure. Tests share
vadv2_self_test.remaining across edits/reselection. Do not repeat unchanged controls after no contact.
Keep the original task and physical/map checks. After a successful VAD test, inspect_trajectory still
supplies attack/road evidence before submission. Main Agent reuses this exact-version evidence.'''

TRAJECTORY_TOOLS.append(function_tool('probe_vadv2',
    'Run an eligible candidate through WE + VADv2. Returns actual target contact, closest approach '
    'and an image. Minute-scale bounded test; exact-version repeats reuse evidence.',
    {'candidate_id':{'type':'string'}}))

TRAJECTORY_TOOLS.append(function_tool('probe_pdm',
    'Test an eligible candidate with real PDM + LQR/kinematic closed loop. Returns actual trajectory '
    'samples, closest approach and an overlay image. Shared simulation budget; repeats reuse the same candidate result.',
    {'candidate_id':{'type':'string'}}))

TRAJECTORY_TOOLS.append(function_tool('evaluate_candidate',
    'Return measured full-horizon reference geometry and task checks for an existing candidate. No optimization.',
    {'candidate_id':{'type':'string'}}))
EDIT_PROMPT += (' When plan_options are present, plan_choice describes the chosen suggestion, not another hard contract. '
    'If inherited state or fixed t=0 placement makes this actor unsuitable, request_object_reselection rather than repeatedly constructing with the same unsupported state. '
    'The next object-selection step can change to another real vehicle or an added vehicle. Switching shares API, construction, action and stage-time budgets; do not reset them. '
    'A failed candidate can still be repaired; switching is an option, not a requirement to abandon it.')

TRAJECTORY_TOOLS.append(function_tool('request_object_reselection', 'Return to object selection within the remaining switching budget when the current actor/placement is unsuitable. Auto alternatives may change existing/added operation; explicit caller locks stay binding. Retain spent API, construction, action and policy-test budgets.', {'reason':REASON}))

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


def _undecoded_call(turn):
    """模型未返回单个原生 tool_call 时的占位记录，使错误可以回灌给下一轮。"""
    return {'id': f'undecoded_{turn:02d}', 'type': 'function',
            'function': {'name': 'none', 'arguments': '{}'}}


def archive_exchange(folder, turn, call, result, *, started=None):
    from common.llm_audit import record_tool_exchange
    path = folder / 'tool_calls' / f'turn_{turn:02d}'
    atomic_json(path / 'call.json', call)
    atomic_json(path / 'result.json', result)
    record_tool_exchange(call, result, path,
                         elapsed_seconds=None if started is None else perf_counter()-started)


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
    from attack_plan_options import selection_tools, choice
    options=constraints.get('plan_options',[])
    object_tools=selection_tools(OBJECT_TOOLS,assets,required_operation,required_track,options)
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
        plan_options=options,previous_object_feedback=constraints.get('object_selection_feedback',[]),
        constraints=constraints,
        ego_reference_point='rear axle',ego_body_center_shift_m=(EGO_FRONT-EGO_REAR)/2,
        ego_body_dimensions_m=dict(length=EGO_FRONT+EGO_REAR,width=EGO_WIDTH,front_from_rear_axle=EGO_FRONT,rear_from_rear_axle=EGO_REAR),
        ego_timeline=[dict(time_s=float(moment), xy_m=position.tolist(), yaw_rad=float(angle),
                          speed_to_next_mps=None if index == len(times) - 1 else float(ego_speeds[index]))
                      for index, (moment, position, angle) in enumerate(zip(times, ego, yaw))],
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
    atomic_json(folder / 'tools.json', object_tools)
    from decision_scene_view import render_overview, actor_view, GUIDANCE
    overview=render_overview(times,ego,yaw,actors,road,folder/'scene-overview.png',required_track,event_window_s=constraints.get('event_window_s'))
    decision_context=dict(context)
    decision_context.pop('actors_at_start',None)
    decision_context['constraints']={key:value for key,value in constraints.items()
        if key not in ('plan_options','object_selection_feedback')}
    decision_context['scene_layout']=actor_view(times,ego,actors,required_track)
    decision_context['visual_guidance']=GUIDANCE
    feedback = None
    for turn in range(1, 5):
        prompt=OBJECT_PROMPT + '\n' + json.dumps(
            compact_model_context(dict(context=decision_context, last_tool_result=feedback)), ensure_ascii=False, separators=(',', ':'))
        if constraints.get('pdm_defense_feedback'):
            prompt=DEFENSE_FEEDBACK_PROMPT+'\n'+prompt
        review=constraints.get('pdm_reselection_feedback')
        if review is None:
            evidence=revision_evidence(constraints)
            if evidence.get('overlay_path'):
                review=evidence
        content=prompt if not review else [dict(type='text',text=prompt),image_part(review['overlay_path'])]
        messages = [{'role':'user', 'content':content}]
        print('Object selection turn', turn, flush=True)
        if turn == 1:
            if isinstance(content,str): content=[{'type':'text','text':content}]
            messages=[{'role':'user','content':[*content,image_part(overview)]}]
        _, attempt = archive.complete(f'object_{turn:02d}', messages, total_seconds=timeout_seconds,
            no_output_seconds=min(60, timeout_seconds), retries=2,
            max_tokens=max_tokens, extra_body={'thinking':{'type':'enabled'}}, reasoning_effort=reasoning_effort(),
            tools=object_tools, tool_choice='required', parallel_tool_calls=False, allow_tool_calls=True)
        call = None
        name = 'none'
        tool_started = perf_counter()
        try:
            call, name, args = decode_call(attempt)
            selection_args=dict(args)
            option_id=args.pop('option_id',None)
            if option_id is not None and option_id not in {item['option_id'] for item in options}:
                raise ValueError('Unknown coarse option_id')
            if not isinstance(args.get('reason'), str) or not args['reason'].strip():
                raise ValueError('A nonempty selection reason is required')
            if name == 'finish_search' and set(args) == {'reason'}:
                result=dict(status='not_submitted',stop_reason='finished_without_object_selection',
                            reason=args['reason'],search_stage='object_selection')
                atomic_json(folder/'output.json',result)
                archive_exchange(folder,turn,call,dict(status='finished',reason=args['reason']),started=tool_started)
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
            # Reject unsafe seed poses at t=0; the rest of the trajectory is editable.
            # Final candidates are screened over the complete observation window.
            fixed_end=float(constraints.get('attack_prefix_lock_s',policy.edit_start(policy.resolve(constraints.get('acceptance_policy')))))
            prefix_times = np.linspace(times[0], fixed_end, 31) if fixed_end>times[0] else np.asarray([times[0]])
            prefix_xy = sample_xy(times, seed, prefix_times)
            prefix_yaw = np.interp(prefix_times, times, angles)
            prefix_road = check_drivable_area_official(prefix_times, prefix_xy, prefix_yaw, road,
                front=size[0] / 2, rear=size[0] / 2, width=size[1], min_margin_m=constraints['official_margin_m'])
            if not prefix_road['passed']:
                raise ValueError('Selected seed leaves the road during the fixed initial interval')
            ego_prefix = sample_xy(times, ego, prefix_times)
            ego_angles = np.interp(prefix_times, times, yaw)
            series = actor_observation_series(times, actors, track)
            # 循环变量不能叫 time：这会让整个函数把 time 当成局部变量，
            # 使上面 `perf_counter()` 变成 UnboundLocalError。
            for moment, position, angle, ego_position, ego_angle in zip(prefix_times, prefix_xy, prefix_yaw, ego_prefix, ego_angles):
                body = polygon(*position, angle, *size[:2])
                center = ego_position + (EGO_FRONT - EGO_REAR) / 2 * np.array([np.cos(ego_angle), np.sin(ego_angle)])
                if body.intersection(polygon(*center, ego_angle, EGO_FRONT + EGO_REAR, EGO_WIDTH)).area > 1e-9:
                    raise ValueError(f'Selected seed contacts ego during observation at {moment:.2f}s')
                for other_track, observed, poses in series:
                    pose = actor_pose_at_time(observed, poses, moment, 1.1 * np.median(np.diff(times)), scene_times=times)
                    if pose is not None and body.intersection(polygon(*pose[:2], pose[2], pose[3], pose[4])).area > 1e-9:
                        raise ValueError(f'Selected seed contacts observed actor {other_track} during observation at {moment:.2f}s')
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
                plan_choice=choice(options,selection_args,target) if options else None,
                selection_reason=args['reason'],initial_relation=initial_relation, annotation_changes=dict(operation='none',
                    note='Identity and motion seed only; full scene is materialized after trajectory acceptance'))
            forecast_editor = AttackEditor(scene, selection, folder / f'placement_check_{turn:02d}', constraints)
            selection['geometry_feedback'] = forecast_editor.public('seed_000')
            atomic_json(folder / 'output.json', selection)
            feedback = dict(status='selected', target=target, geometry_feedback=selection['geometry_feedback'], output_path=str(folder / 'output.json'))
            archive_exchange(folder, turn, call, feedback, started=tool_started)
            return selection
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            feedback = dict(status='invalid', error=str(error)[:350])
            archive_exchange(folder, turn, call or _undecoded_call(turn), feedback, started=tool_started)
    result=dict(status='not_submitted',stop_reason='object_selection_budget_exhausted',
                search_stage='object_selection',reason='No valid object selection or initial placement was found within the object-selection budget.')
    atomic_json(folder/'output.json',result)
    return result


@timed_operation('attack.agent_loop', None)
def edit_object(editor, selection, archive, timeout_seconds, max_tokens, max_turns, stage_timeout_s=420):
    from attack_budget import BudgetedArchive, AttackBudgetExhausted
    from attack_tool_contract import candidate_eligibility
    folder = editor.run_dir
    atomic_json(folder / 'input.json', dict(selection=selection, constraints=editor.constraints))
    if editor.constraints.get('vadv2_self_test'):
        from attack_policy_probe import VADProbe
        editor.policy_probe=VADProbe(editor)
    if editor.constraints.get('pdm_self_check'):
        from attack_pdm_probe import PDMProbe
        editor.pdm_probe=PDMProbe(editor)
    available_tools = [tool for tool in TRAJECTORY_TOOLS if not (
        (editor.constraints.get('require_structured_objective') or editor.constraints.get('task_contract')) and tool['function']['name']=='construct_trajectory')]
    if editor.constraints.get('stationary_target'):
        allowed = {'evaluate_candidate', 'inspect_trajectory', 'submit_candidate',
                   'request_object_reselection', 'finish_search', 'probe_pdm', 'probe_vadv2'}
        available_tools = [tool for tool in available_tools if tool['function']['name'] in allowed]
    if editor.pdm_probe is None:
        available_tools=[tool for tool in available_tools if tool['function']['name']!='probe_pdm']
    if editor.policy_probe is None:
        available_tools=[tool for tool in available_tools if tool['function']['name']!='probe_vadv2']
    atomic_json(folder / 'tools.json', available_tools)
    seed = editor.nodes.get('seed_000')
    fast_stationary = bool(editor.constraints.get('stationary_target') and seed is not None
        and editor.pdm_probe is None and editor.policy_probe is None
        and candidate_eligibility(seed['result'])['pdm_probe_eligible'])
    feedback = editor.public(editor.history[-1]['candidate_id']) if editor.history else editor.public('seed_000')
    from decision_scene_view import prepare_attack_scene
    pending = prepare_attack_scene(editor,folder)
    imported_count = len(editor.history)
    deliveries = []
    evaluated_candidates = set()
    stop = 'action_budget_exhausted'
    termination = None
    if editor.constraints.get('plan_options'):
        from attack_plan_options import editor_deadline
        deadline=editor_deadline(stage_timeout_s)
    else:deadline = perf_counter() + stage_timeout_s
    finalization_grace_used = False
    previous_failure = None
    try:
        for turn in range(1, max_turns + 3):
            remaining = deadline - perf_counter()
            qualified=[key for key,node in editor.nodes.items() if candidate_eligibility(node['result'])['pdm_probe_eligible']]
            session_budget=archive.context() if isinstance(archive,BudgetedArchive) else None
            closing = bool(qualified) and (remaining < 60 or turn > max_turns or finalization_grace_used
                or session_budget is not None and (session_budget['requests_remaining']<=2 or session_budget['api_seconds_remaining']<=90))
            if turn > max_turns and not closing:
                break
            if remaining <= 0 and qualified and not finalization_grace_used:
                deadline=perf_counter()+90
                remaining=90
                finalization_grace_used=True
                closing=True
            if remaining <= 0:
                stop = 'stage_timeout_exhausted'
                termination = dict(reason='Attack trajectory stage exceeded its wall-clock budget.',
                    stage_timeout_seconds=stage_timeout_s,scope='current_tools_and_budget',global_infeasibility_proven=False)
                break
            context = dict(instruction=selection['instruction'], **editor.context(),
                           plan_choice=selection.get('plan_choice'),
                           last_tool_result=feedback, turns_remaining=max(0,max_turns - turn + 1),
                           finalization_only=closing,stage_seconds_remaining=round(remaining,1))
            if closing:
                context['finalization_instruction']='Review and submit an existing qualified candidate, or finish honestly. No more construction or edits.'
            if isinstance(archive,BudgetedArchive):
                archive.closing=closing
                context['cumulative_api_budget']=session_budget
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
            pending_records=[] if pending is None else pending.get('batch',[pending])
            content = prompt if not pending_records else [{'type':'text', 'text':prompt}, *[image_part(record['path']) for record in pending_records]]
            if editor.policy_probe is not None:
                if isinstance(content,str):content=VADV2_PROBE_PROMPT+'\n'+content
                else:content[0]['text']=VADV2_PROBE_PROMPT+'\n'+content[0]['text']
            probe_context = editor.pdm_probe.context() if editor.pdm_probe is not None else {}
            if editor.policy_probe is not None:probe_context=editor.policy_probe.context()
            delivered = getattr(editor,'inspected',set()) | {record['candidate_id'] for record in pending_records
                if record.get('purpose') not in ('pdm_probe','vadv2_probe','scene_layout')}
            turn_tools = trajectory_tools(available_tools,editor.nodes,evaluated_candidates,
                set(probe_context.get('candidates',{})),probe_context.get('remaining',0),
                constructor_remaining=context.get('constructor_budget',{}).get('remaining_evaluations'),
                reselections_remaining=max(0,(2 if editor.constraints.get('plan_options') else 1)-editor.constraints.get('object_reselection_count',0)),
                inspected=delivered)
            if closing:
                turn_tools=[tool for tool in turn_tools if tool['function']['name'] in
                    {'inspect_trajectory','submit_candidate','probe_pdm','probe_vadv2','finish_search'}]
                for tool in turn_tools:
                    if tool['function']['name']=='submit_candidate':
                        tool['function']['parameters']['properties']['candidate_id']['enum']=[key for key in qualified if key in delivered]
                turn_tools=[tool for tool in turn_tools if tool['function']['name']!='submit_candidate'
                    or tool['function']['parameters']['properties']['candidate_id']['enum']]
            from agent_tool_view import creation_contract
            choice, turn_tools = creation_contract(turn_tools, feedback)
            if fast_stationary:
                name = 'submit_candidate' if 'seed_000' in delivered else 'inspect_trajectory'
                turn_tools = [tool for tool in turn_tools if tool['function']['name']==name]
                choice = {'type':'function', 'function':{'name':name}}
            elif choice is None:
                choice = 'auto'
            turn_names = {tool['function']['name'] for tool in turn_tools}
            if (not closing and turn == 1 and not imported_count and not editor.constraints.get('stationary_target')
                    and not editor.constraints.get('agent_decides_interaction') and 'construct_attack_candidates' in turn_names):
                choice = {'type':'function', 'function':{'name':'construct_attack_candidates'}}
            elif turn == 1 and imported_count and not fast_stationary:
                choice = {'type':'function', 'function':{'name':'inspect_trajectory'}}
            if isinstance(choice, dict):
                # The API may use auto for thinking; the selected phase remains exact locally.
                turn_names = {choice['function']['name']}
            print('Trajectory editing turn', turn, flush=True)
            _, attempt = archive.complete(f'edit_{turn:02d}', [{'role':'user', 'content':content}],
                total_seconds=min(timeout_seconds, max(1, remaining)),
                no_output_seconds=min(60, max(1, remaining)), retries=2, max_tokens=max_tokens,
                extra_body={'thinking':{'type':'enabled'}}, reasoning_effort=reasoning_effort(),
                tools=turn_tools, tool_choice=choice, parallel_tool_calls=False, allow_tool_calls=True)
            if pending is not None:
                for record in pending_records:
                    if record.get('purpose') not in ('pdm_probe','vadv2_probe','scene_layout'):
                        editor.inspected.add(record['candidate_id'])
                    deliveries.append(dict(candidate_id=record['candidate_id'],purpose=record.get('purpose'),request_path=str(attempt), turn=turn))
                pending = None
            call = None
            name = 'none'
            tool_started = perf_counter()
            try:
                call, name, args = decode_call(attempt)
                if name not in turn_names:
                    raise ValueError('Tool unavailable for the current candidate eligibility or resource state')
                if name in ('inspect_trajectory','evaluate_candidate','probe_pdm','probe_vadv2'):
                    schema = next(tool['function']['parameters'] for tool in turn_tools if tool['function']['name']==name)
                    if args.get('candidate_id') not in schema['properties']['candidate_id']['enum']:
                        raise ValueError('Candidate is not eligible for this tool in the current state')
                if name not in ('inspect_trajectory', 'evaluate_candidate', 'probe_pdm','probe_vadv2') and (not isinstance(args.get('reason'), str) or not args['reason'].strip()):
                    raise ValueError('A nonempty action reason is required')
                if name == 'evaluate_candidate' and set(args) == {'candidate_id'}:
                    feedback = editor.evaluate_candidate(args['candidate_id'])
                    evaluated_candidates.add(args['candidate_id'])
                elif name == 'probe_vadv2' and set(args)=={'candidate_id'}:
                    feedback=editor.policy_probe.probe(args['candidate_id'])
                    if feedback.get('status')=='completed':
                        pending=dict(candidate_id=args['candidate_id'],path=feedback['overlay_path'],purpose='vadv2_probe')
                elif name == 'probe_pdm' and set(args)=={'candidate_id'}:
                    feedback=editor.pdm_probe.probe(args['candidate_id'])
                    if feedback.get('status')=='completed':
                        pending=dict(candidate_id=args['candidate_id'],path=feedback['overlay_path'],purpose='pdm_probe')
                elif name == 'construct_attack_candidates':
                    prevalidate_action(name, args, editor)
                    feedback = editor.construct_pool(args)
                    candidates=feedback.get('candidates',[])
                    review=[row for row in candidates if row.get('pdm_probe_eligible')] or candidates
                    records=[editor.inspect(row['candidate_id']) for row in review[:2]]
                    if records:
                        pending=dict(batch=records)
                elif name == 'construct_trajectory' and set(args) == {'points', 'end_velocity_mps', 'reason'}:
                    feedback = editor.construct(args)
                elif name in ('edit_speed_profile','edit_route'):
                    prevalidate_action(name, args, editor)
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
                    if editor.policy_probe is not None:
                        editor.policy_probe.require_reselection_evidence()
                    if editor.pdm_probe is not None:
                        editor.pdm_probe.require_reselection_evidence()
                    if editor.constraints.get('object_reselection_count', 0) >= (2 if editor.constraints.get('plan_options') else 1):
                        raise ValueError('Object reselection budget exhausted')
                    feedback = dict(status='object_reselection_requested', reason=args['reason'])
                    stop = 'object_reselection_requested'
                elif name == 'finish_search' and set(args) == {'reason'}:
                    feedback = dict(status='finished', reason=args['reason'])
                    stop = 'finished_without_submission'
                else:
                    raise ValueError('Unknown tool or invalid arguments')
            except (ValueError, KeyError, TypeError) as error:
                feedback = invalid_action_result(name, args if 'args' in locals() else {}, error)
            archive_exchange(folder, turn, call or _undecoded_call(turn), feedback, started=tool_started)
            previous_failure = repetition(previous_failure, name, feedback)
            feedback['archive_ref'] = str(folder/'tool_calls'/f'turn_{turn:02d}')
            feedback = compact_action_feedback(feedback)
            print('Tool', name, 'returned', feedback.get('status', feedback.get('reference_screen_passed')), flush=True)
            if previous_failure is not None:
                signature, count, _ = previous_failure
                if (signature[0] == 'invalid' and count >= 2) or (signature[0] != 'invalid' and count >= 3):
                    stop = 'repeated_failure'
                    termination = dict(reason='Repeated failure without measurable improvement; stopping early.',
                        failure=signature,repetitions=count,scope='current_tools_and_budget',global_infeasibility_proven=False)
                    break
            if editor.submitted or stop in ('finished_without_submission', 'object_reselection_requested'):
                break
            if editor.policy_probe is not None:
                state=editor.policy_probe.context()
                if state['remaining']==0 and not any(item.get('policy_attack_success') is True for item in state['candidates'].values()):
                    stop='vadv2_probe_budget_exhausted_without_target_contact'
                    termination=dict(reason='Cumulative VADv2 budget exhausted without observed post-takeover target contact.',
                        scope='current_tools_and_budget',global_infeasibility_proven=False,
                        calls=state['calls'],latest_feedback=state['latest_feedback'])
                    break
            if name == 'construct_attack_candidates':
                termination = constructor_exhaustion(editor.nodes,editor.context()['constructor_budget'])
                if termination is not None:
                    stop = termination['stop_reason']
                    break
    except AttackBudgetExhausted as error:
        stop='cumulative_attack_api_budget_exhausted'
        termination=dict(reason=str(error),scope='main_agent_execution_scope',global_infeasibility_proven=False)
    except Exception as error:
        timed_out=(getattr(error,'failure_kind',None)=='request_wall_time_exceeded'
            or type(error).__name__ in {'TimeoutExpired','TimeoutError','APITimeoutError','RequestDeadlineExceeded'})
        if timed_out:
            stop = 'stage_timeout_exhausted' if perf_counter() >= deadline-1 else 'request_timeout_exhausted'
            termination = dict(reason=('Attack trajectory stage exceeded its wall-clock budget.' if stop=='stage_timeout_exhausted' else 'A model request exceeded its wall-clock budget; saved candidates remain available.'),
                stage_timeout_seconds=stage_timeout_s,scope='current_tools_and_budget',global_infeasibility_proven=False,
                failure_kind=getattr(error,'failure_kind',None),archive_ref=getattr(error,'archive_ref',None))
        else:
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
            api_budget=archive.context() if isinstance(archive,BudgetedArchive) else None,
            resumable_candidate_ids=[key for key,node in editor.nodes.items() if candidate_eligibility(node['result'])['pdm_probe_eligible']],
            finalization_grace_used=finalization_grace_used,
            annotation_changes=dict(operation='export_target_trajectory', frames_json_modified=False),
            physical_collision='undetermined'))
    return editor.submitted


def reselect_object(scene,output,instruction,assets,config,editor,summary,reason,
                    timeout_seconds,max_tokens,max_turns,required_operation,required_track):
    from attack_plan_options import reselection_limit
    used=len(list((editor.run_dir/'tool_calls').glob('turn_*')))
    remaining_turns=max_turns-used
    count=config.get('object_reselection_count',0)
    if remaining_turns<1 or count>=reselection_limit(config):
        return dict(status='not_submitted',stop_reason='object_reselection_budget_exhausted',
            reason='Object switching or editing action budget exhausted; no accepted scene was produced.')
    feedback=list(config.get('object_selection_feedback',[]))
    feedback.append(dict(operation=editor.target['operation'],target_track=editor.target['track_id'],
        reason=str(reason)[:700],failed_checks=editor.nodes['seed_000']['result'].get('failed_checks',[])))
    revised=dict(config,object_reselection_count=count+1,object_reselection_reason=reason,
        object_selection_feedback=feedback,
        inherited_constructor_budget={k:editor.context()['constructor_budget'][k] for k in ('calls','evaluations')})
    if editor.pdm_probe is not None:
        latest=editor.pdm_probe.context()['latest_feedback']
        if latest is not None:revised['pdm_reselection_feedback']=latest
    if editor.policy_probe is not None:
        latest=editor.policy_probe.context()['latest_feedback']
        if latest is not None:revised['revision_feedback']=dict(stage='native_vadv2',outcome='target_not_hit',evidence=latest,
            guidance='Revise the original task using actual VADv2 collision feedback')
    followup=output/f'object_revision_{count+1:02d}'
    child=run(scene,followup,instruction,assets,revised,timeout_seconds,max_tokens,remaining_turns,required_operation,required_track)
    return dict(status='continued_in_object_revision',followup_run=str(followup),followup_status=child['status'])


@timed_operation('attack.workflow', 'output')
@shared_auto_session
def run(scene, output, instruction, assets=None, constraints=None, timeout_seconds=480,
        max_tokens=12288, max_turns=6, required_operation=None, required_track=None, continue_from=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if (experiments_root()).resolve() not in output.parents or output.exists():
        raise ValueError('Workflow output must be a new directory under experiments')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('A nonempty user instruction is required')
    config = normalize_constraints(constraints)
    if continue_from is not None and config.get('plan_options'):
        prior_path=Path(continue_from)/'input.json'
        if prior_path.is_file():
            saved=json.loads(prior_path.read_text()).get('constraints',{})
            if not same_scene_constraints(saved,config):raise ValueError('Continuation constraints differ')
            for key in ('object_reselection_count','object_reselection_reason','object_selection_feedback'):
                if key in saved:config[key]=saved[key]
    if config.get('vadv2_self_test'):
        if config.get('pdm_self_check'):raise ValueError('Choose VADv2 self-test or legacy PDM, not both')
        config.setdefault('vadv2_probe_root',str(output/'vadv2-probes'))
        count=config.get('vadv2_probe_budget',3)
        if type(count) is not int or not 1<=count<=5:raise ValueError('VADv2 probe budget must be between 1 and 5')
        profile=config['vadv2_policy_config']
        for key in ('python','config_path','checkpoint_path'):
            if not Path(profile[key]).is_file():raise FileNotFoundError('VADv2 self-test missing '+key)
    if config.get('pdm_self_check'):
        config.setdefault('pdm_probe_root',str(output/'pdm-probes'))
        if type(config.get('pdm_probe_budget',2)) is not int or config.get('pdm_probe_budget',2)<1:
            raise ValueError('PDM probe budget must be a positive integer')
    stage_timeout = config.get('attack_stage_timeout_s', 420)
    if type(stage_timeout) not in (int, float) or stage_timeout <= 0:
        raise ValueError('attack_stage_timeout_s must be positive')
    effective = config['acceptance_policy']
    frames = json.loads((scene / 'metadata/frames.json').read_text())
    times, _, _ = ego_path_from_frames(frames)
    for key in ('observation_end_s', 'attack_prefix_lock_s', 'official_margin_m', 'max_ego_clearance_m', 'min_travel_m'):
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not 0. <= config['observation_end_s'] < times[-1]:
        raise ValueError('WE observation horizon must fit the scene; it is not an edit boundary')
    if not times[0] <= config['attack_prefix_lock_s'] < times[-1]:
        raise ValueError('Attack prefix lock must fit the scene')
    if effective['policy_version']=='reference_acceptance.v2' and config.get('pdm_self_check'):
        raise ValueError('Internal PDM is disabled for new editing runs; Main Agent calls policy_runner(native) after defense editing')
    if required_operation not in {None, 'add_vehicle', 'edit_existing'}:
        raise ValueError('Unknown object operation requirement')
    if type(max_turns) is not int or max_turns < 1 or timeout_seconds <= 0:
        raise ValueError('Positive action and timeout budgets are required')
    if 'event_window_s' in config:
        window = np.asarray(config['event_window_s'], dtype=float)
        if window.shape != (2,) or not np.isfinite(window).all() or not times[0] <= window[0] < window[1] <= times[-1] or window[1] <= policy.danger_start(effective):
            raise ValueError('Event window must be ordered, inside the scene and extend after observation')
    if config.get('task_contract') is not None:
        from attack_task_contract import validate_task_contract
        if continue_from is None and 'schema_version' not in config['task_contract']:
            raise ValueError('Task contracts require attack_task.v2 provenance')
        validate_task_contract(config['task_contract'], times, policy.danger_start(effective),edit_start_s=config['attack_prefix_lock_s'])
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
    from attack_budget import AUTO_SESSION_ROOT
    atomic_json(output / 'input.json', dict(schema_version='attack_workflow.v1', scene=str(scene),
        auto_session_root=AUTO_SESSION_ROOT.get(),
        instruction=instruction, assets=assets, constraints=config, attack_difficulty=difficulty, required_operation=required_operation,
        required_track=required_track, reasoning_effort=reasoning_effort(), timeout_seconds=timeout_seconds,
        max_tokens=max_tokens, max_turns=max_turns,
        continued_from=None if continue_from is None else str(Path(continue_from).resolve())))
    profile = load_llm_profile()
    client = OpenAI(api_key=profile['api_key'], base_url=profile['base_url'], max_retries=0,
        http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=float(timeout_seconds)))
    archive = BudgetedArchive(StreamingRequestArchive(output / 'requests', client, profile),output)
    summary = dict(schema_version='attack_workflow.v1', status='failed', physical_collision='undetermined',attack_difficulty=difficulty)
    stage = 'object_selection'
    try:
        from attack_plan_options import selection_seconds
        selection_timeout=selection_seconds(timeout_seconds) if config.get('plan_options') else timeout_seconds
        if selection_timeout<=0:
            record=dict(status='not_submitted',stop_reason='stage_timeout_exhausted',search_stage='object_selection',
                reason='Shared attack editing wall-clock budget exhausted before another object choice.')
            (output/'object_selection').mkdir(exist_ok=True)
            atomic_json(output/'object_selection/output.json',record)
            summary.update(record)
            return summary
        if continue_from is None:
            selection = select_object(scene, output / 'object_selection', instruction, assets, config,
                archive, selection_timeout, max_tokens, required_operation, required_track)
        else:
            import shutil
            previous = Path(continue_from).resolve()
            continuation_root=Path(os.getenv('ADVERSARIAL_CONTINUATION_ROOT',str(experiments_root()))).resolve()
            if previous == output or continuation_root not in previous.parents:
                raise ValueError('Continuation source must be another experiments run')
            prior_input = json.loads((previous / 'input.json').read_text())
            if Path(prior_input['scene']).resolve() != scene or not same_scene_constraints(prior_input['constraints'],config) or prior_input['instruction'] != instruction or prior_input['assets'] != assets or prior_input.get('required_operation') != required_operation or prior_input.get('required_track') != required_track:
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
        summary['plan_choice']=selection.get('plan_choice')
        if config.get('plan_options'):
            from attack_plan_options import editor_deadline
            editor_deadline(stage_timeout)
        precheck = initial_state_precheck(editor)
        if precheck is not None:
            atomic_json(editor.run_dir/'output.json', dict(schema_version='attack_trajectory.v1',
                status='not_submitted',selected_candidate_id=None,target_track=editor.target['track_id'],
                candidate_count=1,**precheck))
            summary.update(status='not_submitted',target=selection['target'],selected_candidate_id=None,
                object_output=str(output/'object_selection/output.json'),
                trajectory_output=str(editor.run_dir/'output.json'),initial_motion_precheck=precheck)
            if config.get('plan_options'):
                summary.update(reselect_object(scene,output,instruction,assets,config,editor,summary,precheck['reason'],
                    timeout_seconds,max_tokens,max_turns,required_operation,required_track))
            return summary
        if continue_from is not None:
            for name in ('constructor-budget.json', 'construction_pools'):
                source = previous / 'trajectory_editing' / name
                if source.is_dir():
                    shutil.copytree(source, editor.run_dir / name)
                elif source.is_file():
                    destination=constructor_budget_path(editor.run_dir)
                    if not destination.exists():
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(source,destination)
            for candidate in sorted((previous / 'trajectory_editing/candidates').glob('candidate_*')):
                if not (candidate / 'operation.json').is_file() or not (candidate / 'trajectory.csv').is_file():
                    continue
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
            destination=constructor_budget_path(editor.run_dir)
            if not destination.exists(): atomic_json(destination, inherited)
        selected = edit_object(editor, selection, archive, timeout_seconds, max_tokens, max_turns,
            stage_timeout_s=stage_timeout)
        if editor.pdm_probe is not None:
            summary['pdm_probe_feedback']=editor.pdm_probe.context()
            if selected is not None:
                summary['pdm_probe_result_path']=editor.pdm_probe.require_challenge(selected)['result_path']
        if editor.policy_probe is not None:
            summary['vadv2_probe_feedback']=editor.policy_probe.context()
            if selected is not None:
                summary['vadv2_probe_result_path']=editor.policy_probe.require_challenge(selected)['result_path']
        summary.update(status='not_submitted' if selected is None else 'reference_candidate_submitted',
            plan_choice=selection.get('plan_choice'),target=selection['target'], selected_candidate_id=selected,
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
    except AttackBudgetExhausted as error:
        summary.update(status='not_submitted',stop_reason='cumulative_attack_api_budget_exhausted',
            reason=str(error),search_stage=stage,api_budget=archive.context())
        atomic_json(output/stage/'output.json',dict(summary))
        return summary
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
        reason=stage_result.get('reason')
        if not reason:
            reason=json.loads(sorted((editor.run_dir/'tool_calls').glob('turn_*/result.json'))[-1].read_text())['reason']
        summary.update(reselect_object(scene,output,instruction,assets,config,editor,summary,reason,
            timeout_seconds,max_tokens,max_turns,required_operation,required_track))
        atomic_json(output/'output.json',summary)

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
