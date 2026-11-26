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
from ego_road_checks import load_official_drivable
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
from trajectory_waypoints import SCHEMA as WAYPOINTS
from connector_route import SCHEMA as CONNECTOR_ROUTE


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
CONNECTOR_PREVIEW_TOOL = function_tool('preview_connector_route',
    'Get initial pose, entry/exit lanes and nominal arrival times for a mapped connector at the requested distances and speed. '
    'This only reads the map and does not check placement or create a candidate.',
    {'connector_route': CONNECTOR_ROUTE, 'reason': REASON})


def placement_preview_tool():
    common={'option_id':{'type':'string'},'operation':{'type':'string'}}
    existing=dict(type='object',properties=dict(common,operation={'const':'edit_existing'},track_id={'type':'string'}),
        required=['option_id','operation','track_id'],additionalProperties=False)
    added=dict(type='object',properties=dict(common,operation={'const':'add_vehicle'},asset_id={'type':'string'},
        x_m=NUMBER,y_m=NUMBER,yaw_rad=NUMBER,speed_mps=NUMBER),
        required=['option_id','operation','asset_id','x_m','y_m','yaw_rad','speed_mps'],additionalProperties=False)
    return function_tool('preview_attack_options',
        'Compare1..3 unranked initial vehicle/placement hypotheses in one cheap geometry/reaction preview. '
        'Creates no candidate and runs no VAD; estimates cannot establish attack success or checked expert feasibility. '
        'Use select_vehicle/add_vehicle afterward to commit the chosen exact initial state. At most6 evaluated hypotheses across this attack scope.',
        {'options':dict(type='array',minItems=1,maxItems=3,items={'oneOf':[existing,added]}),'reason':REASON})
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
POSE_OFFSET=dict(type='object',properties={'forward_m':NUMBER,'left_m':NUMBER},
    required=['forward_m','left_m'],additionalProperties=False)
POSE_SEARCH=dict(type='object',properties={
    'forward_offsets_m':dict(type='array',items=NUMBER,minItems=1,maxItems=4),
    'left_offsets_m':dict(type='array',items=NUMBER,minItems=1,maxItems=4),
    'yaw_mode':dict(type='string',enum=['preserve','route_aligned','offset','explicit']),
    'yaw_offset_rad':NUMBER,'yaw_rad':NUMBER,'initial_speed_mps':NUMBER},additionalProperties=False)
KEYFRAME = dict(type='object', properties={
    'time_window_s':INTERVAL, 'hold_duration_s':NUMBER,
    'relative_position':dict(type='object', properties={'forward_m':INTERVAL, 'left_m':INTERVAL}, additionalProperties=False),
    'heading_difference_rad':INTERVAL, 'speed_mps':INTERVAL},
    required=['time_window_s', 'hold_duration_s'], additionalProperties=False)
TRAJECTORY_TOOLS.insert(0, function_tool('construct_attack_candidates',
    'Optimize a parent acceleration/steering control sequence with continuous official-road and observed-obstacle guidance. '
    'Declare body-center position intervals in the recorded ego BODY-center frame, heading and/or speed keyframes. '
    'For scene-local path goals use objective.waypoints: start, optional intermediate points, and destination. '
    'Alternatively set connector_route to generate those waypoints from a mapped entry lane, connector and exit lane. '
    'Supply only one of connector_route and objective.waypoints. Keyframes may still express relative motion preferences. '
    'Each has position_m=[x,y], one of time_s/time_window_s, and optional speed_mps and yaw_rad. '
    'Set explicit tolerances when needed (defaults: 0.5 m, 0.5 m/s, 0.25 rad). A missed waypoint is reported and cannot qualify the candidate. '
    'Supports stationary/slow lead, merging, crossing or opposite-heading objectives without maneuver templates. '
    'When using connector_route, omit objective.waypoints (map connector generates waypoints automatically). '
    'New first calls use seed_000. Further calls truly start from parent controls. '
    'Optional edit_window_s limits changed control knots; preserve_outside_window additionally requires reconnection '
    '(outside position<=0.05m, yaw<=0.05rad, velocity<=0.1m/s); failure is reported, never spliced away. '
    'By default one call automatically mixes bounded initial-position/yaw/speed variants, conflict-corridor seeds and continuous controls under the same budget; no search mode selection is needed. '
    'Optional initial_pose_search is an explicit reproducible override for up to6 initial-position/yaw/speed variants. '
    'No interaction category or fixed proximity gate is required. Review measured geometry against the task. '
    'reason is explanatory only: all numeric requirements must appear in objective. '
    'Returns fully checked candidates separately from optimization guidance. Motion may be edited from edit_start_s (0s in new runs). No implicit 3-second event delay applies; respect explicit task windows and physical/map checks.',
    {'parent_candidate_id':{'type':'string'},
     'objective':dict(type='object', properties={
         'keyframes':dict(type='array', items=KEYFRAME, minItems=0, maxItems=6),
         'waypoints':WAYPOINTS},
         required=[],additionalProperties=False),
     'connector_route':CONNECTOR_ROUTE, 'edit_window_s':INTERVAL, 'preserve_outside_window':{'type':'boolean'},
     'initial_pose_search':POSE_SEARCH,
     'max_candidates':dict(type='integer',minimum=1,maximum=3), 'reason':REASON},
    required=['parent_candidate_id','objective','max_candidates','reason']))
TRAJECTORY_TOOLS += [copy.deepcopy(tool) for tool in EGO_TOOLS if tool['function']['name'] in {
    'edit_speed_profile', 'edit_route', 'repair_candidate', 'submit_candidate', 'finish_search'}]
for tool in TRAJECTORY_TOOLS:
    if tool['function']['name']=='edit_speed_profile':
        tool['function']['description']+=' In this attack stage the unchanged bicycle policy rejects reverse travel. Generator support for signed speeds does not override that policy.'
TRAJECTORY_TOOLS += [function_tool('inspect_trajectory', 'View a candidate BEV and reference checks before submitting it. Repeated inspection returns the cached image and evidence without rendering again.',
                                  {'candidate_id':{'type':'string'}})]
TRAJECTORY_TOOLS += [function_tool('adjust_initial_pose',
    'Create a new candidate version with bounded initial-position, optional initial yaw and optional initial-speed adjustments. Existing and added actors share scene-local x-forward/y-left offsets. preserve keeps yaw; route_aligned uses the nearest recorded ego-route heading; offset adds yaw_offset_rad; explicit sets yaw_rad. Moving actors regenerate their trajectory from inherited controls; stationary actors are rigidly transformed and must keep zero speed. Full checks and exact-version VAD testing are required again.',
    {'parent_candidate_id':{'type':'string'},'position_offset_m':POSE_OFFSET,
     'yaw_mode':dict(type='string',enum=['preserve','route_aligned','offset','explicit']),
     'yaw_offset_rad':NUMBER,'yaw_rad':NUMBER,'initial_speed_mps':NUMBER,'reason':REASON},
    required=['parent_candidate_id','position_offset_m','reason'])]

OBJECT_PROMPT = """Select the object for the requested scene edit. Obey the user's instruction: use an
existing vehicle when requested, or add a vehicle when requested. You choose between the available
tools; no stationary/cut-in mode is preselected. The next stage receives your object and edits its
trajectory using generic tools. For a new object choose a plausible initial center, yaw and speed;
this straight-motion seed is an initial proposal that may be edited from the first editable time (0s in
current runs); this selection fixes the starting pose until a later explicit initial-pose adjustment. Choose a safe
initial pose with space for a valid early trajectory. Final candidates must pass their explicit task, road and motion checks.
Stage A prepares and freezes ego history before this selection. Read fixed_ego_history: the
actual takeover position, speed and heading are known. Choose/edit the OTHER actor only. Do not
attach ego speed commands here. The post-1.5s reference path is not what VAD must drive; use the
actual takeover state, not an assumed later recorded acceleration, to judge the challenge.
When plan_options are supplied, compare them once and choose a promising existing or added actor;
option order is not a quality rank. You may adapt an option within the unchanged scene mechanism.
Use optional option_id and explain the actual choice/deviation in reason. Do not run all alternatives.
previous_object_feedback contains real failures: reconsider unsupported targets and use an added
vehicle when that resolves a placement or inherited-state problem. Only explicit required_operation
and required_track are hard object requirements. Existing t=0 poses are fixed during this selection step; a later explicit
initial-pose adjustment requires a selected object. Rejected initial placements identify their actual actor,
operation and measured check. Choose an available actor or a different proposed placement after a rejection. A newly
placed stationary car should use add_vehicle with speed_mps=0 and a supported initial pose.
Coordinates are metres in the initial ego frame: x forward, y left, yaw counterclockwise radians.
When the plan names an other-actor connector, preview_connector_route can provide the entry lane,
initial pose and approximate arrival times. Read nominal_connector_entry_s and nominal_connector_exit_s
against the measured ego reference and event window before committing the initial placement.
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
GOALS: explicit caller task_contract and configured event window are hard. New runs require sampled reference target-body contact inside the event window for constructor/VAD eligibility; near-miss alone is insufficient. New attack events default to 1.5–7.5s;
editing/braking/steering may start at 0s. Agent-generated keyframes are optimization preferences unless an
explicit contract makes them required. Read reference_contact_evidence and actual measurements; reference contact is only setup evidence, not proof that VAD will collide. Do not change caller requirements.
TOOLS: first use construct_attack_candidates with optional absolute waypoints or relative motion keyframes.
For a route through a named connector, set connector_route (connector_id, approach_m, exit_m, speed_mps)
instead of explicit waypoints. Use the same route parameters as the placement preview when applicable.
For an observed vehicle choose an approach distance consistent with its current position on the entry lane.
Connector passage measures vehicle centers in native frames. Motion, road and other checks still apply.
Only structured fields guide construction; reason is a brief explanation. Every hold_duration_s must fit its
own time_window_s. Use edit_speed_profile for signed speed targets; use edit_route for smooth lateral offsets
and return. Select an objective-bearing parent exposed by the tool enum. Preview checks do not create a candidate.
Read decision_summary and recommended_previews; an explicit goal revision is different from a numeric repair.
Use adjust_initial_pose for a bounded new initial-position/yaw version, or initial_pose_search in
construct_attack_candidates to compare a small set of initial poses within the same cumulative budget.
Both existing and added actors use these tools; stationary actors move/rotate rigidly, while moving actors
regenerate their trajectory from inherited controls. Every changed initial pose requires all checks and a new VAD test.
Body-relative goal positions use ego BODY center; scene coordinates use initial ego forward/left. Center
separation is not bumper clearance. Reference contact is required for new-run eligibility. The exact-candidate VAD rollout checks target collision and drivable-area compliance. Do not make a stopped car slide or rotate in place.
CHECK: physical/map checks and native-replay checks must pass. Generated and replayed paths can differ, so
use both reports. Speed limits here are a named research profile, not certified vehicle capabilities.
REVIEW: initial layout and newly constructed candidate pictures are supplied automatically. Inspect another
candidate when needed. The diagram must actually be delivered before submission; a policy-test overlay or
layout-only BEV does not satisfy that review. No unnecessary edit is required after the task is satisfied.
If policy self-test is enabled, complete its exact-candidate test. Report sampled target collision and
drivable-area violation separately. When policy tests are disabled, report reference validity only.
BUDGET: read cumulative_api_budget, constructor_budget and stage_seconds_remaining. In finalization_only,
choose an existing qualified candidate for review/submission or finish_search; do not restart construction.
A new tool call cannot reset cumulative budgets. If no candidate can meet the task, finish honestly with the
measured cause. Lateral intrusion strategy: prefer shallow lateral encroachment into ego's lane (encroaching across the lane edge / clipping ego's near side by ~0.2-0.5m) rather than a full-body total intrusion that completely blocks ego's entire lane, so ego still has a feasible avoidance margin while facing a clear conflict. As soon as at least one candidate has reference_screen_passed=true, inspect and call submit_candidate on the best passing candidate instead of spending extra turns constructing another pool (closed-loop VADv2 verification is handled downstream by policy_runner). Use one exposed native tool call and a short reason per turn."""


STATIC_EDIT_PROMPT = """Validate the selected stationary placement. Only the selected target is added or
edited; ego and observed others replay their records. Within each candidate version the target pose is fixed for the entire scene.
Use adjust_initial_pose to create a new rigidly translated/rotated version when a different placement is justified; full checks and any VAD evidence restart for that version.
Read seed_000's reference checks, fixed task contract and geometry_feedback. Use measured body overlap
times, never mental center-arrival estimates. No implicit three-second event constraint applies; use the explicit task interval. Do not invent a narrower hard window, hold duration or reaction delay.
Only explicit task requirements and named checking policy are hard; read provenance and uncertainties.
If pdm_self_check is enabled, passing geometry only makes the placement eligible for probe_pdm.
Test first; a safely defended placement needs reselection or an honest finish. A tested PDM-failing
placement must still use inspect_trajectory and receive the image before submit_candidate(seed_000).
Only when no policy self-test is enabled do geometry plus trajectory inspection alone permit submission.
No motion construction or speed/route editing is available for a static candidate. evaluate_candidate reads existing
measurements without optimization. A failed placement can use adjust_initial_pose or request_object_reselection within the shared budgets; otherwise finish_search with measured reasons.
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
and task checks, including required in-window reference target-body contact for new runs, then call probe_vadv2.
A target collision or a confirmed drivable-area violation by actual VADv2 after takeover qualifies as
a policy failure. Use policy_evaluation.failure_reasons and the exact candidate version for submission.
Use min_target_clearance_m, closest_approach time,
policy_behavior_summary (stopped_detected, progress_ratio_vs_recorded, signed_yaw_change_deg),
forward/left/closing_speed and the plot to understand braking/early-stop or steering avoidance.
Treat missing road coverage as inconclusive and use only actual sampled post-takeover contact.
The test uses 1.5–7.5s with the same model, cameras and native controller as evaluation; videos,
official PDM and training exports are skipped. Tests share vadv2_self_test.remaining across
edits/reselection. Simulation wait does not consume stage_seconds_remaining; whole-tool timeout
still applies. Do not reject a new test solely because its earlier elapsed_s exceeds stage_seconds_remaining.
Yaw change or target-left offset alone does not prove avoidance in a turning scene; compare actual path and progress.
Use compact feedback and the plot: early stopping far away suggests checking exposure,
placement and event timing; steering avoidance suggests checking where the actual ego path went.
These are hypotheses, not fixed parameter changes. Stage B keeps the stage-A history fixed.
If actual VAD feedback identifies a starting-state problem, request_ego_history_revision may return
ONCE to preparation with target_speed_mps at 1.5s. It is separate from target-vehicle editing and
shares switching/action/API/probe budgets. The new scene invalidates old policy evidence.
An explicitly supplied caller ego target remains fixed. Coarse options are suggestions: adjust timing,
position or motion, or switch actor/operation when justified. An untested alternative cannot be
ruled out solely because it seems harder to construct. Only revise or switch when current evidence does not qualify.
Keep the original task and physical/map checks. After a successful VAD test, inspect_trajectory still
supplies attack/road evidence before submission. Main Agent reuses this exact-version evidence.'''

TRAJECTORY_TOOLS.append(function_tool('probe_vadv2',
    'Run an eligible candidate through WE + VADv2. Returns target collision, drivable-area compliance, '
    'closest approach and an image. Minute-scale bounded test; exact-version repeats reuse evidence.',
    {'candidate_id':{'type':'string'}}))

TRAJECTORY_TOOLS.append(function_tool('probe_pdm',
    'Test an eligible candidate with real PDM + LQR/kinematic closed loop. Returns actual trajectory '
    'samples, closest approach and an overlay image. Shared simulation budget; repeats reuse the same candidate result.',
    {'candidate_id':{'type':'string'}}))

TRAJECTORY_TOOLS.append(function_tool('evaluate_candidate',
    'Return measured full-horizon reference geometry and task checks for an existing candidate. No optimization. Repeated evaluation returns cached evidence.',
    {'candidate_id':{'type':'string'}}))
EDIT_PROMPT += (' When plan_options are present, plan_choice describes the chosen suggestion, not another hard contract. '
    'If inherited state or fixed t=0 placement makes this actor unsuitable, request_object_reselection rather than repeatedly constructing with the same unsupported state. '
    'The next object-selection step can change to another real vehicle or an added vehicle. Switching shares API, construction, action and stage-time budgets; do not reset them. '
    'A failed candidate can still be repaired; switching is an option, not a requirement to abandon it.')

TRAJECTORY_TOOLS.append(function_tool('request_object_reselection', 'Return to object selection within the remaining switching budget when the current actor/placement is unsuitable. Auto alternatives may change existing/added operation; explicit caller locks stay binding. Retain spent API, construction, action and policy-test budgets.', {'reason':REASON}))
TRAJECTORY_TOOLS.append(function_tool('request_ego_history_revision',
    'Return once to stage A with a new ego target speed at 1.5s, then restart attack editing on that frozen history. '
    'Only for a justified starting-state problem; all API/construction/action/probe budgets remain shared. '
    'Use actual VAD feedback; do not edit ego jointly with a target-vehicle action.',
    {'target_speed_mps':{'type':'number','minimum':0},'reason':REASON}))

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
    from backend_hooks import record_tool_exchange
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
    from attack_scene_evidence import initial_road_failure
    from connector_route import nearby_connectors, plan_connector_route
    options=constraints.get('plan_options',[])
    frames = json.loads((scene / 'metadata/frames.json').read_text())
    times, ego, yaw = ego_path_from_frames(frames)
    actors = observed_objects_from_frames(frames)
    road = load_official_drivable(scene / 'metadata/map_drivable_official.json')
    common = set(actor['track'] for actor in actors[0] if actor['label'] == 'vehicle')
    for frame in actors:
        common &= {actor['track'] for actor in frame if actor['label'] == 'vehicle'}
    proposed_tracks = {option.get('target_track') for option in options if option['operation'] == 'edit_existing'}
    if required_track is not None:
        proposed_tracks.add(required_track)
    rejections = {}
    attempted_rejections = set()
    margin = constraints.get('official_margin_m', .1)
    catalog = []
    for track in sorted(common):
        sequence = [next(actor for actor in frame if actor['track'] == track) for frame in actors]
        if track in proposed_tracks and required_operation != 'add_vehicle':
            first = sequence[0]
            failure = initial_road_failure(times[:1], np.array([[first['x'], first['y']]]),
                [first['yaw']], (first['length'], first['width']), road, margin)
            if failure is not None:
                rejections['select_vehicle', track] = dict(failure, operation='edit_existing', track_id=track)
        catalog.append(dict(track_id=track,
            poses=[dict(time_s=float(times[index]), xy_m=[sequence[index]['x'], sequence[index]['y']],
                        yaw_rad=sequence[index]['yaw'])
                   for index in sorted(set([0, len(times) // 2, len(times) - 1]))]))
    ego_speeds = np.linalg.norm(np.diff(ego, axis=0), axis=1) / np.diff(times)
    map_connectors = nearby_connectors(scene, frames, ego)
    road_height_bounds = None
    try:
        from road_height import RoadHeightMap, read_recon_translation
        rhm = RoadHeightMap.load(scene)
        recon = read_recon_translation(scene)
        valid_mask = np.isfinite(rhm.heights)
        if valid_mask.any():
            py, px = np.nonzero(valid_mask)
            pts_asset = (np.column_stack([px, py]) / rhm.scale - rhm.translation) @ rhm.rotation
            pts_world = pts_asset + recon[:2]
            origin = np.asarray(frames[0]['ego2global'], float)
            yaw0 = float(np.arctan2(origin[1, 0], origin[0, 0]))
            rot_inv = np.array([[np.cos(yaw0), np.sin(yaw0)], [-np.sin(yaw0), np.cos(yaw0)]])
            pts_local = (pts_world - origin[:2, 3]) @ rot_inv.T
            road_height_bounds = dict(
                x_min_m=round(float(pts_local[:, 0].min()), 1),
                x_max_m=round(float(pts_local[:, 0].max()), 1),
                y_min_m=round(float(pts_local[:, 1].min()), 1),
                y_max_m=round(float(pts_local[:, 1].max()), 1),
                note='All added-vehicle placements must stay inside this reconstructed 3DGS road-height coverage.',
            )
            filtered_connectors = []
            for conn in map_connectors:
                pt = conn.get('entry_xy_m') or conn.get('center_xy_m') or conn.get('xy_m')
                if isinstance(pt, (list, tuple)) and len(pt) == 2:
                    if (road_height_bounds['x_min_m'] <= pt[0] <= road_height_bounds['x_max_m']
                            and road_height_bounds['y_min_m'] <= pt[1] <= road_height_bounds['y_max_m']):
                        filtered_connectors.append(conn)
                else:
                    filtered_connectors.append(conn)
            if filtered_connectors:
                map_connectors = filtered_connectors
    except Exception:
        pass
    context = dict(instruction=instruction, scene=str(scene), assets=assets,
        existing_vehicles=catalog, required_operation=required_operation, required_track=required_track,
        plan_options=options,previous_object_feedback=constraints.get('object_selection_feedback',[]),
        constraints=constraints,
        nearby_connectors=map_connectors,
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

    def finish(stop_reason, reason):
        result = dict(status='not_submitted', stop_reason=stop_reason, reason=reason,
            search_stage='object_selection', rejected_initial_placements=list(rejections.values()))
        atomic_json(folder/'output.json', result)
        return result

    locked_failure = rejections.get(('select_vehicle', required_track))
    if locked_failure is not None:
        return finish('required_initial_placement_rejected', locked_failure['error'])
    from decision_scene_view import render_overview, actor_view, GUIDANCE
    overview=render_overview(times,ego,yaw,actors,road,folder/'scene-overview.png',required_track,event_window_s=constraints.get('event_window_s'))
    decision_context=dict(context)
    decision_context.pop('actors_at_start',None)
    from attack_context_view import selection_view,feedback_view
    from attack_scene_evidence import conflict_context,render_conflict
    conflict=conflict_context(times,ego,yaw,constraints)
    conflict_map=render_conflict(times,ego,yaw,actors,road,folder/'conflict-map.png',constraints)
    atomic_json(folder/'conflict-evidence.json',conflict)
    decision_context['conflict_evidence']=conflict
    decision_context['placement_preview_budget']=dict(max_evaluated_hypotheses=6,scope='shared across attack object revisions and retries')
    decision_context['scene_layout']=actor_view(times,ego,actors,required_track)
    if constraints.get('reference_kinematics'):
        decision_context['reference_kinematics']=constraints['reference_kinematics']
    if constraints.get('spatial_hint'):
        decision_context['spatial_hint']=constraints['spatial_hint']
    if road_height_bounds is not None:
        decision_context['road_height_bounds_m']=road_height_bounds
    decision_context['visual_guidance']=GUIDANCE
    decision_context=selection_view(decision_context)
    feedback = None
    connector_preview = None
    for turn in range(1, 7):
        available = [row for row in catalog if ('select_vehicle', row['track_id']) not in rejections
                     and required_track in (None, row['track_id'])]
        object_tools = selection_tools(OBJECT_TOOLS, assets, required_operation, required_track, options)
        if not available:
            object_tools = [tool for tool in object_tools if tool['function']['name'] != 'select_vehicle']
        elif rejections or required_track is not None:
            for tool in object_tools:
                if tool['function']['name'] == 'select_vehicle':
                    tool['function']['parameters']['properties']['track_id']['enum'] = [row['track_id'] for row in available]
        object_tools.append(placement_preview_tool())
        if map_connectors:
            object_tools.append(CONNECTOR_PREVIEW_TOOL)
        if turn == 1:
            atomic_json(folder/'tools.json', object_tools)
        decision_context['existing_vehicles'] = available
        decision_context['rejected_initial_placements'] = list(rejections.values())
        preview_guidance=('Use conflict_evidence and the time-labelled map to locate interactions inside the event window. '
            'When spatial_hint.attack_start_xy_m is provided (green star start_hint on the map), prioritize placing add_vehicle naturally centered inside its own lane (e.g. oncoming or adjacent lane, aligned with lane direction, never straddling the ego lane or angled artificially) near start_hint inside road_height_bounds_m rather than spending turns probing distant connector IDs. '
            'When alternatives are uncertain, preview_attack_options compares up to3 hypotheses without VAD or candidate creation. '
            'Reference-path contact is not policy failure; assess slower/coasting/braking estimates without treating them as policy predictions. '
            'Preserve an expert defense opportunity; moving an obstacle arbitrarily close is not the objective. '
            'After a failed policy test, use measured fixed-coordinate scene_feedback, not body-relative left offset as a placement command. '
            'Choose from the unranked measurements; previews are optional, budgeted, and never waive full checks.')
        prompt=OBJECT_PROMPT + '\n'+preview_guidance+'\n' + json.dumps(
            compact_model_context(dict(context=decision_context, last_tool_result=feedback_view(feedback))), ensure_ascii=False, separators=(',', ':'))
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
        # Requests are rebuilt independently: a preview response must not drop
        # the only layout image from the following selection decision.
        if isinstance(content,str): content=[{'type':'text','text':content}]
        ref_bev = constraints.get('reference_bev_image')
        extra_images = []
        if ref_bev and Path(ref_bev).is_file():
            extra_images.append({'type':'text','text':'Reference failure BEV (imitate this interaction geometry on the current scene conflict map below):'})
            extra_images.append(image_part(ref_bev))
        messages=[{'role':'user','content':[*content,*extra_images,image_part(conflict_map)]}]
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
            if name == 'preview_connector_route' and map_connectors and set(args) == {'connector_route', 'reason'}:
                acceptance = policy.resolve(constraints.get('acceptance_policy'))
                start_s = float(constraints.get('attack_prefix_lock_s', policy.edit_start(acceptance)))
                plan = plan_connector_route(scene, frames, args['connector_route'], start_s, float(times[-1]))
                first = plan['waypoints'][0]
                feedback = dict(status='route_preview', connector_route={k: v for k, v in plan.items() if k != 'waypoints'},
                    initial_state=dict(x_m=first['position_m'][0], y_m=first['position_m'][1],
                        yaw_rad=first['yaw_rad'], speed_mps=first['speed_mps'], time_s=first['time_s']))
                if assets:
                    try:
                        from road_height import RoadHeightMap, read_recon_translation
                        surface = RoadHeightMap.load(scene)
                        recon = read_recon_translation(scene)
                        origin = np.asarray(frames[0]['ego2global'], float)
                        yaw0 = float(np.arctan2(origin[1, 0], origin[0, 0]))
                        rotation = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
                        ref_size = assets[0]['size_m'][:2]
                        supported = []
                        for wp in plan['waypoints']:
                            pos_local = np.asarray(wp['position_m'], float)
                            pos_recon = pos_local @ rotation.T + origin[:2, 3] - recon[:2]
                            try:
                                plane = surface.placement_plane(*pos_recon, *ref_size, yaw0 + float(wp['yaw_rad']))
                                if plane['max_corner_residual_m'] <= 0.1:
                                    supported.append(wp)
                            except Exception:
                                pass
                        feedback['initial_state_has_road_height'] = bool(supported and supported[0] is first)
                        if not feedback['initial_state_has_road_height']:
                            if supported:
                                s0 = supported[0]
                                feedback['first_supported_initial_state'] = dict(
                                    x_m=s0['position_m'][0], y_m=s0['position_m'][1],
                                    yaw_rad=s0['yaw_rad'], speed_mps=s0['speed_mps'], time_s=s0['time_s'])
                                feedback['road_height_guidance'] = (
                                    'Route start is outside reconstructed road-height coverage. '
                                    'Place add_vehicle at first_supported_initial_state (or reduce approach_m).')
                            else:
                                feedback['road_height_guidance'] = (
                                    'No waypoint along this preview has reconstructed road-height support; '
                                    'choose a closer connector or smaller approach_m.')
                    except Exception:
                        pass
                connector_preview = feedback
                archive_exchange(folder, turn, call, dict(feedback, waypoints=plan['waypoints']), started=tool_started)
                continue
            if name=='preview_attack_options':
                if set(args)!={'options','reason'}:raise ValueError('Preview requires options and reason')
                from attack_scene_evidence import preview_options
                ledger_path=constructor_budget_path(folder).with_name('placement-preview-budget.json')
                ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(evaluations=0)
                hypotheses=args['options']
                if not isinstance(hypotheses,list) or not 1<=len(hypotheses)<=3:raise ValueError('Preview needs1..3 hypotheses')
                if ledger['evaluations']+len(hypotheses)>6:raise ValueError('Cumulative placement preview budget exhausted; select from evidence or finish')
                ledger['evaluations']+=len(hypotheses);atomic_json(ledger_path,ledger)
                feedback=preview_options(hypotheses,times,ego,yaw,actors,road,assets,constraints,required_operation,required_track)
                feedback['remaining_hypotheses']=6-ledger['evaluations']
                atomic_json(folder/'previews'/f'turn_{turn:02d}.json',feedback)
                archive_exchange(folder,turn,call,feedback,started=tool_started)
                continue
            if name == 'finish_search' and set(args) == {'reason'}:
                archive_exchange(folder,turn,call,dict(status='finished',reason=args['reason']),started=tool_started)
                return finish('finished_without_object_selection', args['reason'])
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
                placement_key = (name, track)
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
                from connector_route import lane_centerline_seed
                goal_xy = [float(ego[len(ego)//2, 0]), float(ego[len(ego)//2, 1])] if len(ego) else None
                lane_seed = (None if constraints.get('stationary_target') or args['speed_mps'] <= 0
                             else lane_centerline_seed(scene, frames, args['x_m'], args['y_m'], args['yaw_rad'], args['speed_mps'], times, goal_xy=goal_xy))
                if lane_seed is not None:
                    seed, angles, start_xy, start_yaw = lane_seed
                    init_state = dict(x_m=float(start_xy[0]), y_m=float(start_xy[1]), yaw_rad=float(start_yaw))
                else:
                    direction = np.array([np.cos(args['yaw_rad']), np.sin(args['yaw_rad'])])
                    seed = np.array([args['x_m'], args['y_m']]) + (times - times[0])[:, None] * args['speed_mps'] * direction
                    angles = np.full(len(times), args['yaw_rad'])
                    init_state = dict(x_m=args['x_m'], y_m=args['y_m'], yaw_rad=args['yaw_rad'])
                size = asset['size_m']
                target = dict(operation='add_vehicle', track_id=track, size_m=size,
                              asset_id=asset['asset_id'], asset_path=asset['path'], initial_speed_mps=args['speed_mps'],
                              initial_state=init_state)
                placement_key = (name, args['asset_id'], *numbers)
            else:
                raise ValueError('Unknown object-selection tool')
            # Reject unsafe seed poses at t=0; the rest of the trajectory is editable.
            # Final candidates are screened over the complete observation window.
            fixed_end=float(constraints.get('attack_prefix_lock_s',policy.edit_start(policy.resolve(constraints.get('acceptance_policy')))))
            prefix_times = np.linspace(times[0], fixed_end, 31) if fixed_end>times[0] else np.asarray([times[0]])
            prefix_xy = sample_xy(times, seed, prefix_times)
            prefix_yaw = np.interp(prefix_times, times, angles)
            rejection = rejections.get(placement_key)
            if rejection is None:
                rejection = initial_road_failure(prefix_times, prefix_xy, prefix_yaw, size, road, margin)
                if rejection is not None and name == 'add_vehicle' and len(assets) > 1:
                    from attack_scene_evidence import find_road_compatible_asset
                    compatible = find_road_compatible_asset(
                        assets, asset, prefix_times, prefix_xy, prefix_yaw, road, margin)
                    if compatible is not None:
                        size = compatible['size_m']
                        target.update(asset_id=compatible['asset_id'], asset_path=compatible['path'],
                                      size_m=size, requested_asset_id=asset['asset_id'])
                        rejection = None
                if rejection is not None:
                    rejections[placement_key] = dict(rejection, operation=target['operation'], track_id=track,
                        **({'initial_state': target['initial_state']} if name == 'add_vehicle' else {}))
            if rejection is not None:
                repeated = placement_key in attempted_rejections
                feedback = dict(rejections[placement_key], option_id=option_id, repeated=repeated,
                    next_action='Choose an available actor or a different initial placement.')
                archive_exchange(folder, turn, call, feedback, started=tool_started)
                if repeated:
                    return finish('object_selection_repeated_rejection', feedback['error'])
                attempted_rejections.add(placement_key)
                continue
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
                for local_pt, position, angle in zip(support_xy, support_xy@rotation.T+origin[:2,3]-recon[:2], support_yaw):
                    try:
                        plane=surface.placement_plane(*position,*size[:2],yaw0+angle)
                    except ValueError as exc:
                        raise ValueError(
                            f"Road height map has missing/sparse coverage near scene local ({local_pt[0]:.2f}, {local_pt[1]:.2f}) "
                            f"for start ({args['x_m']:.2f}, {args['y_m']:.2f}) [{exc}]. "
                            "Do NOT repeat this exact start; move closer to the main road/intersection or choose a different start coordinate."
                        ) from exc
                    if plane['max_corner_residual_m']>.1:
                        raise ValueError(f"Initial prefix near scene local ({local_pt[0]:.2f}, {local_pt[1]:.2f}) has non-planar road support; choose another initial state")
            initial_relation=check_initial_relation(times,seed,angles,ego,yaw,size,constraints.get('initial_relation'))
            seed_path = folder / 'placements' / f'attempt_{turn:02d}' / 'seed_trajectory.csv'
            write_trajectory(seed_path, times, seed, angles, size)
            selection = dict(schema_version='attack_object.v1', status='completed', scene=str(scene),
                instruction=instruction, target=target, seed_trajectory_path=str(seed_path),
                plan_choice=choice(options,selection_args,target) if options else None,
                selection_reason=args['reason'],initial_relation=initial_relation, annotation_changes=dict(operation='none',
                    note='Identity and motion seed only; full scene is materialized after trajectory acceptance'))
            selection['editing_scene']=str(scene)
            if connector_preview is not None:
                selection['connector_route_preview'] = connector_preview
            forecast_editor = AttackEditor(scene, selection, folder / f'placement_check_{turn:02d}', constraints)
            selection['geometry_feedback'] = forecast_editor.public('seed_000')
            atomic_json(folder / 'output.json', selection)
            feedback = dict(status='selected', target=target, geometry_feedback=selection['geometry_feedback'], output_path=str(folder / 'output.json'))
            archive_exchange(folder, turn, call, feedback, started=tool_started)
            return selection
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            feedback = dict(status='invalid', error=str(error)[:350])
            archive_exchange(folder, turn, call or _undecoded_call(turn), feedback, started=tool_started)
    return finish('object_selection_budget_exhausted',
        'No valid object selection or initial placement was found within the object-selection budget.')


@timed_operation('attack.agent_loop', None)
@timed_operation('attack.vehicle_editing',None)
def edit_object(editor, selection, archive, timeout_seconds, max_tokens, max_turns, stage_timeout_s=420):
    from attack_budget import BudgetedArchive, AttackBudgetExhausted
    from attack_tool_contract import candidate_eligibility, submission_ready_candidates
    from streaming_requests import request_protocol_details,request_timeout_details
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
        allowed = {'evaluate_candidate', 'inspect_trajectory', 'submit_candidate','adjust_initial_pose',
                   'request_object_reselection','request_ego_history_revision', 'finish_search', 'probe_pdm', 'probe_vadv2'}
        available_tools = [tool for tool in available_tools if tool['function']['name'] in allowed]
    if editor.pdm_probe is None:
        available_tools=[tool for tool in available_tools if tool['function']['name']!='probe_pdm']
    if editor.policy_probe is None:
        available_tools=[tool for tool in available_tools if tool['function']['name']!='probe_vadv2']
    if (not editor.constraints.get('prepare_ego_history') or editor.constraints.get('ego_history_revision_count',0)>=1
            or editor.constraints.get('expert_margin_revision_count',0)>=1
            or editor.constraints.get('ego_history_route') is not None
            or editor.constraints.get('ego_history_target_speed_mps') is not None):
        available_tools=[tool for tool in available_tools if tool['function']['name']!='request_ego_history_revision']
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
    timeout_recovery = None
    protocol_recovery = None
    editing_timeout_recovery_used = False
    previous_failure = None
    try:
        for turn in range(1, max_turns + 3):
            remaining = deadline - perf_counter()
            qualified=[key for key,node in editor.nodes.items() if candidate_eligibility(node['result'])['pdm_probe_eligible']]
            session_budget=archive.context() if isinstance(archive,BudgetedArchive) else None
            if session_budget is not None and not qualified and (
                    session_budget['search_requests_remaining']<=0 or session_budget['search_api_seconds_remaining']<=0):
                stop='cumulative_attack_api_budget_exhausted'
                termination=dict(reason='Cumulative attack search/API budget exhausted without a qualified candidate; finalization reserve cannot fund a new search.',
                    scope='main_agent_execution_scope',global_infeasibility_proven=False)
                break
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
            pending_records=[] if pending is None else pending.get('batch',[pending])
            delivered = getattr(editor,'inspected',set()) | {record['candidate_id'] for record in pending_records
                if record.get('purpose') not in ('pdm_probe','vadv2_probe','scene_layout')}
            context = dict(instruction=selection['instruction'], **editor.context(),
                           plan_choice=selection.get('plan_choice'),
                           last_tool_result=feedback, turns_remaining=max(0,max_turns - turn + 1),
                           finalization_only=closing,stage_seconds_remaining=round(remaining,1))
            if closing:
                ready = submission_ready_candidates(editor, qualified, delivered)
                context['ready_to_submit_candidate_ids'] = ready
                context['finalization_instruction'] = (
                    'Submit one listed candidate or finish_search with a concrete reason for declining it. '
                    'The listed candidates have passed all submission checks; their trajectory images have been delivered or are attached.'
                    if ready else 'Review or test an existing qualified candidate to meet submission checks, or finish_search. '
                    'Inspect only trajectories whose images have not been delivered.')
            if timeout_recovery is not None:
                context['request_timeout_recovery']=timeout_recovery
            if protocol_recovery is not None:
                context['request_protocol_recovery']=protocol_recovery
            if isinstance(archive,BudgetedArchive):
                archive.closing=closing
                context['cumulative_api_budget']=session_budget
            termination = constructor_exhaustion(editor.nodes,context['constructor_budget'])
            if termination is not None:
                stop = termination['stop_reason']
                break
            from attack_context_view import editing_view
            context = compact_model_context(editing_view(context,editor.public))
            prompt = (STATIC_EDIT_PROMPT if editor.constraints.get('stationary_target') else EDIT_PROMPT) + '\n' + json.dumps(context, ensure_ascii=False, separators=(',', ':'))
            if editor.pdm_probe is not None:
                prompt=PDM_PROBE_PROMPT+'\n'+prompt
                if editor.constraints.get('pdm_defense_feedback'):
                    prompt=DEFENSE_FEEDBACK_PROMPT+'\n'+prompt
            content = prompt if not pending_records else [{'type':'text', 'text':prompt}, *[image_part(record['path']) for record in pending_records]]
            if editor.policy_probe is not None:
                if isinstance(content,str):content=VADV2_PROBE_PROMPT+'\n'+content
                else:content[0]['text']=VADV2_PROBE_PROMPT+'\n'+content[0]['text']
            probe_context = editor.pdm_probe.context() if editor.pdm_probe is not None else {}
            if editor.policy_probe is not None:probe_context=editor.policy_probe.context()
            turn_tools = trajectory_tools(available_tools,editor.nodes,evaluated_candidates,
                set(probe_context.get('candidates',{})),probe_context.get('remaining',0),
                constructor_remaining=context.get('constructor_budget',{}).get('remaining_evaluations'),
                reselections_remaining=max(0,(2 if editor.constraints.get('plan_options') else 1)-editor.constraints.get('object_reselection_count',0)),
                inspected=delivered)
            if closing:
                allowed = {'submit_candidate','finish_search'} if ready else {
                    'inspect_trajectory','probe_pdm','probe_vadv2','finish_search'}
                turn_tools=[tool for tool in turn_tools if tool['function']['name'] in allowed]
                for tool in turn_tools:
                    if tool['function']['name'] == 'submit_candidate':
                        tool['function']['parameters']['properties']['candidate_id']['enum'] = ready
                    elif tool['function']['name'] == 'inspect_trajectory':
                        tool['function']['parameters']['properties']['candidate_id']['enum'] = [
                            key for key in qualified if key not in delivered]
                turn_tools=[tool for tool in turn_tools if tool['function']['name']!='inspect_trajectory'
                    or tool['function']['parameters']['properties']['candidate_id']['enum']]
            from agent_tool_view import creation_contract
            choice, turn_tools = creation_contract(turn_tools, feedback)
            if fast_stationary and not closing:
                name = 'submit_candidate' if 'seed_000' in delivered else 'inspect_trajectory'
                turn_tools = [tool for tool in turn_tools if tool['function']['name']==name]
                choice = {'type':'function', 'function':{'name':name}}
            elif choice is None:
                choice = 'auto'
            turn_names = {tool['function']['name'] for tool in turn_tools}
            if (not closing and turn == 1 and not imported_count and not editor.constraints.get('stationary_target')
                    and not editor.constraints.get('agent_decides_interaction') and 'construct_attack_candidates' in turn_names):
                choice = {'type':'function', 'function':{'name':'construct_attack_candidates'}}
            elif not closing and turn == 1 and imported_count and not fast_stationary:
                choice = {'type':'function', 'function':{'name':'inspect_trajectory'}}
            if isinstance(choice, dict):
                # The API may use auto for thinking; the selected phase remains exact locally.
                turn_names = {choice['function']['name']}
            print('Trajectory editing turn', turn, flush=True)
            try:
                _, attempt = archive.complete(f'edit_{turn:02d}', [{'role':'user', 'content':content}],
                    total_seconds=min(timeout_seconds, max(1, remaining)),
                    no_output_seconds=min(60, max(1, remaining)), retries=2, max_tokens=max_tokens,
                    extra_body={'thinking':{'type':'enabled'}}, reasoning_effort=reasoning_effort(),
                    tools=turn_tools, tool_choice=choice, parallel_tool_calls=False, allow_tool_calls=True)
            except Exception as error:
                timed_out=request_timeout_details(error)
                budget=archive.context() if isinstance(archive,BudgetedArchive) else None
                can_finalize=(budget is None or budget['requests_remaining']>0 and budget['api_seconds_remaining']>0)
                search_remains=(budget is None or budget['search_requests_remaining']>0 and budget['search_api_seconds_remaining']>0)
                protocol_error=request_protocol_details(error)
                if (protocol_error is not None and protocol_recovery is None and not closing
                        and deadline-perf_counter()>0 and search_remains):
                    protocol_recovery=dict(protocol_error,mode='single_tool_call_correction',
                        policy='Return exactly one exposed native tool call; no partial call was executed')
                    feedback=dict(status='invalid',error='The previous response contained multiple tool calls. No tool was executed; return exactly one exposed native tool call.',
                        executed=False,candidate_created=False,archive_ref=protocol_error.get('archive_ref'))
                    continue
                if (timed_out is not None and not closing and not editing_timeout_recovery_used
                        and deadline-perf_counter()>=60 and search_remains):
                    # A single-request deadline does not exhaust the stage. Keep
                    # its original deadline and actual VAD feedback; no partial
                    # response is appended and the thinking profile is unchanged.
                    editing_timeout_recovery_used=True
                    timeout_recovery=dict(timed_out,mode='continue_editing',
                        policy='one recovery decision within the original remaining stage/search budget; preserve evidence and take a concrete next action')
                    continue
                needs_finalization=closing or deadline-perf_counter()<60 or not search_remains
                if timed_out is None or not qualified or finalization_grace_used or not can_finalize or not needs_finalization:
                    raise
                # A timed-out partial call is never decoded or executed. Re-enter
                # once with only saved-candidate review tools, under a bounded clock.
                finalization_grace_used=True
                deadline=perf_counter()+90.
                timeout_recovery=dict(timed_out,mode='finalization_only',
                    policy='one bounded review of existing qualified candidates; no new construction')
                feedback=dict(status='request_timed_out',reason='The previous request did not complete. Review saved candidates or finish; partial tool arguments were not executed.')
                continue
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
                if name in ('inspect_trajectory','evaluate_candidate','probe_pdm','probe_vadv2') or (closing and name == 'submit_candidate'):
                    schema = next(tool['function']['parameters'] for tool in turn_tools if tool['function']['name']==name)
                    if args.get('candidate_id') not in schema['properties']['candidate_id']['enum']:
                        raise ValueError('Candidate is not eligible for this tool in the current state')
                if name not in ('inspect_trajectory', 'evaluate_candidate', 'probe_pdm','probe_vadv2') and (not isinstance(args.get('reason'), str) or not args['reason'].strip()):
                    raise ValueError('A nonempty action reason is required')
                if name == 'evaluate_candidate' and set(args) == {'candidate_id'}:
                    repeated_evaluation=args['candidate_id'] in evaluated_candidates
                    feedback = editor.evaluate_candidate(args['candidate_id'])
                    if repeated_evaluation:feedback=dict(feedback,cached=True,cache_scope='previous_evaluation')
                    evaluated_candidates.add(args['candidate_id'])
                elif name == 'probe_vadv2' and set(args)=={'candidate_id'}:
                    from attack_plan_options import exclude_probe_wait
                    probe_started=perf_counter()
                    try:
                        feedback=editor.policy_probe.probe(args['candidate_id'])
                    finally:
                        deadline+=exclude_probe_wait(perf_counter()-probe_started)
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
                elif name == 'adjust_initial_pose' and set(args) <= {'parent_candidate_id','position_offset_m','yaw_mode','yaw_offset_rad','yaw_rad','initial_speed_mps','reason'}:
                    feedback = editor.adjust_initial_pose(args)
                    pending = editor.inspect(feedback['candidate_id'])
                elif name == 'inspect_trajectory' and set(args) == {'candidate_id'}:
                    feedback = editor.inspect(args['candidate_id'])
                    pending = None if feedback.get('cached') else feedback
                elif name == 'submit_candidate' and set(args) == {'candidate_id', 'reason'}:
                    feedback = editor.submit(args['candidate_id'], args['reason'])
                    stop = 'candidate_submitted'
                elif name == 'request_ego_history_revision' and set(args)=={'reason','target_speed_mps'}:
                    if editor.policy_probe is not None:editor.policy_probe.require_reselection_evidence()
                    from attack_ego_preparation import prepare_acceleration
                    prepared=prepare_acceleration(editor.scene,folder/f'history_revision_{turn:02d}',
                        policy.resolve(editor.constraints.get('acceptance_policy')),target_speed_mps=args['target_speed_mps'])
                    if prepared['status']=='unavailable':
                        feedback=dict(status='invalid',reason=prepared['reason'],report_path=prepared['report_path'])
                    else:
                        editor.constraints['_ego_history_revision_target']=args['target_speed_mps']
                        editor.constraints['_ego_history_revision_requested']=True
                        feedback=dict(status='object_reselection_requested',reason=args['reason'],shared_state=prepared['shared_state'])
                        stop='object_reselection_requested'
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
                    stop='vadv2_probe_budget_exhausted_without_policy_failure'
                    termination=dict(reason='Cumulative VADv2 budget exhausted without confirmed target collision or drivable-area violation after takeover.',
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
        timed_out=request_timeout_details(error)
        if timed_out is not None:
            stop = 'stage_timeout_exhausted' if perf_counter() >= deadline-1 else 'request_timeout_exhausted'
            reason=('The model produced partial output but no completed decision before the request deadline.'
                if timed_out['timeout_category']=='model_decision_timeout' else 'A model request timed out; see timeout category and archived evidence.')
            termination = dict(timed_out,reason=reason,
                stage_timeout_seconds=stage_timeout_s,scope='current_tools_and_budget',global_infeasibility_proven=False,
                finalization_grace_used=finalization_grace_used)
        else:
            stop = 'request_or_execution_failed'
            raise
    finally:
        resumable = [key for key,node in editor.nodes.items() if candidate_eligibility(node['result'])['pdm_probe_eligible']]
        atomic_json(folder / 'output.json', dict(schema_version='attack_trajectory.v1',
            status='completed' if editor.submitted else 'not_submitted', stop_reason=stop,
            selected_candidate_id=editor.submitted, target_track=editor.target['track_id'],
            reason=None if termination is None else termination['reason'],
            termination_evidence=termination,
            candidate_count=len(editor.nodes), image_deliveries=deliveries,
            imported_candidate_count=imported_count, new_candidate_count=len(editor.history) - imported_count,
            constructor_budget=editor.context()['constructor_budget'],
            api_budget=archive.context() if isinstance(archive,BudgetedArchive) else None,
            resumable_candidate_ids=resumable,
            ready_to_submit_candidate_ids=submission_ready_candidates(editor, resumable, getattr(editor, 'inspected', set())),
            finalization_grace_used=finalization_grace_used,
            request_timeout_recovery=timeout_recovery,
            request_protocol_recovery=protocol_recovery,
            editing_timeout_recovery_used=editing_timeout_recovery_used,
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
    if editor.constraints.get('_ego_history_revision_requested'):
        revised['_ego_history_revision_target']=editor.constraints['_ego_history_revision_target']
        revised['ego_history_revision_count']=config.get('ego_history_revision_count',0)+1
    if editor.pdm_probe is not None:
        latest=editor.pdm_probe.context()['latest_feedback']
        if latest is not None:revised['pdm_reselection_feedback']=latest
    if editor.policy_probe is not None:
        latest=editor.policy_probe.context()['latest_feedback']
        if latest is not None:revised['revision_feedback']=dict(stage='native_vadv2',outcome='target_not_hit',evidence=latest,
            guidance='Revise the original task using actual VADv2 collision and road-compliance feedback')
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
    if continue_from is not None:
        prior_path=Path(continue_from)/'input.json'
        if prior_path.is_file():
            saved=json.loads(prior_path.read_text()).get('constraints',{})
            if not same_scene_constraints(saved,config):raise ValueError('Continuation constraints differ')
            for key in ('object_reselection_count','object_reselection_reason','object_selection_feedback','ego_history_revision_count','expert_margin_revision_count'):
                if key in saved:config[key]=saved[key]
            for key in ('vadv2_probe_root','pdm_probe_root'):
                if saved.get(key):
                    previous_root=Path(saved[key]).resolve()
                    allowed=Path(os.getenv('ADVERSARIAL_CONTINUATION_ROOT',str(experiments_root()))).resolve()
                    if allowed not in previous_root.parents:raise ValueError('Prior probe budget is outside the continuation area')
                    config[key]=str(previous_root)
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
        min_v = -0.5 if key == 'official_margin_m' else 0
        if type(config[key]) not in (int, float) or not np.isfinite(config[key]) or config[key] < min_v:
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
        budget=archive.context()
        if continue_from is None and (budget['search_requests_remaining']<=0 or budget['search_api_seconds_remaining']<=0):
            summary.update(status='not_submitted',stop_reason='cumulative_attack_api_budget_exhausted',
                reason='Cumulative attack search/API budget exhausted before a new selection; reuse qualified saved candidates only.',
                search_stage=stage,api_budget=budget,resumable_candidate_ids=[])
            atomic_json(output/stage/'output.json',dict(summary))
            return summary
        if config.get('vadv2_self_test'):
            from backend_hooks import start_prewarm
            start_prewarm(config['vadv2_policy_config'],output/'vad-prewarm.json')
        preparation=None
        if 'prepare_ego_history' in config:
            prior_scene=None
            if continue_from is None:
                from expert_attack_feedback import revision_history
                prior_scene=revision_history(config,scene,instruction,output,required_operation,required_track)
            if continue_from is not None:
                prior=Path(continue_from).resolve()
                allowed=Path(os.getenv('ADVERSARIAL_CONTINUATION_ROOT',str(experiments_root()))).resolve()
                if prior==output or allowed not in prior.parents:
                    raise ValueError('Continuation source must be another experiments run')
                archived_input=json.loads((prior/'input.json').read_text())
                from ego_history import source_scene
                if (source_scene(archived_input['scene'])!=source_scene(scene)
                        or not same_scene_constraints(archived_input['constraints'],config)
                        or archived_input['instruction']!=instruction or archived_input['assets']!=assets
                        or archived_input.get('required_operation')!=required_operation
                        or archived_input.get('required_track')!=required_track):
                    raise ValueError('Continuation scene, constraints and instruction must match')
                prior_choice=json.loads((prior/'object_selection/output.json').read_text())
                prior_scene=prior_choice.get('editing_scene',archived_input.get('editing_scene',archived_input['scene']))
            stage='ego_history_prepare'
            from attack_ego_preparation import prepare_acceleration
            route_arguments = {'route': config['ego_history_route']} if config.get('ego_history_route') is not None else {}
            preparation=prepare_acceleration(scene,output/'ego_history',effective,
                enabled=config['prepare_ego_history'],
                target_speed_mps=config.get('_ego_history_revision_target',config.get('ego_history_target_speed_mps')),
                previous_scene=prior_scene, **route_arguments)
            summary['ego_history_preparation']=preparation
            if preparation['status']=='unavailable':
                summary.update(status='not_submitted',stop_reason='ego_history_unavailable',reason=preparation['reason'],search_stage=stage)
                atomic_json(output/'ego_history_prepare/output.json',summary)
                return summary
            scene=Path(preparation['scene'])
            config['_fixed_ego_history']=dict(preparation['shared_state'],status=preparation['status'],report_path=preparation['report_path'])
            config.pop('_ego_history_revision_target',None)
            saved=json.loads((output/'input.json').read_text());saved['editing_scene']=str(scene)
            saved['ego_history_preparation']=preparation
            atomic_json(output/'input.json',saved)
        stage='official_map_coverage'
        from official_map_coverage import prepare_coverage
        scene=prepare_coverage(scene, output)
        stage='object_selection'
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
            from ego_history import source_scene
            if source_scene(prior_input['scene']) != source_scene(scene) or not same_scene_constraints(prior_input['constraints'],config) or prior_input['instruction'] != instruction or prior_input['assets'] != assets or prior_input.get('required_operation') != required_operation or prior_input.get('required_track') != required_track:
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
            summary.update(selection, api_budget=archive.context())
            return summary
        if preparation is not None:
            selection['editing_scene']=str(scene)
            atomic_json(output/'object_selection/output.json',selection)
        if selection.get('editing_scene'):
            scene=Path(selection['editing_scene'])
            from ego_history import verify,clone
            if continue_from is not None and preparation is None:
                scene=clone(scene,output/'shared_history')
                selection['editing_scene']=str(scene)
                atomic_json(output/'object_selection/output.json',selection)
            verify(scene)
            path=output/'input.json'
            archived=json.loads(path.read_text());archived['editing_scene']=str(scene)
            atomic_json(path,archived)
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
                          'adjust_initial_pose':editor.adjust_initial_pose,
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
            from ego_history import attach
            attach(scene,frozen)
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
        from streaming_requests import request_timeout_details
        timed_out=request_timeout_details(error)
        if timed_out is not None:
            summary.update(status='not_submitted',stop_reason='request_timeout_exhausted',
                reason=('The model produced partial output but no completed decision before the request deadline.'
                    if timed_out['timeout_category']=='model_decision_timeout' else 'A model request timed out; see timeout category and archived evidence.'),
                search_stage=stage,api_budget=archive.context(),resumable_candidate_ids=[],
                termination_evidence=dict(timed_out,scope='current_tools_and_budget',global_infeasibility_proven=False))
            atomic_json(output/stage/'output.json',dict(summary))
            return summary
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
