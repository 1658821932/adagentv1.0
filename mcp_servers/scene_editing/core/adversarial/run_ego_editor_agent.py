"""Run a geometry-informed native-tool agent over constructor/FOCI candidates and parent edits."""
from llm_profile import load_llm_profile
from workflow_timing import timed_operation

import argparse
import json
import os
from pathlib import Path

import httpx
from openai import OpenAI

from ego_editor import EgoEditor
from experiment_runtime import ROOT, atomic_json
from run_ego_reference import DEFAULT_TARGET_TRACK
from streaming_requests import StreamingRequestArchive
from model_context import compact_model_context
from kimi_request_parameters import normalize_kimi_parameters

PROMPT = """You edit driving trajectories by stating a concrete change, checking it, then comparing it with retained alternatives.
For speed/route calls, read decision_summary before raw evidence. recommended_previews contain complete arguments and an explicit condition; they are options to preview, never automatic goal changes. When a recommendation matches the task and budget remains, preview it before deciding the modified behavior is unsuitable. create_candidate_call repeats a passed preview with preview_only=false. Candidate cards share identity, eligibility and failure fields; detailed numeric evidence remains available.
Use edit_speed_profile for speed/timing changes. set_speed_targets takes a start time and timed signed speed states: positive forward, zero stop, negative reverse with body heading unchanged. Two zero states wait; zero then positive restarts. Explicitly stop before reversing. The final target speed is maintained to scene end; stopping is not mandatory. Default preserves the parent spatial route, including known rear geometry. Choose recorded_baseline explicitly when continuing past a stopped parent endpoint; no reference is extrapolated. Existing history through the edit start is preserved.
For an explicit braking parent, retime_braking moves its original onset by braking_start_shift_s while preserving either stop_position or stop_time. The backend rebuilds from the earliest affected time; it does not retain the old braking before a later takeover. Omit transition_style to keep the parent braking style. For lateral avoidance use edit_route: signed left_offset_m, transition_duration_s, optional hold/return durations. preserve_speed keeps total parent speed while spending progress on lateral travel; preserve_progress may raise total speed. Return means the same spatial route, not the same-time parent pose. Check continuous total acceleration/normal component and intermediate progress before claiming improvement. Saved spatial routes are reused across speed edits; initial legacy routes are reconstructed from consistent position geometry. Inspect actual_motion for peak time and tangential/normal acceleration when a motion check fails. Sampled FOCI parents have no invented named braking phase: use set_speed_targets. Preview screens without creating a candidate; execute to create a submittable result. Do not specify mutually conflicting fields or infer all alternatives infeasible from one rejection.
For broader search use optimize_defense_candidate with preserve_route, terminal_behavior, improvement_goal and optional speed_targets (the same signed speeds and absolute times, but explicitly soft sample-based FOCI preferences). Inspect speed_target_feedback rather than claiming those targets are met. A resulting candidate can be refined with edit_speed_profile for exact timed speed states. Use edit_route for lateral changes. Legacy coordinate/offset operations remain replayable but are not exposed in this tool set.
Generated braking parents use reconstructed continuous position/velocity at arbitrary takeover times; CSV-only parents use the reported sampled-state estimate. Inspect actual_start_state and actual generated curve bounds. max_lateral_speed_mps is body sideslip, not road-coordinate lateral travel. Geometry uses the ego rear axle. Planar acceleration includes road curvature, so a tangential braking cap alone does not ensure a motion pass. A failed local edit does not prove every alternative impossible. Keep a passing fallback while exploring within budget; a passed screen establishes eligibility, not optimality. Finish may report no passing candidate found and remaining uncertainties; no required attempt count.
Return exactly one native tool call, then end the turn. Wait for real tool feedback before any next call; do not simulate tool execution or future turns. You are an ego trajectory editing agent. Use the input task, scene evidence and candidate feedback to choose a maneuver. Compare relevant alternatives (for example continued travel, speed adjustment, lateral movement, stopping or intentional reverse) when supported by the scene and available tools; there is no default maneuver or required order of alternatives. Explain the tradeoffs relevant to the task. Clearance, progress, comfort and evaluator scores are separate evidence, not objectives to maximize automatically. Keep a reasonable passing fallback while exploring within the remaining budget. A passed screen establishes eligibility for submission, not global optimality or a requirement to stop searching immediately.
Read context.acceptance_policy for the effective reference-screen policy, including observation_end_s, motion limits, required checks, progress/stop rules and comfort_reference_mps2. Preserve observation history through that boundary. Treat the comfort reference as a diagnostic preference, distinct from hard acceleration limits. Scene/conflict recommendations are advisory; their suggested actions or thresholds do not add acceptance gates. Trace their claims to image or geometry evidence; a model recommendation alone does not establish road legality or available space. Tool argument constraints are described separately below.
Respect visible no-cross markings: do not cross a solid or double solid line that prohibits crossing. Use available images, geometry and their provenance to assess the proposed maneuver. Missing marking geometry does not itself forbid all lateral movement; state unknown compliance and avoid claiming permission from missing evidence. A drivable-area pass or zero lateral offset does not establish lane compliance.
Check actual_edit, motion_evidence, reverse distance/intervals and declared motion_intent. Intentional reverse needs scene support, attention to rear-space uncertainty, feasible motion and a suitable terminal state; it is not a generic way to improve clearance. Inspect terminal_speed_mps and terminal_target_clearance_rate_mps alongside the scene: a final frame still closing on an obstacle does not demonstrate a safe stop. Report unresolved end-of-window risk for any maneuver. Sampled reference checks do not establish continuous-time collision avoidance, physical execution, complete lawfulness or unavoidability.
The controller requires initial candidate selection; each edit or repair call screens one candidate. Only an existing reference_screen_passed candidate may be submitted. Within available calls, revise from feedback or submit with a task-based reason. The controller can force submission at its turn/edit limit when an eligible candidate exists; this is a budget rule, not an optimality judgment. FOCI candidates are precomputed candidates, not agent edits. If no candidate can be submitted, report the search limitation without claiming unavoidability."""
TOOLS = [
    {'type':'function','function':{'name':'select_constructor_candidate','description':'Choose an initial constructor or FOCI trajectory.','parameters':{'type':'object','properties':{'candidate_id':{'type':'string'},'reason':{'type':'string'}},'required':['candidate_id','reason'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'edit_trajectory_points','description':'Apply 1-5 ordered forward/left/yaw edit points to a parent. One candidate is screened.','parameters':{'type':'object','properties':{'parent_candidate_id':{'type':'string'},'points':{'type':'array','items':{'type':'object','properties':{'time_s':{'type':'number'},'forward_m':{'type':'number'},'left_m':{'type':'number'},'yaw_rad':{'type':'number'}},'required':['time_s','forward_m','left_m','yaw_rad'],'additionalProperties':False},'minItems':1,'maxItems':5},'reason':{'type':'string'}},'required':['parent_candidate_id','points','reason'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'repair_candidate','description':'Screen one blend of a parent toward another existing candidate; one ratio per call.','parameters':{'type':'object','properties':{'parent_candidate_id':{'type':'string'},'target_candidate_id':{'type':'string'},'ratio':{'type':'number','description':'Target fraction in (0,1].'},'window_s':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2,'description':'Optional time interval; offsets smoothly return to zero at its boundaries.'},'reason':{'type':'string'}},'required':['parent_candidate_id','target_candidate_id','ratio','reason'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'submit_candidate','description':'Submit only an existing reference_screen_passed candidate.','parameters':{'type':'object','properties':{'candidate_id':{'type':'string'},'reason':{'type':'string'}},'required':['candidate_id','reason'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'finish_search','description':'Stop this search without submission. State tested options and uncertainties; local failure does not prove all trajectories infeasible.','parameters':{'type':'object','properties':{'reason':{'type':'string'}},'required':['reason'],'additionalProperties':False}}}
]


# One generic edit entry, with explicit mode-specific validation shared by both runners.
_POINT_PROPERTIES = TOOLS[1]['function']['parameters']['properties']
_POINT_PROPERTIES.update(
    edit_mode={'type':'string','enum':['delta_parent','set_offsets','replace_positions','path_progress']},
    anchor_candidate_id={'type':'string'}, path_candidate_id={'type':'string'},
    window_s={'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2},
    start_time_s={'type':'number'},
    motion_intent={'type':'string','enum':['unrestricted','forward_stop','forward','reverse']},
    end_behavior={'type':'string','enum':['hold','reconnect']})
_POINT_PROPERTIES['points']['items']['properties'].update({key:{'type':'number'} for key in
    ('x_m','y_m','distance_from_start_m','speed_mps')})
_POINT_PROPERTIES['points']['items']['required'] = ['time_s']
_POINT_PROPERTIES['points']['items']['properties']['distance_from_start_m']['description'] = 'Signed arc travel from start_time_s on the fixed forward path; not an offset added to the parent.'
_POINT_PROPERTIES['points']['items']['properties']['speed_mps']['description'] = 'Signed desired path speed at this target; zero at final hold.'
TOOLS[1]['function']['description'] = ('Screen one edit with 1-5 strictly time-ordered targets. '
    'delta_parent (default) adds forward/left/yaw offsets in parent pose frames; repeated edits accumulate. '
    'set_offsets sets forward_m/left_m/yaw_rad relative to anchor_candidate_id at each target time over window_s, '
    'without adding parent offsets. replace_positions sets scene-local x_m/y_m/absolute yaw_rad over window_s. '
    'These replacement modes inherit parent start pose/velocity: hold requires the final target at window end '
    'and freezes its absolute pose afterward; reconnect requires the last target strictly before window end '
    'and returns to parent pose/velocity at that boundary. '
    'path_progress uses path_candidate_id, start_time_s and targets with time_s, distance_from_start_m and '
    'signed speed_mps on a fixed forward reference path matching the parent start position/heading. '
    'It inherits parent start signed speed, requires final speed zero and end_behavior=hold; no reconnect '
    'or terminal cruise is supported. All new modes require motion_intent and end_behavior; targets must '
    'be strictly after the inherited start, without a start-time zero point.')
PROMPT += (" Use only the tools exposed in this request and their exact schemas. "
    "The speed and route tools select their own backend modes; do not pass edit_mode, "
    "coordinate points or legacy intent parameters. repair_candidate blends toward another "
    "existing candidate, optionally within window_s, while preserving observation history. "
    "Explain the intended change and retained benefits, then compare the actual feedback "
    "with the parent. Correct missing or unexpected fields before retrying; inspecting "
    "an unchanged candidate does not repair an invalid call.")


from ego_edit_contract import (validate_point_arguments, normalize_edit_call,
    mode_tools, EDIT_TOOL_NAMES)

# Agent-facing schemas are flat and exact; execution still uses the existing backend.
_point_template = TOOLS[1]
TOOLS[1:2] = mode_tools(_point_template)


def _arguments(call):
    try:
        value = json.loads(call['function']['arguments'])
    except json.JSONDecodeError as error:
        raise ValueError(f'invalid_json at character {error.pos}: resend one valid JSON object using the selected tool schema; no format markers or duplicated fragments') from None
    if not isinstance(value, dict):
        raise ValueError('Tool arguments must be an object')
    return value


def _tool_contract(editor, turn, max_turns, max_edit_candidates, require_edit=False):
    passing = [key for key, node in editor.nodes.items() if node['result']['status'] == 'reference_screen_passed']
    choice = 'auto'
    if turn == 1:
        choice = {'type':'function','function':{'name':'select_constructor_candidate'}}
    if (turn == max_turns or editor.edit_count >= max_edit_candidates) and passing \
            and (not require_edit or editor.edit_count > 0):
        choice = {'type':'function','function':{'name':'submit_candidate'}}
    available_tools = TOOLS if editor.current is None else [
        tool for tool in TOOLS if tool['function']['name'] != 'select_constructor_candidate']
    if require_edit and editor.current is not None and editor.edit_count == 0:
        available_tools = [tool for tool in available_tools if tool['function']['name'] != 'submit_candidate']
    if editor.edit_count >= max_edit_candidates:
        available_tools = [tool for tool in available_tools if tool['function']['name'] not in EDIT_TOOL_NAMES | {'repair_candidate'}]
    return choice, available_tools


def _restore_progress(editor, run_dir, old_summary, profile, timeout_seconds,
        max_turns, max_edit_candidates, max_invalid_calls, require_edit=False):
    if old_summary['stop_reason'] != 'request_or_execution_failed':
        raise ValueError('Only an interrupted request can be explicitly resumed')
    actions = old_summary['model_actions']
    folders = sorted((run_dir/'tool-exchanges').glob('turn_*'))
    if not folders or len(folders) != actions:
        raise ValueError('Recorded tool turns do not match the interrupted summary')
    invalid_calls = 0; invalid_signatures = {}; last_feedback = None
    for expected, folder in enumerate(folders, 1):
        if folder.name != f'turn_{expected:02d}':
            raise ValueError('Recorded tool turns are not contiguous')
        attempts = list((run_dir/'requests').glob(f'turn_{expected:02d}_*/attempt_001'))
        success = []
        for path in attempts:
            if not (path/'response.json').is_file() or not (path/'result.json').is_file():
                continue
            response = json.loads((path/'response.json').read_text())
            response_result = json.loads((path/'result.json').read_text())
            response_choice = response['choices'][0]
            message = response_choice['message']
            calls = message.get('tool_calls') or []
            if (not (path/'error.json').exists() and response_result.get('status') == 'received'
                    and response_choice.get('finish_reason') in {'tool_calls','stop'}
                    and response_result.get('finish_reason') == response_choice.get('finish_reason')
                    and len(calls) == 1):
                success.append((path, response, message, calls))
        if len(success) != 1:
            raise ValueError('Expected exactly one complete response per recorded turn')
        attempt, response, message, calls = success[0]
        request = json.loads((attempt/'request.json').read_text())
        if request.get('endpoint') != profile['base_url'] or request['payload'].get('model') != profile['model']:
            raise ValueError('Archived request model or endpoint differs from the active profile')
        if json.loads((folder/'call.json').read_text()) != calls[0]:
            raise ValueError('Archived native call differs from its complete response')
        last_feedback = json.loads((folder/'result.json').read_text())
        if last_feedback['status'] in {'invalid','rejected'}:
            invalid_calls += 1
            signature = calls[0]['function']['name']+'\n'+calls[0]['function']['arguments']
            invalid_signatures[signature] = invalid_signatures.get(signature, 0)+1
    if invalid_calls != old_summary['invalid_tool_calls'] or invalid_calls >= max_invalid_calls:
        raise ValueError('Archived invalid-call budget differs or is exhausted')
    next_turn = actions+1
    if (run_dir/'tool-exchanges'/f'turn_{next_turn:02d}').exists():
        raise ValueError('The interrupted turn already has a tool exchange')
    context = editor.model_context(max_turns-next_turn+1, max_edit_candidates-editor.edit_count, last_feedback)
    if json.loads((run_dir/'config.json').read_text()).get('context_view_version') == 'lossless_context.v1':
        context = compact_model_context(context)
    messages = [{'role':'user','content':PROMPT+'\nCurrent geometry-informed state:\n'+json.dumps(context, ensure_ascii=False, separators=(',',':'))}]
    choice, available_tools = _tool_contract(editor, next_turn, max_turns, max_edit_candidates, require_edit)
    failed = list((run_dir/'requests').glob(f'turn_{next_turn:02d}_*/attempt_001'))
    if not 1 <= len(failed) < 3:
        raise ValueError('Resume requires one or two failed requests for the next turn')
    for attempt in failed:
        request = json.loads((attempt/'request.json').read_text()); payload = request['payload']
        expected = normalize_kimi_parameters(profile['model'], dict(messages=messages,
            tools=available_tools, tool_choice=choice, reasoning_effort='high', max_tokens=6144,
            extra_body={'thinking':{'type':'enabled'}}, parallel_tool_calls=False))
        if request.get('endpoint') != profile['base_url'] or payload.get('model') != profile['model'] \
                or any(payload.get(field) != expected.get(field) for field in (
                    'messages','tools','tool_choice','reasoning_effort','max_tokens','extra_body','parallel_tool_calls')):
            raise ValueError('Failed request does not match the restored next-turn contract')
        error = json.loads((attempt/'error.json').read_text())
        if error.get('request_total_seconds') != timeout_seconds:
            raise ValueError('Failed request uses a different timeout')
        response_path = attempt/'response.json'
        if response_path.exists():
            result_path = attempt/'result.json'
            if not result_path.is_file():
                raise ValueError('Failed request has a response without its validation result')
            response = json.loads(response_path.read_text())
            result = json.loads(result_path.read_text())
            response_choice = response['choices'][0]
            message = response_choice['message']
            # A length-limited empty stream is archived before complete() rejects it.
            # It contains no executable action and remains a failed request.
            incomplete_empty = (response_choice.get('finish_reason') == 'length'
                and not (message.get('content') or '').strip()
                and not message.get('tool_calls')
                and result.get('status') == 'empty'
                and result.get('finish_reason') == response_choice.get('finish_reason'))
            if not incomplete_empty:
                raise ValueError('Failed request contains a complete response or executable tool call')
    return actions, invalid_calls, last_feedback, invalid_signatures


def _summary(editor, stop_reason, model_actions, invalid_calls, max_turns, max_edit_candidates, max_invalid_calls):
    best = editor.nodes[editor.best]['result']
    constructor_profile = editor._profile_comparison(editor.best, editor.constructor_best)
    pool_profile = editor._profile_comparison(editor.best, editor.pool_best)
    best_clearance = best['min_target_clearance_m']
    return dict(global_feasibility_proven=False,
        search_evidence_scope='Candidates and edits tested in this run; no global infeasibility proof',
        status='reference_candidate_submitted' if editor.submitted else 'not_submitted',
        selected_candidate_id=editor.submitted, current_candidate_id=editor.current,
        best_candidate_id=editor.best, constructor_best_candidate_id=editor.constructor_best,
        constructor_best_clearance_m=editor.constructor_best_clearance,
        pool_best_candidate_id=editor.pool_best, pool_best_clearance_m=editor.pool_best_clearance,
        foci_best_candidate_id=editor.foci_best,
        best_min_target_clearance_m=best_clearance,
        agent_improvement_over_constructor_m=best_clearance-editor.constructor_best_clearance,
        agent_pareto_improvement_over_constructor=constructor_profile['dominates_reference'],
        best_profile_vs_constructor=constructor_profile,
        foci_improvement_over_constructor_m=editor.pool_best_clearance-editor.constructor_best_clearance,
        agent_improvement_over_pool_m=best_clearance-editor.pool_best_clearance,
        agent_pareto_improvement_over_pool=pool_profile['dominates_reference'],
        best_profile_vs_pool=pool_profile,
        initial_candidate_counts={source:sum(node['source'] == source for node in editor.nodes.values())
            for source in ('constructor', 'foci')},
        edit_candidate_count=editor.edit_count,
        selected_initial_candidate_id=editor.selected_initial, edit_candidates=editor.edit_count,
        stop_reason=stop_reason, model_actions=model_actions, invalid_tool_calls=invalid_calls,
        max_model_actions=max_turns, max_edit_candidates=max_edit_candidates, max_invalid_calls=max_invalid_calls,
        context_mode='rebuilt_geometry_informed_state_each_turn',
        physical_avoidance='undetermined', road_status='checked',
        rear_coverage='partial_observations_only', carla_executed=False, omnisg_rendered=False)


@timed_operation('ego.agent', 'run_dir')
def run(input_dir, constructor_run, run_dir, timeout_seconds=240, max_turns=12,
        max_edit_candidates=5, max_invalid_calls=3, resume=False, foci_run=None, target_track=DEFAULT_TARGET_TRACK, require_edit=False):
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 600:
        raise ValueError('timeout_seconds must be an integer in [1,600]')
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or not 4 <= max_turns <= 16:
        raise ValueError('max_turns must be an integer in [4,16]')
    if isinstance(max_edit_candidates, bool) or not isinstance(max_edit_candidates, int) or not 1 <= max_edit_candidates <= 6:
        raise ValueError('max_edit_candidates must be an integer in [1,6]')
    if isinstance(max_invalid_calls, bool) or not isinstance(max_invalid_calls, int) or not 1 <= max_invalid_calls <= 5:
        raise ValueError('max_invalid_calls must be an integer in [1,5]')
    run_dir = Path(run_dir).resolve()
    if (ROOT/'experiments') not in run_dir.parents:
        raise ValueError('Run must be inside experiments')
    config_path = run_dir/'config.json'
    stored = json.loads(config_path.read_text()) if resume else None
    context_view = stored.get('context_view_version') if resume else 'lossless_context.v1'
    if context_view != 'lossless_context.v1':
        raise ValueError('Unsupported archived context view version')
    config = dict(version='ego_editor_agent.v18', prompt_contract_version='goal_edit.v2',
        input_dir=str(Path(input_dir).resolve()), constructor_run=str(Path(constructor_run).resolve()),
        target_track=target_track, timeout_seconds=timeout_seconds, max_turns=max_turns,
        max_edit_candidates=max_edit_candidates, max_invalid_calls=max_invalid_calls,
        context_mode='rebuilt_geometry_informed_state_each_turn', require_edit=require_edit, carla_executed=False)
    if context_view is not None:
        config['context_view_version'] = context_view
    if foci_run is not None:
        config['foci_run'] = str(Path(foci_run).resolve())
    if resume:
        if stored.get('acceptance_policy'):
            config['acceptance_policy'] = stored['acceptance_policy']
        if json.loads(config_path.read_text()) != config or json.loads((run_dir/'tools.json').read_text()) != TOOLS:
            raise ValueError('Resume config or tool schema differs from the frozen run')
    editor = EgoEditor(input_dir, constructor_run, run_dir, foci_run=foci_run, resume=resume,
        target_track=target_track)
    config['acceptance_policy'] = editor.effective_policy
    if not resume:
        atomic_json(config_path, config)
        atomic_json(run_dir/'tools.json', TOOLS)
    profile = load_llm_profile()
    stop = 'model_turn_budget_exhausted'; actions = 0; invalid_calls = 0; last_feedback = None; invalid_signatures = {}
    if resume:
        old_summary = json.loads((run_dir/'summary.json').read_text())
        actions, invalid_calls, last_feedback, invalid_signatures = _restore_progress(editor, run_dir,
            old_summary, profile, timeout_seconds, max_turns, max_edit_candidates, max_invalid_calls, require_edit)
        history = run_dir/'status-history'; history.mkdir(exist_ok=True)
        saved = history/f'before_resume_{len(list(history.iterdir()))+1:02d}.json'
        with saved.open('x') as stream:
            json.dump(old_summary, stream, indent=2)
    client = OpenAI(api_key=profile['api_key'], base_url=profile['base_url'], max_retries=0,
        http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=float(timeout_seconds)))
    archive = StreamingRequestArchive(run_dir/'requests', client, profile)
    try:
        for turn in range(actions+1, max_turns+1):
            context = editor.model_context(max_turns-turn+1, max_edit_candidates-editor.edit_count, last_feedback)
            if context_view is not None:
                context = compact_model_context(context)
            messages = [{'role':'user','content':PROMPT+'\nCurrent geometry-informed state:\n'+json.dumps(context, ensure_ascii=False, separators=(',',':'))}]
            choice, available_tools = _tool_contract(editor, turn, max_turns, max_edit_candidates, require_edit)
            print('Editor turn', turn, 'started', flush=True)
            _, attempt = archive.complete(f'turn_{turn:02d}', messages, total_seconds=timeout_seconds,
                max_tokens=6144, extra_body={'thinking':{'type':'enabled'}}, reasoning_effort='high',
                tools=available_tools, tool_choice=choice, parallel_tool_calls=False, allow_tool_calls=True)
            message = json.loads((attempt/'response.json').read_text())['choices'][0]['message']
            calls = message.get('tool_calls') or []; actions += 1
            if not calls:
                stop = 'model_stopped_without_submission'; break
            if len(calls) != 1:
                stop = 'invalid_tool_call'; break
            call = calls[0]; name = call['function']['name']
            try:
                arguments = _arguments(call)
                if name == 'select_constructor_candidate':
                    if set(arguments) != {'candidate_id','reason'}: raise ValueError('Invalid select arguments')
                    result = editor.select_constructor(arguments)
                elif name in EDIT_TOOL_NAMES | {'repair_candidate'}:
                    if editor.edit_count >= max_edit_candidates: raise ValueError('Edit candidate budget exhausted')
                    if name in EDIT_TOOL_NAMES:
                        arguments = normalize_edit_call(name, arguments)
                    expected = (set(arguments) if name in EDIT_TOOL_NAMES
                        else {'parent_candidate_id','target_candidate_id','ratio','reason'})
                    allowed = expected | ({'window_s'} if name == 'repair_candidate' else set())
                    if not expected <= set(arguments) <= allowed or not isinstance(arguments['reason'], str) or not arguments['reason'].strip():
                        raise ValueError('Invalid edit arguments')
                    result = editor.edit(dict(arguments, mode='point_edit' if name in EDIT_TOOL_NAMES else 'repair'))
                elif name == 'submit_candidate':
                    if require_edit and editor.edit_count == 0: raise ValueError('A valid edit or repair is required before submission')
                    if set(arguments) != {'candidate_id','reason'}: raise ValueError('Invalid submit arguments')
                    result = editor.submit(arguments)
                elif name == 'finish_search':
                    if set(arguments) != {'reason'} or not isinstance(arguments['reason'], str) \
                            or not arguments['reason'].strip(): raise ValueError('Invalid finish arguments')
                    result = {'status':'finished','reason':arguments['reason'],
                        'conclusion':'no_qualified_candidate_found','global_feasibility_proven':False,
                        'tested_candidate_ids':[row['candidate_id'] for row in editor.history]}; stop = 'model_finished_without_submission'
                else:
                    raise ValueError('Unknown tool')
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                result = {'status':'invalid','error':str(error)[:300]}
            folder = run_dir/'tool-exchanges'/f'turn_{turn:02d}'; folder.mkdir(parents=True, exist_ok=False)
            atomic_json(folder/'call.json', call); atomic_json(folder/'result.json', result)
            last_feedback = result
            print('Tool', name, 'returned', result['status'], flush=True)
            if result['status'] in {'invalid', 'rejected'}:
                invalid_calls += 1
                signature = name+'\n'+call['function']['arguments']
                invalid_signatures[signature] = invalid_signatures.get(signature, 0)+1
                if invalid_signatures[signature] >= 2:
                    stop = 'repeated_invalid_tool_call'; break
                if invalid_calls >= max_invalid_calls:
                    stop = 'invalid_call_budget_exhausted'; break
            if editor.submitted:
                stop = 'candidate_submitted'; break
            if result['status'] == 'finished':
                break
            if editor.edit_count >= max_edit_candidates and not any(
                    node['result']['status'] == 'reference_screen_passed' for node in editor.nodes.values()):
                stop = 'edit_candidate_budget_exhausted'; break
    except KeyboardInterrupt:
        stop = 'interrupted_by_operator'
        raise
    except Exception:
        stop = 'request_or_execution_failed'; raise
    finally:
        client.close()
        summary = _summary(editor, stop, actions, invalid_calls, max_turns, max_edit_candidates, max_invalid_calls)
        atomic_json(run_dir/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--constructor-run', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=240)
    parser.add_argument('--max-turns', type=int, default=12)
    parser.add_argument('--max-edit-candidates', type=int, default=5)
    parser.add_argument('--max-invalid-calls', type=int, default=3)
    parser.add_argument('--foci-run', type=Path, help='Optional FOCI candidates optimized from this constructor run')
    parser.add_argument('--target-track', required=True)
    parser.add_argument('--require-edit', action='store_true', help='Require one valid edit or repair before submission')
    parser.add_argument('--resume', action='store_true', help='Explicitly resume an interrupted run after verifying its archived state')
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.constructor_run, args.run, args.timeout_seconds,
        args.max_turns, args.max_edit_candidates, args.max_invalid_calls, resume=args.resume,
        foci_run=args.foci_run, target_track=args.target_track, require_edit=args.require_edit), ensure_ascii=False))
