"""Bounded on-demand visual inspection for the ego trajectory editor."""
from llm_profile import load_llm_profile
from visual_review import review_state, tool_contract as visual_tool_contract, rejection
from workflow_timing import timed_operation

import argparse
import json
import os
from pathlib import Path
import httpx
from openai import OpenAI
from ego_editor import EgoEditor
from ego_edit_contract import EDIT_TOOL_NAMES, normalize_edit_call
from ego_trajectory_inspection import create_inspection
from experiment_runtime import ROOT, atomic_json, experiments_root
from run_ego_reference import DEFAULT_TARGET_TRACK
from run_ego_decision_agent import image_part
from run_ego_editor_agent import PROMPT, TOOLS, _arguments, _summary, _tool_contract
from streaming_requests import StreamingRequestArchive
from model_context import compact_model_context
from defense_refinement import refinement_state, VERSION as REFINEMENT_VERSION, GUIDANCE as REFINEMENT_GUIDANCE
from kimi_request_parameters import normalize_kimi_parameters
from ego_tool_transport import constrained_auto, validate_exposed_call

INSPECT = {'type':'function','function':{'name':'inspect_trajectory', 'description':'Inspect a generated map and candidate/parent trajectory diagram, not a camera image.', 'parameters':{'type':'object','properties':{'candidate_id':{'type':'string'}},'required':['candidate_id'],'additionalProperties':False}}}


@timed_operation('ego.visual_agent', 'run_dir')
def run(input_dir, constructor_run, foci_run, run_dir, timeout_seconds=240, max_turns=10, max_edit_candidates=5, max_inspections=2, point_only=False, max_tokens=6144, target_track=DEFAULT_TARGET_TRACK, conflict_brief=None, resume=False, allow_timeout_change=False, controlled_generation=False, target_effect=None, foci_enabled=True):
    run_dir = Path(run_dir).resolve()
    if (experiments_root()) not in run_dir.parents or (run_dir.exists() and not resume):
        raise ValueError('Visual run must be a new directory under experiments or explicitly resumed')
    editor = EgoEditor(input_dir, constructor_run, run_dir, foci_run=foci_run, target_track=target_track, resume=resume)
    brief = None
    if conflict_brief is not None:
        brief_dir = Path(conflict_brief).resolve()
        brief_config = json.loads((brief_dir/'config.json').read_text())
        if Path(brief_config['scene_dir']).resolve() != Path(input_dir).resolve() or brief_config['target_track'] != target_track:
            raise ValueError('Conflict brief must match the frozen scene and target')
        if json.loads((brief_dir/'summary.json').read_text()).get('status') != 'completed':
            raise ValueError('Conflict brief is not completed')
        brief = json.loads((brief_dir/'conflict-brief.json').read_text())
        from run_ego_conflict_brief import validate_brief
        validate_brief(brief, float(editor.times[-1]))
    archived = json.loads((run_dir/'config.json').read_text()) if resume else None
    context_view = archived.get('context_view_version') if resume else 'lossless_context.v1'
    if context_view != 'lossless_context.v1':
        raise ValueError('Unsupported archived context view version')
    candidate_view = archived.get('candidate_view_version') if resume else ('unranked_motion.v2' if controlled_generation else None)
    if candidate_view not in (None,'unranked_motion.v2'):
        raise ValueError('Unsupported candidate comparison view')
    editor.unranked_candidates = candidate_view is not None
    editor.behavior_enabled = candidate_view=='unranked_motion.v2'
    if candidate_view is not None:
        from defense_candidate_view import unranked_view, GUIDANCE
        if editor.behavior_enabled:
            from defense_candidate_view import BEHAVIOR_GUIDANCE
            GUIDANCE=GUIDANCE+' '+BEHAVIOR_GUIDANCE
    refinement_version=archived.get('refinement_workflow_version') if resume else (REFINEMENT_VERSION if controlled_generation and not point_only else None)
    if refinement_version not in (None,REFINEMENT_VERSION):
        raise ValueError('Unsupported refinement workflow')
    agent_prompt=PROMPT
    if refinement_version is not None:
        agent_prompt=PROMPT.replace('no required attempt count.','the refinement_workflow defines the editing attempt phase and its resource limits.')
        GUIDANCE=GUIDANCE.replace('minimum progress target or required number of edits.','minimum progress target; refinement_workflow defines the editing attempt phase.')
        GUIDANCE=GUIDANCE.replace('No fixed progress percentage, edit count or preferred candidate is imposed.','No fixed progress percentage or preferred candidate is imposed; use refinement_workflow for the attempt phase.')
    config = dict(acceptance_policy=editor.effective_policy, prompt_contract_version='goal_edit.v2', version=('ego_visual_editor.v21' if refinement_version else 'ego_visual_editor.v15' if candidate_view=='unranked_motion.v2' else 'ego_visual_editor.v14' if candidate_view else 'ego_visual_editor.v13') if controlled_generation else 'ego_visual_editor.v12', input_dir=str(Path(input_dir).resolve()),
        constructor_run=str(Path(constructor_run).resolve()),
        foci_run=None if foci_run is None else str(Path(foci_run).resolve()),
        target_track=target_track,
        conflict_brief=None if conflict_brief is None else str(brief_dir),
        timeout_seconds=timeout_seconds, max_turns=max_turns, max_edit_candidates=max_edit_candidates,
        max_inspections=max_inspections, point_only=point_only, max_tokens=max_tokens, first_inspection_required=True,
        image_scope='generated 2D reference trajectory and map; not a camera image',
        physical_execution='not_run', resume_supported=True, final_review_limit=1)
    if controlled_generation:
        config.update(controlled_generation=True,foci_enabled=foci_enabled,target_effect=target_effect)
    if refinement_version is not None:
        config['refinement_workflow_version']=refinement_version
    if candidate_view is not None:
        config['candidate_view_version'] = candidate_view
    if context_view is not None:
        config['context_view_version'] = context_view
    if resume:
        if archived != config:
            changed = {key for key in set(archived)|set(config) if archived.get(key) != config.get(key)}
            if not allow_timeout_change or changed != {'timeout_seconds'}:
                raise ValueError('Resume configuration differs from original run')
            history = run_dir/'status-history'; history.mkdir(exist_ok=True)
            atomic_json(history/f'timeout-change-{len(list(history.iterdir()))+1:02d}.json',
                dict(previous_timeout_seconds=archived['timeout_seconds'], timeout_seconds=timeout_seconds,
                    scope='new requests only; archived requests and original config remain unchanged'))
    else:
        atomic_json(run_dir/'config.json', config)
    enabled_tools = [tool for tool in TOOLS if not point_only or tool['function']['name'] != 'repair_candidate']
    generation=None
    generation_names=set()
    if controlled_generation and not point_only:
        from defense_generation import DefenseGeneration
        from defense_generation_contract import NAMES, TOOLS as GENERATION_TOOLS
        generation=DefenseGeneration(editor,max_edit_candidates,foci_enabled=foci_enabled)
        generation_names=set(NAMES)
        enabled_tools=enabled_tools+[tool for tool in GENERATION_TOOLS
            if foci_enabled or tool['function']['name']!='optimize_defense_candidate']
    if resume:
        if json.loads((run_dir/'tools.json').read_text()) != enabled_tools+[INSPECT]:
            raise ValueError('Resume tool schema differs from original run')
    else:
        atomic_json(run_dir/'tools.json', enabled_tools+[INSPECT])
    profile = load_llm_profile()
    client = OpenAI(api_key=profile['api_key'], base_url=profile['base_url'], max_retries=0,
        http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=float(timeout_seconds)))
    archive = StreamingRequestArchive(run_dir/'requests', client, profile)
    actions = invalid = 0
    inspections = []; deliveries = []; pending = None; pending_review = None; feedback = None; stop = 'model_turn_budget_exhausted'

    def manual_count():
        return sum(not row.get('automatic',False) for row in inspections)

    def inspect(candidate_id, automatic=False, closing_review=False):
        nonlocal pending
        number = len(inspections)+1
        path = run_dir/'inspections'/f'inspection_{number:02d}.png'
        record = create_inspection(editor,candidate_id,path)
        record.update(path=str(path.relative_to(run_dir)),inspection_id=number,automatic=automatic,closing_review=closing_review)
        inspections.append(record); pending = record
        atomic_json(run_dir/'inspections'/f'inspection_{number:02d}.json',record)
        return record

    if resume:
        old = json.loads((run_dir/'summary.json').read_text())
        if old['stop_reason'] != 'request_or_execution_failed' or old['selected_candidate_id'] is not None:
            raise ValueError('Only a failed, unsubmitted request can be resumed')
        actions = old['model_actions']; invalid = old['invalid_tool_calls']
        folders = sorted((run_dir/'tool-exchanges').glob('turn_*'))
        if [p.name for p in folders] != [f'turn_{i:02d}' for i in range(1, actions+1)]:
            raise ValueError('Archived tool actions are not contiguous')
        for folder in folders:
            attempts = list((run_dir/'requests').glob(folder.name+'_*/attempt_001'))
            completed = [a for a in attempts if (a/'response.json').is_file() and (a/'result.json').is_file()
                         and not (a/'error.json').exists()]
            if len(completed) != 1:
                raise ValueError('Expected one completed response for each action')
            request = json.loads((completed[0]/'request.json').read_text())
            response = json.loads((completed[0]/'response.json').read_text())
            calls = response['choices'][0]['message'].get('tool_calls') or []
            if request.get('endpoint') != profile['base_url'] or request['payload'].get('model') != profile['model'] or len(calls) != 1 or calls[0] != json.loads((folder/'call.json').read_text()):
                raise ValueError('Archived action provenance or model does not match')
        failed = list((run_dir/'requests').glob(f'turn_{actions+1:02d}_*/attempt_001'))
        if failed:
            if len(failed) > 2 or any(not (a/'error.json').is_file() or (a/'response.json').exists() for a in failed):
                raise ValueError('Resume requires an incomplete failed next request')
            prior_request = json.loads((failed[-1]/'request.json').read_text())
            if prior_request.get('endpoint') != profile['base_url'] or prior_request['payload'].get('model') != profile['model']:
                raise ValueError('Failed request model differs from the active profile')
        else:
            previous = list((run_dir/'requests').glob(f'turn_{actions:02d}_*/attempt_001'))
            if not (run_dir/'status-history').is_dir() or not any((a/'error.json').is_file() for a in previous):
                raise ValueError('No archived resumed turn explains the unsent next request')
            prior_request = None
        inspections = old['inspections']; deliveries = old['completed_image_deliveries']
        pending_review = old.get('pending_review_candidate_id')
        delivered = {d['inspection_id'] for d in deliveries}
        pending = next((i for i in inspections if i['inspection_id'] not in delivered), None)
        feedback = json.loads((folders[-1]/'result.json').read_text()) if folders else None
        history = run_dir/'status-history'; history.mkdir(exist_ok=True)
        snapshot = history/f'visual-before-resume-{len(list(history.iterdir()))+1:02d}.json'
        with snapshot.open('x') as stream: json.dump(old, stream, indent=2)
    resume_turn = actions+1 if resume else None
    try:
        for turn in range(actions+1, max_turns+1):
            visual = review_state(editor.nodes, inspections, deliveries, pending, pending_review, max_inspections)
            context = editor.model_context(max_turns-turn+1, max_edit_candidates-editor.edit_count, feedback, pending_review_candidate_id=pending_review)
            if generation is not None:
                context['generation_budget']=generation.context()
            if target_effect is not None:
                context['target_effect']=target_effect
            refinement=refinement_state(editor,generation.remaining() if generation is not None else max(0,max_edit_candidates-editor.edit_count),max_turns-turn+1,refinement_version is not None)
            if refinement_version is not None:
                context['refinement_workflow']=refinement
                context['refinement_task']=REFINEMENT_GUIDANCE
            if candidate_view is not None:
                context = unranked_view(context)
                context['candidate_order_note']='Generation order, no ranking or recommended ID.'
                context['comparison_guidance']=GUIDANCE
            if context_view is not None:
                context = compact_model_context(context)
            if brief is not None:
                context['conflict_understanding'] = brief
                context['conflict_understanding_scope'] = ('Advisory conflict and possible recorded-image lane analysis. '
                    'Recommendations do not create hard acceptance gates. Respect visible no-cross markings; '
                    'drivable-area checks do not verify markings or prove lane compliance.')
            context = dict(visual_submission_state=visual, **context)
            context['inspection_budget_remaining'] = visual['manual_inspections_remaining']
            if point_only:
                context['task'] = 'This experiment requires a trajectory edit after selecting and inspecting the sole initial candidate. Compare edits against the input task and keep a passing parent as fallback.'
                context['edit_modes'] = 'Use the exposed speed or route editing tools; repair is disabled. Their exact schemas and the shared acceptance policy apply.'
                context['search_guidance'] = 'Compare task-relevant tradeoffs against the parent; an edit need not improve every metric. A passing parent remains an eligible fallback after the required edit.'
            prompt = ('Read visual_submission_state first. Submit only a candidate whose own image is delivered or attached to this request. If your intended candidate needs review, inspect that exact ID; changing your mind requires an explicit new choice. Ranking labels never select the final candidate for you.\n'+agent_prompt+'\nThis experiment requires an initial trajectory inspection after selection. '
                'Further inspections are subject to the inspection budget and controller requirements. '
                'Diagrams depict map and reference geometry, not camera images. Numerical checks determine '
                'reference-screen eligibility; use diagram evidence only within its scope. Every generated edit returns a fresh diagram next turn without consuming a manual inspection. Describe which '
                'visible feature influenced your action and identify unresolved evidence.'
                '\nCurrent geometry-informed state:\n'+json.dumps(context, ensure_ascii=False, separators=(',',':')))
            if point_only:
                prompt = (PROMPT + "\nDirect-edit experiment: select the sole supplied initial candidate, "
                    "inspect it, then make at least one valid speed or route edit before submission. This requirement "
                    "tests editing; it does not prove the parent needs a change or require submitting the edit. "
                    "repair_candidate is disabled. Compare edited candidates and any passing parent fallback "
                    "against the input task using the shared policy, evidence and tool semantics above. "
                    "Further inspections follow the budget and controller requirements. Diagrams depict map "
                    "and reference geometry, not camera images; describe relevant features and uncertainties. "
                    "Numerical checks determine reference-screen eligibility.\nState:\n"
                    + json.dumps(context, ensure_ascii=False, separators=(',',':')))

            content = prompt if pending is None else [{'type':'text','text':prompt+'\nRequested diagram for '+pending['candidate_id']}, image_part(run_dir/pending['path'])]
            messages = [{'role':'user','content':content}]
            choice, tools = _tool_contract(editor, turn, max_turns, max_edit_candidates, require_edit=point_only)
            if generation is not None:
                if not generation.remaining():
                    tools=[t for t in tools if t['function']['name'] not in EDIT_TOOL_NAMES | {'repair_candidate'}]
                if editor.current is not None:
                    tools=tools+generation.tools()
            if point_only:
                tools = [tool for tool in tools if tool['function']['name'] != 'repair_candidate']
            if refinement['pending']:
                tools=[t for t in tools if t['function']['name']!='submit_candidate']
            choice, tools = visual_tool_contract(tools, INSPECT, visual,
                selected=editor.current, first_inspection=not inspections,
                closing=turn >= max_turns-1 and not refinement['pending'], final=turn == max_turns,
                edit_names=EDIT_TOOL_NAMES | generation_names, require_edit=point_only or refinement['pending'],
                edit_count=0 if refinement['pending'] else editor.edit_count)
            choice, tools = constrained_auto(choice, tools)
            if resume and prior_request is not None and turn == resume_turn:
                payload = prior_request['payload']
                expected = normalize_kimi_parameters(profile['model'], dict(messages=messages,
                    tools=tools, tool_choice=choice, max_tokens=max_tokens, reasoning_effort='high',
                    extra_body={'thinking':{'type':'enabled'}}, parallel_tool_calls=False))
                if any(payload.get(field) != expected.get(field) for field in (
                        'messages','tools','tool_choice','max_tokens','reasoning_effort','extra_body','parallel_tool_calls')):
                    raise ValueError('Restored request differs from the archived failed request')
            print('Visual editor turn', turn, 'started', flush=True)
            _, attempt = archive.complete(f'turn_{turn:02d}', messages, total_seconds=timeout_seconds,
                max_tokens=max_tokens, extra_body={'thinking':{'type':'enabled'}}, reasoning_effort='high',
                tools=tools, tool_choice=choice, parallel_tool_calls=False, allow_tool_calls=True)
            if pending is not None:
                deliveries.append(dict(inspection_id=pending['inspection_id'], candidate_id=pending['candidate_id'], request_turn=turn, request_path=str(attempt.relative_to(run_dir))))
                pending = None
                if pending_review in {d['candidate_id'] for d in deliveries}:
                    pending_review = None
            message = json.loads((attempt/'response.json').read_text())['choices'][0]['message']
            calls = message.get('tool_calls') or []; actions += 1
            if len(calls) != 1:
                stop = 'invalid_or_missing_tool_call'; break
            call = calls[0]; name = call['function']['name']
            try:
                args = _arguments(call)
                validate_exposed_call(name, args, tools)
                if name == 'select_constructor_candidate' and set(args) == {'candidate_id','reason'}:
                    result = editor.select_constructor(args)
                elif name == 'inspect_trajectory' and set(args) == {'candidate_id'}:
                    allowance = review_state(editor.nodes, inspections, deliveries, None, pending_review, max_inspections)
                    extra = allowance['manual_inspections_remaining'] == 0
                    if extra and (turn < max_turns-1 or not allowance['final_review_remaining']):
                        raise ValueError('Inspection budget exhausted; choose a submittable candidate or finish')
                    record = inspect(args['candidate_id'], closing_review=extra)
                    result = dict(status='inspection_ready', **record)
                elif name in generation_names:
                    result=generation.execute(name,args)
                elif name in EDIT_TOOL_NAMES | {'repair_candidate'}:
                    if point_only and name not in EDIT_TOOL_NAMES:
                        raise ValueError('Only trajectory-point editing is enabled')
                    if editor.edit_count >= max_edit_candidates or (generation is not None and not generation.remaining()):
                        raise ValueError('Edit budget exhausted')
                    expected = (set(args) if name in EDIT_TOOL_NAMES else {'parent_candidate_id','target_candidate_id','ratio','reason'})
                    if name in EDIT_TOOL_NAMES:
                        args = normalize_edit_call(name,args)
                        expected = set(args)
                    allowed = expected | ({'window_s'} if name == 'repair_candidate' else set())
                    if not expected <= set(args) <= allowed or not isinstance(args['reason'], str) or not args['reason'].strip():
                        raise ValueError('Invalid edit arguments')
                    result = editor.edit(dict(args, mode='point_edit' if name in EDIT_TOOL_NAMES else 'repair'))
                    if result.get('candidate_id') is not None:
                        result['automatic_inspection'] = inspect(result['candidate_id'],automatic=True)
                elif name == 'submit_candidate' and set(args) == {'candidate_id','reason'}:
                    if refinement['pending']:
                        raise ValueError('A changed candidate must be edited and checked before final comparison; the original fallback remains available afterward')
                    if point_only and not editor.edit_count:
                        raise ValueError('One valid edit is required in this experiment')
                    candidate_id = args['candidate_id']
                    state = review_state(editor.nodes, inspections, deliveries, None, pending_review, max_inspections)
                    if candidate_id in state['reference_passed_candidate_ids'] and candidate_id not in state['submittable_candidate_ids']:
                        pending_review = candidate_id
                        result = rejection(candidate_id, state)
                    else:
                        pending_review = None
                        result = editor.submit(args)
                elif name == 'finish_search' and set(args) == {'reason'} and isinstance(args['reason'], str) and args['reason'].strip():
                    pending_review = None
                    has_passing = any(n['result']['status'] == 'reference_screen_passed' for n in editor.nodes.values())
                    result = {'status':'finished','reason':args['reason'],
                        'conclusion':'submission_not_completed' if has_passing else 'no_qualified_candidate_found','global_feasibility_proven':False,
                        'tested_candidate_ids':[row['candidate_id'] for row in editor.history]}; stop = 'model_finished_without_submission'
                else:
                    raise ValueError('Unknown tool or invalid arguments')
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                result = {'status':'invalid','error':str(error)[:300]}
            except Exception as error:
                folder=run_dir/'tool-exchanges'/f'turn_{turn:02d}'
                folder.mkdir(parents=True,exist_ok=False)
                atomic_json(folder/'call.json',call)
                atomic_json(folder/'result.json',dict(status='execution_failed',error_type=type(error).__name__))
                raise
            if candidate_view is not None:
                result = unranked_view(result)
            folder = run_dir/'tool-exchanges'/f'turn_{turn:02d}'; folder.mkdir(parents=True, exist_ok=False)
            atomic_json(folder/'call.json', call); atomic_json(folder/'result.json', result)
            feedback = result
            print('Tool', name, 'returned', result['status'], flush=True)
            if result['status'] in {'invalid','rejected'}:
                invalid += 1
                if invalid >= 3:
                    stop = 'invalid_call_budget_exhausted'; break
            if editor.submitted:
                stop = 'candidate_submitted'; break
            if result['status'] == 'finished':
                break
    except KeyboardInterrupt:
        stop = 'interrupted_by_operator'
        raise
    except Exception:
        stop = 'request_or_execution_failed'
        raise
    finally:
        client.close()
        summary = _summary(editor, stop, actions, invalid, max_turns, max_edit_candidates, 3)
        if candidate_view is not None:
            summary=unranked_view(summary)
            summary['candidate_view_version']=candidate_view
            summary['selection_basis']='Explicit Agent submission; candidate order carries no quality rank.'
            if editor.submitted is not None and hasattr(editor,'_public'):
                summary['selected_candidate_evidence']=editor._public(editor.submitted)
        if refinement_version is not None:
            summary['refinement_workflow']=refinement_state(editor,generation.remaining() if generation is not None else max(0,max_edit_candidates-editor.edit_count),max(0,max_turns-actions),True)
        summary.update(pending_review_candidate_id=pending_review, visual_submission_state=review_state(editor.nodes, inspections, deliveries, pending, pending_review, max_inspections), inspection_calls=manual_count(),automatic_inspection_count=len(inspections)-manual_count(), inspections=inspections, completed_image_deliveries=deliveries,
            visual_input_used=bool(deliveries), visual_scope='generated map and trajectory diagram',
            visual_influence='not established without response analysis')
        if generation is not None:
            summary['generation_budget']=generation.context()
            summary['new_candidate_count']=editor.edit_count
            summary['direct_edit_count']=sum((node.get('operation') or {}).get('mode')=='point_edit'
                and not (node.get('operation') or {}).get('generation_source') for node in editor.nodes.values())
            summary['generated_candidate_counts']={source:sum((node.get('operation') or {}).get('generation_source')==source
                for node in editor.nodes.values()) for source in ('constructor','foci')}
        atomic_json(run_dir/'summary.json', summary)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--constructor-run', type=Path, required=True)
    parser.add_argument('--foci-run', type=Path)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=240)
    parser.add_argument('--point-only', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=6144)
    parser.add_argument('--target-track', required=True)
    parser.add_argument('--conflict-brief', type=Path, help='Completed conflict understanding run for this frozen scene')
    parser.add_argument('--controlled-generation', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--target-effect')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--allow-timeout-change', action='store_true')
    args = parser.parse_args()
    run(args.input, args.constructor_run, args.foci_run, args.run, timeout_seconds=args.timeout_seconds, point_only=args.point_only, max_tokens=args.max_tokens, target_track=args.target_track, conflict_brief=args.conflict_brief, resume=args.resume, allow_timeout_change=args.allow_timeout_change,
        controlled_generation=args.controlled_generation,target_effect=args.target_effect)
