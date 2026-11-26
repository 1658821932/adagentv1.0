"""Stage-level JSON bridge; no real editor is bundled or simulated here."""
from __future__ import annotations
import csv
import json
import math
import os
import re
from pathlib import Path

from pydantic import ValidationError
from api_config import get_profile
from mcp_servers.process import run_logged
from mcp_servers.runtime import current_run, runtime_task, write_json
from mcp_servers.scene_editing.shared_history import shared_history_end_index
from .config import PROJECT_ROOT, llm_profile_name, stage_command, public_pipeline_root
from .schemas import AttackInput, AttackOutput, EgoInput, EgoOutput, AttackRefusal, DefenseNotFound

class EditingError(RuntimeError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


def _file(value):
    p = Path(value).expanduser()
    if not p.is_absolute() or not p.is_file():
        raise EditingError("editing_artifact_missing", f"Expected an existing absolute file: {p}")
    return p.resolve()


def _inside(value, root):
    p = Path(value).expanduser()
    if not p.is_absolute() or not p.resolve().is_relative_to(root.resolve()):
        raise EditingError("editing_output_invalid", f"New artifacts must be inside {root}: {value}")
    return _file(value)


def _timeline(path):
    frames = json.loads(_file(path).read_text())
    if not isinstance(frames, list) or not frames:
        raise EditingError("editing_input_invalid", "frames.json must contain a non-empty frame list")
    times = [int(f["timestamp"]) for f in frames]
    if any(b <= a for a,b in zip(times,times[1:])):
        raise EditingError("editing_input_invalid", "Frame timestamps must increase strictly")
    return frames


def _handoffs(output, task):
    public_root = public_pipeline_root()
    for value in output.rendered_videos.values():
        try:
            _inside(value, task.output)
        except EditingError:
            if public_root is None:
                raise
            _inside(value, public_root)
    if output.simulation_input is not None:
        s = output.simulation_input
        if Path(s.scenario_pkl_file_name).name != s.scenario_pkl_file_name:
            raise EditingError("editing_output_invalid", "scenario_pkl_file_name must be a basename")
        _inside(str(Path(s.scenario_data_folder_path)/s.scenario_pkl_file_name),task.output)
        if not Path(s.asset_folder_path).is_absolute() or not Path(s.asset_folder_path).is_dir():
            raise EditingError("editing_artifact_missing", "Native simulation asset directory is missing")
    labels = getattr(output, "ilft_training_data", None)
    if labels is not None:
        _inside(labels.ann_file,task.output)
        _inside(labels.token_file,task.output)


def _validate_attack(data, result, task):
    result = AttackOutput.model_validate(result)
    source = data.scene_ref
    if result.mode != "real" or result.scene_id != source.scene_id or result.scene_token != source.scene_token:
        raise EditingError("editing_identity_mismatch", "Attack result must retain source scene identity")
    if result.source_scene_ref.model_dump() != source.model_dump():
        raise EditingError("editing_identity_mismatch", "Attack result changed source_scene_ref")
    if data.operation == "edit_existing" and data.target_track is not None and result.target_track != data.target_track:
        raise EditingError("editing_identity_mismatch", "Submitted attack track differs from requested track")
    frozen = Path(result.frozen_scene_path)
    if data.operation=='auto' and result.operation not in ('edit_existing','add_vehicle'):
        raise EditingError('editing_identity_mismatch','Auto selection must report its actual operation')
    if data.operation!='auto' and result.operation is not None and result.operation!=data.operation:
        raise EditingError('editing_identity_mismatch','Attack operation differs from explicit caller requirement')
    if not frozen.is_absolute() or not frozen.resolve().is_relative_to(task.output.resolve()):
        raise EditingError("editing_output_invalid", "Frozen scene must be a new task output directory")
    frames_path = _inside(result.frames_path, frozen)
    original, edited = _timeline(source.frames_path), _timeline(frames_path)
    if len(original) != len(edited) or [f["timestamp"] for f in original] != [f["timestamp"] for f in edited]:
        raise EditingError("editing_identity_mismatch", "Attack editing must retain the complete source timeline")
    history=frozen/'ego-history.json'
    expected=original
    if history.is_file():
        import hashlib
        proof=json.loads(history.read_text())
        if (Path(proof['source_scene']).resolve()!=Path(source.sample_root).resolve()
                or proof['source_frames_sha256']!=hashlib.sha256(Path(source.frames_path).read_bytes()).hexdigest()):
            raise EditingError('editing_identity_mismatch','Ego history belongs to another source scene')
        prepared=_inside(str(Path(proof['prepared_scene'])/'metadata/frames.json'),task.output)
        if hashlib.sha256(prepared.read_bytes()).hexdigest()!=proof['edited_frames_sha256']:
            raise EditingError('editing_identity_mismatch','Prepared ego history has changed')
        expected=_timeline(prepared)
    for before, after in zip(expected, edited):
        for key in ("token", "ego2global", "ego2global_translation", "ego2global_rotation"):
            if key in before and before.get(key) != after.get(key):
                raise EditingError("editing_identity_mismatch", f"Attack editing changed source {key}")
    for name in ("trajectory_path", "provenance_path", "submission_path"):
        val = getattr(result, name)
        if val is None and name == "trajectory_path" and result.operation == "ego_only":
            continue
        _inside(val, task.output)
    _inside(result.checks.report_path,task.output)
    if result.pdm_probe_result_path is not None:
        _inside(result.pdm_probe_result_path,task.output)
    _handoffs(result,task)
    return result.model_dump()


def _validate_ego(data, result, task):
    result = EgoOutput.model_validate(result)
    attack = data.attack_result
    if result.mode != "real" or (result.scene_id,result.scene_token,result.target_track) != (attack.scene_id,attack.scene_token,attack.target_track):
        raise EditingError("editing_identity_mismatch", "Ego result must retain attack scene and target identity")
    if Path(result.frozen_scene_path).resolve() != Path(attack.frozen_scene_path).resolve() or Path(result.attack_run_dir).resolve() != Path(attack.run_dir).resolve():
        raise EditingError("editing_identity_mismatch", "Ego trajectory belongs to a different attack revision")
    trajectory = _inside(result.expert_trajectory_path,task.output)
    for name in ("trajectory_metadata_path", "conflict_brief_path", "submission_path"):
        _inside(getattr(result,name),task.output)
    _inside(result.checks.report_path,task.output)
    with trajectory.open() as f:
        rows = list(csv.DictReader(f))
    columns = ("time_s", "ego_x_m", "ego_y_m", "ego_yaw_rad")
    if not rows or any(key not in rows[0] for key in columns):
        raise EditingError("editing_output_invalid", "Expert CSV requires time_s, ego_x_m, ego_y_m, ego_yaw_rad")
    try:
        values = [[float(row[key]) for key in columns] for row in rows]
    except (ValueError,TypeError) as e:
        raise EditingError("editing_output_invalid", "Expert CSV contains invalid numbers") from e
    if not all(math.isfinite(v) for row in values for v in row):
        raise EditingError("editing_output_invalid", "Expert CSV contains non-finite values")
    frames = _timeline(attack.frames_path)
    times = [(f["timestamp"]-frames[0]["timestamp"])/1_000_000 for f in frames]
    if len(values) != len(times) or any(abs(row[0]-t)>0.001 for row,t in zip(values,times)):
        raise EditingError("editing_identity_mismatch", "Expert trajectory must align with frozen frame timestamps")
    first=frames[0]['ego2global'];angle0=math.atan2(first[1][0],first[0][0])
    c,s=math.cos(angle0),math.sin(angle0)
    history_count = shared_history_end_index(times) + 1
    for row, frame in zip(values[:history_count], frames[:history_count]):
        pose=frame['ego2global'];dx,dy=pose[0][3]-first[0][3],pose[1][3]-first[1][3]
        yaw=math.atan2(pose[1][0],pose[0][0])-angle0
        if (abs(row[1]-(c*dx+s*dy))>1e-6 or abs(row[2]-(-s*dx+c*dy))>1e-6
                or abs(math.atan2(math.sin(row[3]-yaw),math.cos(row[3]-yaw)))>1e-6):
            raise EditingError('editing_identity_mismatch','Expert changed the shared 0–1.5s attack history')
    meta = json.loads(Path(result.trajectory_metadata_path).read_text())
    expected = {"coordinate_frame":"initial_ego_rear_axle", "position_reference":"ego_rear_axle", "length_unit":"m", "angle_unit":"rad", "time_unit":"s"}
    if any(meta.get(k)!=v for k,v in expected.items()):
        raise EditingError("editing_output_invalid", "Expert trajectory coordinate/unit metadata is missing or incompatible")
    if result.expert_feedback is not None:
        import hashlib
        path=(result.expert_feedback.get('measurements') or {}).get('feedback_path')
        report=json.loads(_inside(path,task.output).read_text())
        if (report.get('schema_version')!='expert_attack_feedback.v1'
                or report.get('candidate_id')!=result.selected_candidate_id
                or Path(report['frozen_scene_path']).resolve()!=Path(attack.frozen_scene_path).resolve()
                or report['frames_sha256']!=hashlib.sha256(Path(attack.frames_path).read_bytes()).hexdigest()
                or Path(report['expert_trajectory_path']).resolve()!=trajectory.resolve()
                or report['expert_trajectory_sha256']!=hashlib.sha256(trajectory.read_bytes()).hexdigest()):
            raise EditingError('editing_identity_mismatch','Expert feedback differs from the exact attack/expert version')
        _inside(report['expert_review_path'],task.output)
    _handoffs(result,task)
    return result.model_dump()


def backend_api_env(command, environment, component="attack_editor"):
    """Inject project credentials only into the bundled backend, never persist them."""
    result = dict(environment)
    backend = Path(__file__).resolve().parents[1] / 'core/adversarial/editing_backend.py'
    if any(Path(part).resolve() == backend for part in command):
        result['ADVERSARIAL_HOST_HOOKS'] = 'mcp_servers.scene_editing.adapter.core_hooks'
        result.pop('ADVERSARIAL_ATTACK_SESSION', None)
        job_dir = os.getenv('ADAGENT_JOB_DIR')
        if component == 'attack_editor' and job_dir:
            manifest = json.loads((Path(job_dir)/'manifest.json').read_text())
            scope = manifest.get('scope')
            if not isinstance(scope,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',scope):
                raise ValueError('Attack budget requires a valid execution scope')
            result['ADVERSARIAL_ATTACK_SESSION'] = str(Path(job_dir).parent/'attack-budgets'/scope)
        public_root = public_pipeline_root()
        # Clear inherited values: this path is owned exclusively by the adapter.
        result.pop('ADVERSARIAL_FULL_PIPELINE_ROOT', None)
        if public_root is not None:
            result['ADVERSARIAL_FULL_PIPELINE_ROOT'] = str(public_root)
        from common.api_profiles import effort_for, profile_argument, reasoning_effort_level
        profile = get_profile(profile_argument(component), require_key=True)
        # A tool-level thinking level pins this editor; otherwise the global value applies.
        tool_effort = effort_for(component) or reasoning_effort_level()
        if tool_effort:
            result['LLM_REASONING_EFFORT'] = tool_effort
        if tool_effort and profile.get('request_dialect'):
            level_to_effort = {'thinking': {'low': 'low', 'medium': 'high', 'high': 'max'},
                               'qwen': {'low': 'low', 'medium': 'medium', 'high': 'xhigh'},
                               'kimi': {'low': 'low', 'medium': 'high', 'high': 'max'}}
            mapped = level_to_effort.get(profile['request_dialect'], {}).get(tool_effort)
            if mapped:
                profile = dict(profile, reasoning_effort=mapped)
        result.pop('ADVERSARIAL_REQUIRED_TOOL_THINKING', None)
        if profile.get('required_tool_thinking'):
            result['ADVERSARIAL_REQUIRED_TOOL_THINKING'] = profile['required_tool_thinking']
        result.pop('ADVERSARIAL_IMAGE_TOOL_CHOICE', None)
        if profile.get('image_tool_choice'):
            result['ADVERSARIAL_IMAGE_TOOL_CHOICE'] = profile['image_tool_choice']
        result.update(ADVERSARIAL_API_KEY=profile['api_key'],
                      ADVERSARIAL_API_BASE_URL=profile['base_url'], ADVERSARIAL_API_MODEL=profile['model'],
                      ADVERSARIAL_API_TIMEOUT=str(profile['timeout']),
                      ADVERSARIAL_API_MAX_TOKENS=str(profile['max_tokens']))
        for field, name in [('reasoning_effort','ADVERSARIAL_REASONING_EFFORT'), ('request_dialect','ADVERSARIAL_REQUEST_DIALECT'), ('thinking_mode','ADVERSARIAL_THINKING_MODE'), ('tool_choice_policy','ADVERSARIAL_TOOL_CHOICE_POLICY')]:
            result.pop(name, None)
            if profile.get(field):
                result[name] = str(profile[field])
    return result


def _execute(stage, arguments):
    task = current_run("scene_editing")
    command = stage_command(stage)
    if command is None:
        raise EditingError("editing_backend_not_configured", f"Configure scene_editing/adapter/settings.json commands.{stage}; the new editing backend is not connected")
    data = (AttackInput if stage=="attack_editing" else EgoInput).model_validate(arguments)
    if stage=="attack_editing":
        _timeline(data.scene_ref.frames_path)
        _file(data.scene_ref.cameras_path)
        if data.vehicle is not None:
            _file(data.vehicle.asset_path)
    else:
        if data.attack_result.mode != "real":
            raise EditingError("editing_input_invalid", "Real ego editing cannot consume mock attack artifacts")
        _timeline(data.attack_result.frames_path)
    input_frames = Path(data.scene_ref.frames_path if stage=="attack_editing" else data.attack_result.frames_path)
    source_frames = _timeline(input_frames)
    payload = {"schema_version":"editing_stage.v1", "stage":stage,
               "arguments":data.model_dump(), "output_dir":str(task.output)}
    write_json(task.input / "stage_request.json",payload)
    completed = run_logged(command,cwd=PROJECT_ROOT,env=backend_api_env(command, task.child_env(), "attack_editor" if stage=="attack_editing" else "ego_editor"),input=json.dumps(payload,ensure_ascii=False),json_stdout=True,log_path=task.logs/"backend.log",timeout=data.timeout)
    try:
        current_frames = _timeline(input_frames)
    except EditingError as error:
        raise EditingError("editing_input_modified", "Source frames became unreadable during backend execution") from error
    if len(source_frames) != len(current_frames) or any(
            before.get("timestamp") != after.get("timestamp") or before.get("token") != after.get("token")
            or before.get("ego2global") != after.get("ego2global")
            or before.get("ego2global_translation") != after.get("ego2global_translation")
            or before.get("ego2global_rotation") != after.get("ego2global_rotation")
            for before, after in zip(source_frames, current_frames)):
        raise EditingError("editing_input_modified", "Backend changed source frame identity or ego poses")
    if completed.returncode:
        raise EditingError("editing_backend_failed", "Editing backend exited unsuccessfully; see backend.log")
    try:
        result = json.loads(completed.stdout)
        if not isinstance(result,dict):
            raise ValueError("Backend response must be an object")
        write_json(task.logs/"stage_response.json",result)
        business_type = AttackRefusal if stage == "attack_editing" else DefenseNotFound
        if result.get("status") == ("unable_to_complete" if stage == "attack_editing" else "defense_not_found"):
            result["mode"], result["run_dir"] = "real", str(task.root)
            business = business_type.model_validate(result)
            for path in business.evidence_paths:
                _inside(path, task.output)
            return business.model_dump()
        if result.get("status") == "backend_error":
            error_code = ('editing_configuration_error' if result.get('error_code') == 'editing_configuration_error'
                          else 'editing_execution_failed')
            details = dict(backend_error_type=result.get('error_type', 'unknown'),
                failure_stage=result.get('failure_stage', 'execution'),
                run_dir=str(task.root), log_path=str(task.logs/'stage_response.json'))
            for key in ('retryable', 'error_details', 'recovery_hint'):
                if key in result:
                    details[key] = result[key]
            reason = str(result.get('error') or 'No backend error detail')[:600]
            raise EditingError(error_code,
                f"Backend execution failed ({details['backend_error_type']}): {reason}; see {task.logs/'stage_response.json'}", **details)
        if result.get("status") != "completed":
            raise EditingError("editing_not_completed", f"Backend did not submit a candidate: {result.get('status')}; see stage_response.json")
        result["mode"], result["run_dir"] = "real", str(task.root)
        return (_validate_attack if stage=="attack_editing" else _validate_ego)(data,result,task)
    except EditingError:
        raise
    except (ValidationError,ValueError,KeyError,TypeError) as e:
        raise EditingError("editing_output_invalid", str(e)) from e

@runtime_task("scene_editing","attack_editing")
def run_attack_editing(arguments):
    return _execute("attack_editing",arguments)

@runtime_task("scene_editing","ego_editing")
def run_ego_editing(arguments):
    return _execute("ego_editing",arguments)
