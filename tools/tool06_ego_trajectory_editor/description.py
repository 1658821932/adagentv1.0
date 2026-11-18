"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='ego_trajectory_editor',
    description='Generate a correct ego expert trajectory for the frozen attack scene.',
    when_to_use='Create the expert reference after attack editing.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'attack_result: exact successful attack result_ref, result or envelope. Tool05 normally includes an exact-version native VAD collision self-test and freezes the shared 0–1.5s history. Preserve that history and takeover state. target_effect: the one global scene_instance_generator.target_effect.', 'output': 'Return expert trajectory, expert_motion_summary, checks, optional video and simulation_input to Main Agent. Keep this expert trajectory separate from the frozen tool05 package used for VAD evaluation. Tool07 tests the tool05 package independently and can run in parallel with this editor; the expert future must not replace the policy test input. Current backend returns ilft_training_data=null: no converted ILFT annotations are exported; never pass the expert CSV as ann_file.'},
    rules=[
        'Selecting an initial candidate automatically supplies its trajectory diagram in the next model request; manual inspection remains available for other candidates. A diagram is not recorded as delivered until that request completes. Completed experts return measured clearance and motion headroom as expert_feedback for Main to decide whether one same-task attack revision is worthwhile; no new minimum-gap gate is imposed.',
        'Preserve the complete shared ego history through 1.5s, including the takeover pose. Edit only after 1.5s and pass motion/road/safety checks. Both VAD and the expert use the same tool05 history; do not repair that prefix by moving its poses.',
        'FOCI is available by default but is not run during initial preparation. The internal editor can request FOCI only for a concrete improvement need; all new candidates share its bounded generation/edit budget and unchanged acceptance checks.',
        'A success envelope with status=defense_not_found means completed bounded defense search: return compact feedback to attack editing on the original instruction and hard contract. An execution error does not justify lowering attack difficulty.',
        'Distinguish the final task goal from intermediate actions: stopping to wait and then proceeding is a driving goal, not a terminal-stop task. require_route_progress is an optional intent hint, inferred when null; it does not add a progress percentage or automatic maneuver-completion gate. Review the actual target_effect against scene evidence and measured progress/safety; safe waiting alone is not proof of failure.',
        'Preserve the frozen attack scene and its target vehicle.',
        'An expert trajectory is a reference, not a policy closed-loop success. Its CSV is not ann_file training data; use the explicit converted handoff blocks.',
        'On editing_backend_not_configured, return the missing-backend evidence; the caller decides the remedy.',
        'For an explicit retry after preparation_timeout or job_interrupted, pass the returned run_dir as reuse_completed_from. Completed understanding, constructor and initial FOCI checkpoints are reused only after content validation; missing or stale stages are recomputed. Never reuse a different task or attack revision.',
    ],
)
