"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='attack_vehicle_editor',
    description='Edit one attack vehicle and export the accepted frozen scene.',
    when_to_use='Apply a vehicle-edit strategy to a retrieved scene.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'For a ready tool04 result, use result.attack_input unchanged as the call arguments. New tool04 handoffs use operation=auto and plan_options: preserve them so the SubAgent can choose/switch actors inside this call. Existing direct fixed-operation inputs remain supported. VADv2 self-test is disabled by default; enable it explicitly only when requested. When revising after policy_runner shows an early safe stop or unmet maneuver failure, supply compact policy_behavior_summary (stopped_detected, progress_ratio_vs_recorded, signed_yaw_change_deg and task-specific failure_goal_met) in revision_feedback.evidence; on retry after timeout/interruption with unchanged inputs, pass prior run_dir as reuse_completed_from.', 'output': 'The internal attack SubAgent edits and returns a frozen attack scene plus trajectory. A completed envelope means geometry and task checks passed, not that the policy collides: verify with policy_runner(execution_mode=native) on the frozen scene. When vadv2_self_test=true the envelope also carries vadv2_probe_result_path and policy_collision_report, which may be reused only for that exact attack revision and test configuration. A subsequent expert-package evaluation is different. No legacy PDM.'},
    rules=[
        'New runs may edit at any scene time from 0s. No implicit 3-second edit or attack-event delay applies. Preserve a valid initial pose, continuity, road and motion checks; obey explicit task windows. Edited history is part of the exported scene.',
        'A success envelope may contain status=unable_to_complete: no frozen scene exists and PDM/ego editing cannot run on it. Return the measured reasons so the caller can revise its upstream plan (target, scene or requirement) and retry. Only status=completed provides a frozen scene.',
        'For auto, supply 1..3 coarse plan_options and a real vehicle asset up front when available; options are suggestions, not hard acceptance gates. The SubAgent may switch object/operation twice inside the same cumulative API, constructor and action budgets. Explicit edit_existing/add_vehicle requests remain binding.',
        'For edit_existing, supply an evidenced target_track or let the backend select it from annotations. For add_vehicle, provide the asset, dimensions and supported maneuver required by input_schema.',
        'By default the attack SubAgent only edits under geometric and task checks and returns a frozen attack scene; it does not test the policy. Set vadv2_self_test=true only when an in-editor check is required: it then runs the shared native VADv2 backend, only eligible candidates are tested, unchanged versions reuse results, and all branches share vadv2_probe_budget. Either way, road failure or PDM zero is not evidence that the target was hit.',
        'On editing_backend_not_configured, return the missing-backend evidence instead of inventing an attack; the caller decides the remedy.',
        'No internal PDM self-check or PDM defense-feasibility search is available in this tool. A completed attack is an editing result, not a policy score or policy-failure claim.',
        'A frozen scene without simulation_input is not an executable WorldEngine package.',
    ],
)
