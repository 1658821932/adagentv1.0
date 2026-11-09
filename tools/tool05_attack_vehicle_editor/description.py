"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='attack_vehicle_editor',
    description='Edit one attack vehicle and export the accepted frozen scene.',
    when_to_use='Apply a vehicle-edit strategy to a retrieved scene.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'For a ready tool04 result, use result.attack_input unchanged as the call arguments. Existing direct scene_ref/instruction/operation inputs remain supported.', 'output': 'For adversarial generation, run policy_runner(execution_mode=pdm_closed_loop) on this frozen version first. Safe PDM defense returns to this tool with revision_feedback; failed PDM defense permits ego_trajectory_editor.attack_result. Non-null simulation_input can supply native scene/asset paths to policy_runner.'},
    rules=[
        'A success envelope may contain status=unable_to_complete: report bounded reasons/evidence and end the task without PDM or ego editing. Only status=completed provides a frozen scene.',
        'For edit_existing, supply an evidenced target_track or let the backend select it from annotations. For add_vehicle, provide the asset, dimensions and supported maneuver required by input_schema.',
        'When PDM defended safely, call again on the original scene with unchanged instruction/task_contract and compact revision_feedback containing the prior placement and measured PDM outcome. Search a more challenging alternative within the attack-attempt budget; do not freeze an extra copy of the previous added vehicle.',
        'On editing_backend_not_configured, stop: the stage has no configured algorithm backend.',
        'Set pdm_self_check=true to expose probe_pdm inside editing. Geometry-passing candidates must be tested; safely defended candidates need revision or placement reselection. pdm_probe_budget is shared across branches. Return pdm_probe_result_path for verified tool08 reuse.',
        'pdm_defense_feedback=true adds advisory braking and bounded constructor witnesses to probe_pdm. Prefer preserving a tested defense witness while challenging PDM; no witness is not global infeasibility or a new attack gate.',
        'A frozen scene without simulation_input is not an executable WorldEngine package.',
    ],
)
