"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='attack_vehicle_editor',
    description='Edit one attack vehicle and export the accepted frozen scene.',
    when_to_use='Apply a vehicle-edit strategy to a retrieved scene.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'scene_ref: retrieval result_ref or unchanged scene_ref. instruction: selected tool02 candidate.edit_strategy.', 'output': 'Pass this call result_ref to ego_trajectory_editor.attack_result. Non-null simulation_input can supply native scene/asset paths to policy_runner.'},
    rules=[
        'For edit_existing, supply an evidenced target_track or let the backend select it from annotations. For add_vehicle, provide the asset, dimensions and supported maneuver required by input_schema.',
        'On editing_backend_not_configured, stop: the stage has no configured algorithm backend.',
        'A frozen scene without simulation_input is not an executable WorldEngine package.',
    ],
)
