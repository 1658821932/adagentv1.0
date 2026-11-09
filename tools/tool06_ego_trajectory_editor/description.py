"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='ego_trajectory_editor',
    description='Generate a correct ego expert trajectory for the frozen attack scene.',
    when_to_use='Create the expert reference after attack editing.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'attack_result: successful attack editor result_ref, result or envelope. target_effect: the one global scene_instance_generator.target_effect.', 'output': 'Use non-null simulation_input for policy_runner; non-null ilft_training_data for policy_trainer ILFT. Existing backend render paths remain artifacts, not visual acceptance.'},
    rules=[
        'FOCI is enabled by default. The internal editor can request structured construction or FOCI optimization; all new candidates share its bounded generation/edit budget and unchanged acceptance checks.',
        'A success envelope with status=defense_not_found means completed bounded defense search: return compact feedback to attack editing on the original instruction and hard contract. An execution error does not justify lowering attack difficulty.',
        'Preserve the frozen attack scene and its target vehicle.',
        'An expert trajectory is a reference, not a policy closed-loop success. Its CSV is not ann_file training data; use the explicit converted handoff blocks.',
        'On editing_backend_not_configured, stop.',
    ],
)
