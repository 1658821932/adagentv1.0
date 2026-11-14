"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='policy_trainer',
    description='Post-train a driving policy using imitation learning (ilft) or reinforcement learning (rlft).',
    when_to_use="Train when the requested mode's converted data is ready.",
    input_model=Input,
    output_model=Output,
    handoff={'input': 'ILFT: training_data.ann_file and token_file. RLFT: pass a ready policy_runner rlft_training_data bundle unchanged, including manifest_path; the trainer verifies annotation and reward provenance.', 'output': 'Use the returned checkpoint in a subsequent policy_runner call to measure post-training performance.'},
    rules=[
        'RLFT requires rollout images/metadata, native 8192-candidate PDM rewards and plan_idx.csv. It keeps zero-PDM failure scenes and all their valid frames.',
        'Videos, frozen metadata or an expert CSV alone are not training_data. Do not silently switch RLFT to ILFT.',
        'A saved checkpoint and training_audit establish execution; improvement requires a separate evaluation.',
    ],
)
