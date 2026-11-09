"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='policy_runner',
    description='Run a policy in WorldEngine closed loop for evaluation or post-training data collection.',
    when_to_use='Evaluate prepared scenes, challenge a frozen attack with independent PDM, or collect post-training rollouts.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Use existing model config/checkpoint paths and native scene/asset paths. Merge an editor simulation_input only when it is non-null.', 'output': 'result_dir feeds failure_case_filter. rollout_dir and native dense rewards feed policy_trainer RLFT.'},
    rules=[
        'PDM mode simulates the edited scene only by default. challenge_baseline_scene_path is the original data source; challenge_run_baseline_diagnostic=true optionally runs the original scene for diagnosis, never as an acceptance gate.',
        "Validate new edited renders before policy evaluation; prepared native dataset scenes can run directly.",
        'For RLFT data collection enable with_dense_reward_manager; ordinary evaluation does not imply candidate rewards are present.',
        'Match history/future frame counts to the native scenario. Do not pad missing observations.',
        'In pdm_closed_loop mode, pass the accepted attack pdm_probe_result_path as challenge_probe_result_path when available. The backend verifies exact scene/history/target/runtime/map semantics before reusing its real simulation; a mismatch is an error, never a successful test.',
    ],
)
