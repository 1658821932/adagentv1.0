"""Tool purpose, invocation context and workflow handoff; fields come from Schema."""
from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(
    name='policy_runner',
    description='Run a policy in WorldEngine closed loop for evaluation or post-training data collection.',
    when_to_use='Test a frozen attack against the driving policy, evaluate prepared scenes, or collect post-training rollouts.',
    input_model=Input,
    output_model=Output,
    handoff={'input': 'Pass the frozen scene/asset paths from tool05 attack simulation_input and its target_track as collision_target_track. Native model config/checkpoint/model_name/data_type are optional: omit them to use the fixed configured adapter policy. Supply overrides only for an explicitly requested different policy.', 'output': 'Read official pdm_subscores, motion measurements in policy_behavior_summary and collision_report. Progress and yaw ratios impose no acceptance threshold or automatic maneuver verdict; turn_completed is deprecated and null. Evaluate target_effect against the scene and actual motion; a safe wait can be appropriate. For an explicitly requested attack test, actual target contact after takeover is attack success; PDM zero alone is not collision evidence. result_dir feeds failure_case_filter, and eligible dense-reward exports feed RLFT.'},
    rules=[
        'PDM mode simulates the edited scene only by default. challenge_baseline_scene_path is the original data source; challenge_run_baseline_diagnostic=true optionally runs the original scene for diagnosis, never as an acceptance gate.',
        'For the normal full pipeline, evaluate tool05 simulation_input with collision_target_track and its shared 0–1.5s history and frozen navigation. Full scoring/video collection is separate from the light internal attack probe.',
        'Omit config_path, checkpoint_path, model_name and data_type to use the adapter defaults. Actual model files are checked by the backend before execution; never invent paths or ask the user for overrides merely because these optional fields are absent.',
        'Normal native scoring and RL collection use the same 1.5-7.5s window at 2Hz: num_history=4, max_frame_per_scene=16, num_future=12 or omitted, sampling_poses omitted. Dense collection needs 24 real source frames including its 8-frame lookahead. Only change this window when the user explicitly requests it; never extend scoring to the whole source scene as a fallback after export failure.',
        'with_dense_reward_manager is on by default, so every native run also exports an RLFT package; set it false only for a pure score-only diagnosis. Native collection exports validated annotations, camera bindings, actual plan indices and candidate rewards. no_eligible_scenes means no official PDM-zero training scene; skip training.',
        'target_collision_detected=null or incomplete coverage is unknown; interpolated possible contact and pre-takeover contact do not establish VAD failure.',
        'Match history/future frame counts to the native scenario. Do not pad missing observations.',
        'In pdm_closed_loop mode, pass the accepted attack pdm_probe_result_path as challenge_probe_result_path when available. The backend verifies exact scene/history/target/runtime/map semantics before reusing its real simulation; a mismatch is an error, never a successful test.',
    ],
)
