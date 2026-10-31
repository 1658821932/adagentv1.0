"""
Tool description for the policy_runner tool (tool13_policy_runner).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'policy_runner',
    "description": 'Run a driving policy in WorldEngine closed loop for evaluation or IL/RL data collection.',
    "when_to_use": 'Use after scene validation succeeds to load the scenario, start SimEngine and run the policy.',
    'inputs': {
        'config_path': 'Absolute existing AlgEngine config. Standard: <WORLDENGINE_ROOT>/projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py.',
        'checkpoint_path': 'Absolute existing policy checkpoint. Standard: <WORLDENGINE_ROOT>/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth.',
        'model_name': 'Experiment model name.',
        'data_type': 'WorldEngine scenario data type.',
        'react_type': 'R or NR.',
        'asset_name': 'Optional renderer asset name.',
        'added_name': 'Optional output suffix.',
        'num_scenarios': 'Optional positive scenario count.',
        'scenario_data_folder_path': 'Optional scenario directory; supports custom/DGGT-converted scenarios.',
        'asset_folder_path': 'Optional scenario asset directory.',
        'scenario_pkl_file_name': 'Optional scenario pickle filename.',
        'dggt_output_dir': 'Optional DGGT edited output directory; policy_runner automatically stages the selected instance and converts it to WorldEngine data.',
        'instance_id': 'DGGT instance to stage when dggt_output_dir is provided; defaults to 0.',
        'translation_scale': 'WorldEngine metres per DGGT reconstruction unit; defaults to 10000 for normalized DGGT outputs.',
        'num_history': 'Optional history frame count.',
        'num_future': 'Optional future frame count.',
        'max_frame_per_scene': 'Positive frame budget (at least 5), default 40. num_history + num_future must equal this budget; future is derived when omitted. Never pad observations.',
        'with_dense_reward_manager': 'Set true to collect native 8192-candidate rewards for RL; false preserves ordinary IL/evaluation collection.',
        'sampling_poses': 'Optional native rollout timing override in 0.5-second units.',
    },
    'outputs': {
        'result_dir': 'WorldEngine closed-loop output directory.',
        'reports': 'Native runner reports.',
        'rollout_dir': 'WE openscene_format directory; pass as training_data.rollout_dir to RL training.',
        'reward_dir': 'Candidate rewards and actual plan_idx.csv when dense rewards were requested.',
    },
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
DESCRIPTION["inputs"] = {key: DESCRIPTION["inputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["input_schema"]["properties"].items()}

DESCRIPTION["outputs"] = {key: DESCRIPTION["outputs"].get(key, value.get("description", value.get("title", key))) for key,value in DESCRIPTION["output_schema"]["properties"].items()}
