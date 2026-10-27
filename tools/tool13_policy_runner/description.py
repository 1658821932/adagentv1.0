"""
Tool description for the policy_runner tool (tool13_policy_runner).

This is the single source of truth shown to the LLM orchestrator when it
decides whether and how to call this tool. Keep orchestrator-facing rules
about this tool here, not in the global system prompt.
"""

DESCRIPTION = {
    "name": 'policy_runner',
    "description": 'Load a WorldEngine scenario and run the autonomous driving policy in one closed-loop execution for imitation-learning data generation.',
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
        'max_frame_per_scene': 'Exactly 40; closed-loop simulation uses only the first 40 frames.',
        'sampling_poses': 'Optional native rollout timing override in 0.5-second units.',
    },
    'outputs': {
        'results': 'WorldEngine closed-loop result, output directory and runner reports.',
    },
}
