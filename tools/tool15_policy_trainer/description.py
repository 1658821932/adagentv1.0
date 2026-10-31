"""Single source of training tool instructions shown to the Agent."""
DESCRIPTION = {
    "name": "policy_trainer",
    "description": "Fine-tune the driving policy with imitation learning (ilft) or reinforcement learning (rlft).",
    "when_to_use": "Use ilft for trajectory annotations. Use rlft only after WorldEngine exports complete rollout images, metadata and 8192-candidate PDM rewards. Scene packages and videos alone are not RL training data. Evaluate the new checkpoint independently before accepting it.",
    "inputs": {
        "training_mode": "ilft (default) or rlft. Missing RL inputs cause an error; never silently fall back to IL.",
        "training_data": "ilft: {ann_file, token_file}. rlft: {rollout_dir, token_file, vocab_path, sample_filter?}. rollout_dir is the WE openscene_format directory with meta_datas/, sensor_blobs/, pdms_pkl/ including plan_idx.csv. vocab_path is the (8192,40,3) vocabulary used to compute those rewards. sample_filter=v2 selects first decisions; v1 selects hard frames.",
        "base_checkpoint": "Existing policy checkpoint. Record the architecture/config when comparing checkpoints.",
        "config_path": "Optional mode-appropriate AlgEngine config; defaults to existing IL config or native RL rollout config.",
        "epochs": "Positive epoch count, default 1.",
        "seed": "Nonnegative random seed, default 0.",
        "worldengine_root": "Optional WorldEngine code root.",
        "python": "Optional AlgEngine interpreter.",
        "work_dir": "Optional export parent; execution always uses an isolated runtime directory.",
        "timeout": "Positive per-process timeout in seconds."
    },
    "outputs": {
        "training_mode": "The actual mode used.",
        "checkpoint": "Checkpoint saved by real training.",
        "log_path": "Training log.",
        "training_audit": "Mode-specific losses, finite-value check and actual parameter updates.",
        "data_preparation": "Training inputs and rollout preparation summary."
    }
}

from .schemas import Input, Output
DESCRIPTION["input_schema"] = Input.model_json_schema()
DESCRIPTION["output_schema"] = Output.model_json_schema()
