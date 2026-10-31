"""Artifact contracts for simulation, evaluation and policy training."""
from typing import Any, TypedDict, List

class SimulationResult(TypedDict, total=False):
    run_dir: str
    result_dir: str
    external_result_dir: str
    log_path: str
    reports: List[Any]
    command: List[str]
    scene_conversion: Any

class TrainingResult(TypedDict, total=False):
    run_dir: str
    training_mode: str
    checkpoint: str
    work_dir: str
    log_path: str
    epochs: int
    final_loss: float
    export_dir: str
    training_config: str
    data_preparation: dict
    training_audit: dict

class EvaluationResult(TypedDict, total=False):
    run_dir: str
    is_failure_case: bool
    failure_cases: List[Any]
    collision: bool
    collision_events: List[Any]
    frames_evaluated: int
    trajectory_source: str
