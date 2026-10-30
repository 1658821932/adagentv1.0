"""Public failure-analysis result contract."""
from dataclasses import dataclass

@dataclass
class WfaSceneResult:
    token: str
    run_dir: str
    status: str
    failure_reason: str
    reason_category: str
    observations: list
    candidate_assessment: list
    uncertainty: str
    official_metrics: dict
    report_path: str
    result_path: str
    raw: dict
