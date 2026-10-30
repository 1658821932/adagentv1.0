"""Artifact references returned by the AdAgent SmartEdit adapter."""
from typing import TypedDict, List

class EditConfigResult(TypedDict, total=False):
    saved_path: str
    tool_calls: List[str]
    run_dir: str
    output_kind: str
