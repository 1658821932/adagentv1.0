"""Deterministic edit plan with the same grounding rules as real."""
from .real import TOOL
from .schemas import known_instance_ids


def _result(data):
    known=known_instance_ids(data)
    target=next(iter(known.values()),None)
    request=data.edit_request if isinstance(data.edit_request,str) else str(data.edit_request)
    return {
        "edit_plan": "Prepare a trajectory edit for: " + request,
        "target_instances": [] if target is None else [target],
        "operations": [{"operation":"modify_trajectory","target":target,"parameters":{"target_role":"relevant traffic participant"}}],
        "constraints":["Use relative trajectory offsets and validate the rendered result."],
    }


def run(input_data: dict) -> dict:
    return TOOL.execute(input_data, mock_result=_result)
