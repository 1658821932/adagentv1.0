"""Agent tool contract for edit_config_generator; backend execution lives in its adapter."""

from mcp_servers.smart_edit.adapter.service import generate_edit_config


def run(input_data: dict) -> dict:
    backend = input_data.get('backend') or 'dggt'
    if backend == 'ufo':
        from mcp_servers.ufo.adapter import generate_edit_config as ufo_backend
        return {'status': 'success', 'tool': 'edit_config_generator', 'result': ufo_backend(input_data)}
    if backend not in {'dggt', 'dggt_yolo'}:
        raise ValueError('backend must be ufo or dggt')
    return {
        "status": "success",
        "tool": "edit_config_generator",
        "result": generate_edit_config(input_data),
    }
