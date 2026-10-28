"""Agent tool contract for instance_discovery; backend execution lives in its adapter."""

from mcp_servers.dggt.adapter.instances import discover_instances


def run(input_data: dict) -> dict:
    backend = input_data.get('backend') or 'dggt'
    if backend == 'ufo':
        from mcp_servers.ufo.adapter import discover_instances as ufo_backend
        return {'status': 'success', 'tool': 'instance_discovery', 'result': ufo_backend(input_data)}
    if backend not in {'dggt', 'dggt_yolo'}:
        raise ValueError('backend must be ufo or dggt')
    return {
        "status": "success",
        "tool": "instance_discovery",
        "result": discover_instances(input_data),
    }
