"""Agent tool contract for reconstruction; backend execution lives in its adapter."""

from mcp_servers.dggt.adapter.client import reconstruct_video


def run(input_data: dict) -> dict:
    backend = input_data.get('backend') or 'dggt'
    if backend == 'ufo':
        from mcp_servers.ufo.adapter import reconstruct_video as ufo_backend
        return {'status': 'success', 'tool': 'reconstruction', 'result': ufo_backend(input_data)}
    if backend not in {'dggt', 'dggt_yolo'}:
        raise ValueError('backend must be ufo or dggt')
    return {
        "status": "success",
        "tool": "reconstruction",
        "result": reconstruct_video(input_data),
    }
