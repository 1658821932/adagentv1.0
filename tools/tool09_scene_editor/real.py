"""Agent tool contract for scene_editor; backend execution lives in its adapter."""

from mcp_servers.dggt.adapter.editing import edit_scene


def run(input_data: dict) -> dict:
    backend = input_data.get('backend') or 'dggt'
    if backend == 'ufo':
        from mcp_servers.ufo.adapter import edit_scene as ufo_backend
        return {'status': 'success', 'tool': 'scene_editor', 'result': ufo_backend(input_data)}
    if backend not in {'dggt', 'dggt_yolo'}:
        raise ValueError('backend must be ufo or dggt')
    return {
        "status": "success",
        "tool": "scene_editor",
        "result": edit_scene(input_data),
    }
