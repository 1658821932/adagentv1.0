"""Local mock implementation of DGGT scene editing."""

def run(input_data: dict) -> dict:
    return {"status": "success", "tool": "scene_editor", "result": {"edited_scene": "outputs/mock/edited_scene.ply"}}
