"""Schema-compatible mock DGGT editing."""
def run(input_data):
    return {"status":"success","tool":"scene_editor","result":{
        "scene_ply":"outputs/mock/edited_scene.ply","rendered_video":"outputs/mock/edited.mp4",
        "output_dir":"outputs/mock/edit","scene_config":input_data.get("edit_config_path"),"mode":"mock"}}
