"""Public tools reject malformed calls before loading expensive backends."""
from importlib import import_module
from unittest.mock import patch
import pytest
from tools.registry import call_tool, _BACKEND_MODULES
from agent.tool_descriptions import get_tool_descriptions

CASES = {
 "failure_analyzer": {"token":"scene-token"},
 "video_retriever": {"requirement":"daytime highway"},
 "instance_discovery": {"image_dir":"/frames","output_path":"/instances.json"},
 "reconstruction": {"video_path":"/video.mp4"},
 "edit_config_generator": {"edit_plan":"move vehicle", "output_path":"/edit.json"},
 "scene_editor": {"video_path":"/video.mp4","edit_config_path":"/edit.json"},
 "scene_validator": {"rendered_video":"/render.mp4","scene_requirement":"lead car"},
 "policy_runner": {"config_path":"/cfg.py","checkpoint_path":"/model.pth","model_name":"test","data_type":"test"},
 "failure_case_filter": {"scenario_pkl":"/scenario.pkl"},
 "policy_trainer": {"training_data":{"ann_file":"/ann.pkl","token_file":"/tokens.yaml"},"base_checkpoint":"/base.pth"},
}

@pytest.mark.parametrize("name", list(CASES))
def test_mock_and_real_share_executable_contract(name):
    result = call_tool(name, CASES[name], mode="mock")
    assert result["status"] == "success", result
    module = import_module(_BACKEND_MODULES[name]+".schemas")
    module.Output.model_validate(result["result"])
    description = next(d for d in get_tool_descriptions() if d["name"] == name)
    assert set(description["inputs"]) == set(module.Input.model_fields)
    assert set(module.Output.model_fields) <= set(description["outputs"])

@pytest.mark.parametrize("name", list(CASES))
def test_invalid_inputs_stop_before_any_real_backend(name):
    result = call_tool(name, {**CASES[name],"typo_parameter":True}, mode="real")
    assert result["status"] == "error"
    assert result["error"]["type"] == "input_validation_error"

def test_invalid_backend_success_is_not_forwarded():
    from tools.contracts import validated
    called=[]
    def malformed(data):
        called.append(data)
        return {"status":"success","tool":"reconstruction","result":{"scene_ply":"/only.ply"}}
    wrapped=validated("reconstruction",_BACKEND_MODULES["reconstruction"],malformed)
    result=wrapped(CASES["reconstruction"])
    assert called and result["error"]["type"]=="output_validation_error"

def test_invalid_horizon_and_negative_scale():
    result=call_tool("policy_runner",{**CASES["policy_runner"],"num_history":4,"num_future":8,"max_frame_per_scene":40},mode="mock")
    assert result["error"]["type"]=="input_validation_error"
    result=call_tool("scene_editor",{**CASES["scene_editor"],"meters_to_dggt":-1.0},mode="mock")
    assert result["error"]["type"]=="input_validation_error"

def test_wfa_description_uses_scene_tokens_and_registry_uses_mock_modules():
    description=next(d for d in get_tool_descriptions() if d["name"]=="failure_analyzer")
    assert "token" in description["inputs"] and "input_path" not in description["inputs"]
    for module in _BACKEND_MODULES.values():
        assert import_module(module+".mock").run

def test_boolean_epoch_is_not_an_integer_epoch():
    result=call_tool("policy_trainer",{**CASES["policy_trainer"],"epochs":True},mode="mock")
    assert result["error"]["type"]=="input_validation_error"
