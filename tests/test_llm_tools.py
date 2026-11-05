"""Behavioral regression tests for typed text-only LLM tools (no network)."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tools.registry import call_tool, REAL_TOOL_REGISTRY
from tools.scripts.smoke_llm_tools import ANALYSIS

CASES=[
    ("tool02_scene_instance_generator","scene_instance_generator",{"failure_analysis":ANALYSIS}),
]


class ScriptedClient:
    model="offline-test-model"
    def __init__(self,*responses):
        self.responses=list(responses)
        self.messages=[]
    def chat(self,messages):
        self.messages.append(deepcopy(messages))
        value=self.responses.pop(0)
        if isinstance(value,Exception):
            raise value
        return value


def response_for(module_name,arguments):
    module=importlib.import_module("tools."+module_name+".mock")
    result=module.run(deepcopy(arguments))
    assert result["status"]=="success",result
    content={key:value for key,value in result["result"].items()
             if key not in {"source_analysis","source_requirement"}}
    return json.dumps(content)


@pytest.mark.parametrize("folder,name,arguments",CASES)
def test_registry_mock_and_real_share_contract(folder,name,arguments):
    schemas=importlib.import_module("tools."+folder+".schemas")
    client=ScriptedClient(response_for(folder,arguments))
    original=deepcopy(arguments)
    real=call_tool(name,{**arguments,"_llm_client":client},mode="real")
    mock=call_tool(name,arguments,mode="mock")
    assert real["status"]==mock["status"]=="success"
    schemas.Output.model_validate(real["result"])
    assert real["result"]==mock["result"]
    assert arguments==original
    assert real["meta"]["attempts"]==1
    assert mock["meta"]["attempts"]==0
    assert real["meta"]["mode"]=="real"
    assert mock["meta"]["mode"]=="mock"
    assert "_llm_client" not in client.messages[0][1]["content"]
    prompt=(Path("tools")/folder/"prompt.md").read_text().strip()
    assert prompt in client.messages[0][0]["content"]
    json.dumps(real)


@pytest.mark.parametrize("folder,name,arguments",CASES)
def test_bad_input_is_rejected_before_model_call(folder,name,arguments):
    client=ScriptedClient()
    result=call_tool(name,{"_llm_client":client},mode="real")
    assert result["error"]["type"]=="input_validation_error"
    assert result["meta"]["attempts"]==0
    assert not client.messages


@pytest.mark.parametrize("folder,name,arguments",CASES)
def test_invalid_json_is_corrected_once(folder,name,arguments):
    valid=response_for(folder,arguments)
    client=ScriptedClient("not JSON",valid)
    result=call_tool(name,{**arguments,"_llm_client":client},mode="real")
    assert result["status"]=="success",result
    assert result["meta"]["attempts"]==2
    assert "Errors:" in client.messages[1][-1]["content"]


@pytest.mark.parametrize("folder,name,arguments",CASES)
def test_invalid_output_cannot_escape_retry_limit(folder,name,arguments):
    client=ScriptedClient("[]","{}")
    result=call_tool(name,{**arguments,"_llm_client":client},mode="real")
    assert result["error"]["type"]=="output_validation_error"
    assert len(client.messages)==2
    assert result["error"]["validation_errors"]


def test_type_errors_are_not_silently_coerced():
    folder,name,arguments=CASES[0]
    content=json.loads(response_for(folder,arguments))
    content["candidates"]="straight road"
    invalid=json.dumps(content)
    result=call_tool(name,{**arguments,"_llm_client":ScriptedClient(invalid,invalid)},mode="real")
    assert result["error"]["type"]=="output_validation_error"


def test_unknown_input_fields_are_rejected():
    folder,name,arguments=CASES[0]
    result=call_tool(name,{**arguments,"misspelled_parameter":123},mode="mock")
    assert result["error"]["type"]=="input_validation_error"


def test_transport_failure_is_not_a_json_retry_and_credentials_are_redacted():
    folder,name,arguments=CASES[0]
    client=ScriptedClient(RuntimeError("request failed: Bearer sk-testcredential"))
    result=call_tool(name,{**arguments,"_llm_client":client},mode="real")
    assert result["error"]["type"]=="llm_request_error"
    assert len(client.messages)==1
    assert "sk-testcredential" not in json.dumps(result)


def test_fenced_json_is_supported_without_accepting_surrounding_prose():
    folder,name,arguments=CASES[0]
    fence=chr(96)*3
    valid=response_for(folder,arguments)
    client=ScriptedClient(fence+"json\n"+valid+"\n"+fence)
    assert call_tool(name,{**arguments,"_llm_client":client},mode="real")["status"]=="success"
    client=ScriptedClient("Here is JSON: "+valid,"Here is JSON: "+valid)
    assert call_tool(name,{**arguments,"_llm_client":client},mode="real")["status"]=="error"


def test_source_context_is_attached_by_code_and_upstream_errors_are_not_evidence():
    requirement=call_tool("scene_instance_generator",{"failure_analysis":{
        "status":"success","tool":"failure_analyzer","result":{"failure_analysis":ANALYSIS}
    }},mode="mock")
    assert json.loads((Path(requirement["meta"]["record_dir"])/"input.json").read_text())["failure_analysis"]==ANALYSIS
    assert set(requirement["result"])=={"scene_requirement","target_effect","candidates"}
    from tools.tool04_video_retriever.schemas import Input
    for candidate in requirement["result"]["candidates"]:
        Input.model_validate({"requirement":candidate["base_video_requirement"]})
    bad={"status":"error","tool":"failure_analyzer","error":{"message":"failed"}}
    result=call_tool("scene_instance_generator",{"failure_analysis":bad},mode="mock")
    assert result["error"]["type"]=="input_validation_error"












@pytest.mark.parametrize("folder,name,arguments",CASES)
def test_description_fields_come_from_schemas_and_do_not_advertise_test_client(folder,name,arguments):
    description=importlib.import_module("tools."+folder+".description").DESCRIPTION
    assert set(description["inputs"])==set(description["input_schema"]["properties"])
    assert set(description["outputs"])==set(description["output_schema"]["properties"])
    assert "_llm_client" not in json.dumps(description)


def test_invalid_tool_configuration_fails_without_network(tmp_path):
    from dataclasses import replace
    from tools.tool02_scene_instance_generator.real import TOOL
    (tmp_path/"config.json").write_text('{"model_role":"planner","prompt_version":"v1","validation_retries":99}')
    (tmp_path/"prompt.md").write_text("prompt")
    client=ScriptedClient()
    result=replace(TOOL,directory=tmp_path).execute({"failure_analysis":ANALYSIS,"_llm_client":client})
    assert result["error"]["type"]=="configuration_error"
    assert not client.messages


def test_llm_role_and_per_tool_limits_reach_actual_client():
    from agent.llm_client import LLMClient
    from tools.tool02_scene_instance_generator.real import run
    valid=json.loads(response_for(CASES[0][0],CASES[0][2]))
    client=ScriptedClient(json.dumps(valid))
    with patch.object(LLMClient,"_load_llm_config",return_value={"roles":{"planner":{"model":"role-model"}}}):
        with patch("agent.llm_client.LLMClient",return_value=client) as factory:
            factory._load_llm_config.return_value={"roles":{"planner":{"model":"role-model"}}}
            result=run({"failure_analysis":ANALYSIS})
    assert result["status"]=="success"
    assert factory.call_args.kwargs["overrides"]=={"model":"role-model","timeout":600,"max_tokens":8192}


def test_llm_client_overrides_do_not_replace_global_config(monkeypatch):
    from agent.llm_client import LLMClient
    monkeypatch.setenv("OPENAI_API_KEY","test-only")
    monkeypatch.setenv("OPENAI_BASE_URL","https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL","global-model")
    sdk=Mock()
    with patch.dict("sys.modules", {"openai":SimpleNamespace(OpenAI=sdk)}):
        client=LLMClient(overrides={"model":"editor-model","timeout":25,"max_tokens":2000})
    assert client.model=="editor-model"
    assert client.timeout==25
    assert client.max_tokens==2000
    assert sdk.call_args.kwargs["timeout"]==25
    assert LLMClient._load_llm_config()["model"]!="editor-model"




def test_omitted_tool_limits_inherit_role_settings(tmp_path):
    from dataclasses import replace
    from tools.tool02_scene_instance_generator.real import TOOL
    config={"model_role":"planner","prompt_version":"v1"}
    (tmp_path/"config.json").write_text(json.dumps(config))
    (tmp_path/"prompt.md").write_text("Use source evidence and return JSON.")
    client=ScriptedClient(response_for(CASES[0][0],CASES[0][2]))
    with patch("agent.llm_client.LLMClient",return_value=client) as factory:
        factory._load_llm_config.return_value={
            "roles":{"planner":{"model":"role-model","timeout":35,"max_tokens":2500}}}
        result=replace(TOOL,directory=tmp_path).execute({"failure_analysis":ANALYSIS})
    assert result["status"]=="success"
    assert factory.call_args.kwargs["overrides"]=={
        "model":"role-model","timeout":35,"max_tokens":2500}


def test_empty_prompt_is_a_configuration_error_without_model_call(tmp_path):
    from dataclasses import replace
    from tools.tool02_scene_instance_generator.real import TOOL
    (tmp_path/"config.json").write_text('{"model_role":"planner","prompt_version":"v1"}')
    (tmp_path/"prompt.md").write_text(" ")
    client=ScriptedClient()
    result=replace(TOOL,directory=tmp_path).execute({
        "failure_analysis":ANALYSIS,"_llm_client":client})
    assert result["error"]["type"]=="configuration_error"
    assert not client.messages
