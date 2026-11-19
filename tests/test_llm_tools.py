"""Behavioral regression tests for typed text-only LLM tools (no network)."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tools.registry import call_tool, REAL_TOOL_REGISTRY
from scripts.smoke_llm_tools import ANALYSIS

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
        "status":"success","tool":"failure_analyzer","result":ANALYSIS
    }},mode="mock")
    saved=json.loads((Path(requirement["meta"]["record_dir"])/"input.json").read_text())["failure_analysis"]
    assert set(saved)=={"violation_step","failure","attack","environment","verification","source"}
    assert saved["source"]["token"]==ANALYSIS["token"]
    assert saved["attack"]==ANALYSIS["attack"]
    assert set(requirement["result"])=={"target_scene","attack","candidates"}
    from tools.tool03_video_retriever.schemas import Input
    for candidate in requirement["result"]["candidates"]:
        Input.model_validate({"requirement":candidate["retrieval_query"]})
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
    (tmp_path/"config.json").write_text('{"prompt_version":"v1","validation_retries":99}')
    (tmp_path/"prompt.md").write_text("prompt")
    client=ScriptedClient()
    result=replace(TOOL,directory=tmp_path).execute({"failure_analysis":ANALYSIS,"_llm_client":client})
    assert result["error"]["type"]=="configuration_error"
    assert not client.messages


def test_tool02_uses_shared_client_with_its_component_profile():
    from tools.tool02_scene_instance_generator.real import run
    client=ScriptedClient(response_for(CASES[0][0],CASES[0][2]))
    with patch("agent.llm_client.LLMClient",return_value=client) as factory:
        result=run({"failure_analysis":ANALYSIS})
    assert result["status"]=="success"
    factory.assert_called_once_with(component='scene_instance_generator')


def test_llm_client_inherits_root_limits(monkeypatch):
    from agent.llm_client import LLMClient
    monkeypatch.setenv("OPENAI_API_KEY","test-only")
    profile={"model":"root-model","base_url":"https://example.invalid/v1",
             "timeout":77,"max_tokens":3456}
    sdk=Mock(); timeout_calls=[]
    def timeout(value,**kwargs):
        timeout_calls.append((value,kwargs)); return "timeout-object"
    with patch.object(LLMClient,"_load_llm_config",return_value=profile), patch.dict(
        "sys.modules",{"openai":SimpleNamespace(OpenAI=sdk,Timeout=timeout)}):
        client=LLMClient()
    assert (client.model,client.timeout,client.max_tokens)==("root-model",77,3456)
    assert sdk.call_args.kwargs["timeout"]=="timeout-object"
    assert timeout_calls==[(77.0,{"connect":15.0,"read":60.0})]
    assert sdk.call_args.kwargs["max_retries"]==0


def test_tool_configuration_rejects_duplicate_model_settings():
    from tools.llm_runtime import LLMToolConfig
    for field,value in [("model_role","planner"),("timeout",20),("max_tokens",100)]:
        with pytest.raises(ValueError):
            LLMToolConfig.model_validate({"prompt_version":"v1",field:value})


def test_empty_prompt_is_a_configuration_error_without_model_call(tmp_path):
    from dataclasses import replace
    from tools.tool02_scene_instance_generator.real import TOOL
    (tmp_path/"config.json").write_text('{"prompt_version":"v1"}')
    (tmp_path/"prompt.md").write_text(" ")
    client=ScriptedClient()
    result=replace(TOOL,directory=tmp_path).execute({
        "failure_analysis":ANALYSIS,"_llm_client":client})
    assert result["error"]["type"]=="configuration_error"
    assert not client.messages
