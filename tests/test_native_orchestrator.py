"""Native tool calling: stream assembly and validation before execution."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from agent.llm_client import LLMClient
from agent.orchestrator import Orchestrator
from agent.tool_descriptions import get_function_tools

def fragment(index=0, name=None, arguments=None, id=None):
    return NS(index=index,id=id,function=NS(name=name,arguments=arguments))

def chunk(parts=(), finish=None, content=None, usage=None):
    return NS(usage=usage, choices=[NS(index=0,delta=NS(tool_calls=parts,content=content),finish_reason=finish)])

class Stream:
    def __init__(self,items):
        self.items=items
        self.closed=False
    def __iter__(self):
        yield from self.items
    def close(self):
        self.closed=True

def client_for(stream):
    client=LLMClient.__new__(LLMClient)
    client.model="kimi-k3";client.max_tokens=8192;client.timeout=600
    client.client=Mock()
    client.client.chat.completions.create.return_value=stream
    return client

def reply(name,payload,finish="tool_calls",extra=None):
    calls=[{"id":"call_1","type":"function","function":{"name":name,"arguments":json.dumps(payload)}}]
    if extra:
        calls.append(extra)
    return {"message":{"role":"assistant","content":None,"tool_calls":calls},"finish_reason":finish}

def test_stream_fragments_and_trailing_usage_are_collected():
    usage=NS(model_dump=lambda:{"total_tokens":123})
    stream=Stream([
        chunk([fragment(name="failure_",arguments='{"reason":"analyze","arguments":',id="one")]),
        chunk([fragment(name="analyzer",arguments='{"token":"scene"}}')]),
        chunk(finish="tool_calls"),
        NS(usage=usage,choices=[]),
    ])
    client=client_for(stream)
    response=client.call_functions([],get_function_tools())
    call=response["message"]["tool_calls"][0]
    assert call["function"]["name"]=="failure_analyzer"
    assert json.loads(call["function"]["arguments"])["arguments"]=={"token":"scene"}
    assert client.last_usage=={"total_tokens":123}
    assert stream.closed
    params=client.client.chat.completions.create.call_args.kwargs
    assert params["stream"] and params["tool_choice"]=="required"
    assert params["parallel_tool_calls"] is False
    assert params["max_tokens"]==8192

def test_broken_stream_never_returns_a_call():
    stream=Stream([chunk([fragment(name="finish",arguments='{"reason":"done"}',id="one")])])
    with pytest.raises(RuntimeError,match="completion marker"):
        client_for(stream).call_functions([],get_function_tools())
    assert stream.closed

def test_stream_exception_closes_response():
    class Broken(Stream):
        def __iter__(self):
            yield chunk([fragment(name="finish",arguments="{",id="one")])
            raise OSError("connection lost")
    stream=Broken([])
    with pytest.raises(OSError):
        client_for(stream).call_functions([],get_function_tools())
    assert stream.closed

@pytest.mark.parametrize("response",[
    reply("failure_analyzer",{"reason":"analyze","arguments":{}}),
    reply("finish",{"reason":""}),
    reply("finish",{"reason":"done"},finish="length"),
    reply("unknown",{"reason":"try","arguments":{}}),
    reply("finish",{"reason":"done"},extra={"id":"two","type":"function","function":{"name":"stop","arguments":'{"reason":"blocked"}'}}),
    {"message":{"role":"assistant","content":'{"action":"finish","reason":"done"}',"tool_calls":[]},"finish_reason":"stop"},
])
def test_invalid_or_old_text_protocol_is_rejected(response):
    fake=NS(call_functions=lambda *args:response)
    result=Orchestrator(fake,decision_correction_attempts=0).decide("test",{})
    assert result["action"]=="error"

def test_correction_uses_native_tool_error_messages():
    responses=[
        reply("failure_analyzer",{"reason":"analyze","arguments":{}}),
        reply("failure_analyzer",{"reason":"analyze","arguments":{"token":"scene"}}),
    ]
    client=Mock()
    client.call_functions.side_effect=responses
    decision=Orchestrator(client).decide("Analyze scene",{})
    assert decision["arguments"]=={"token":"scene"}
    conversation=client.call_functions.call_args.args[0]
    assert any(m["role"]=="tool" and m["tool_call_id"]=="call_1" for m in conversation)
    assert "executed" in next(m["content"] for m in conversation if m["role"]=="tool")

def test_retry_metadata_is_separate_from_business_arguments():
    response=reply("failure_analyzer",{"reason":"retry failed job","arguments":{"token":"scene"},"execution":{"retry":True}})
    result=Orchestrator(NS(call_functions=lambda *a:response)).decide("test",{})
    assert result["arguments"]=={"token":"scene","_execution":{"retry":True}}


def test_background_and_reconnection_metadata_stays_outside_policy_arguments():
    for execution in ({'background':True},{'background':False,'job_id':'existing-job'}):
        response=reply('policy_runner',{'reason':'Evaluate the frozen attack independently',
            'arguments':{},'execution':execution})
        result=Orchestrator(NS(call_functions=lambda *a:response)).decide('test',{})
        assert result['action']=='call_tool'
        assert result['arguments']=={'_execution':execution}

def test_definitions_are_valid_at_function_root():
    for tool in get_function_tools():
        root=tool["function"]["parameters"]
        def visit(node):
            if isinstance(node,dict):
                if "$ref" in node:
                    target=root
                    for key in node["$ref"].removeprefix("#/").split("/"):
                        target=target[key]
                for value in node.values():visit(value)
            elif isinstance(node,list):
                for value in node:visit(value)
        visit(root)
