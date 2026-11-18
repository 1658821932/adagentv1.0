"""Native function-call planning; business execution remains in ToolExecutor."""
import json
from pathlib import Path
from importlib import import_module
from agent.llm_client import LLMClient
from agent.schemas import validate_decision
from agent.tool_descriptions import get_tool_descriptions, get_function_tools
from mcp_servers.runtime import trace_context

class Orchestrator:
    def __init__(self, llm_client=None, decision_correction_attempts=1):
        if type(decision_correction_attempts) is not int or decision_correction_attempts < 0:
            raise ValueError("decision_correction_attempts must be a non-negative integer")
        self.llm_client = llm_client or LLMClient()
        self.decision_correction_attempts = decision_correction_attempts
        self.tools = get_tool_descriptions(compact=True)
        self.tool_names = {tool["name"] for tool in self.tools}
        self.functions = get_function_tools()
        root = Path(__file__).resolve().parents[1] / "tools"
        self.input_models = {}
        for name in self.tool_names:
            folder = next(root.glob("tool[0-9][0-9]_" + name))
            self.input_models[name] = import_module("tools." + folder.name + ".schemas").Input

    def _build_prompt(self, task, state):
        prompt = Path(__file__).with_name("prompt.md").read_text().strip()
        return [
            {"role":"system","content":prompt},
            {"role":"user","content":json.dumps(
                {"task":task,"current_state":{k:v for k,v in state.items() if k!="task"}},ensure_ascii=False,separators=(",",":"))},
        ]

    def _parse_and_validate(self, response):
        message = response["message"]
        calls = message.get("tool_calls") or []
        if response.get("finish_reason") not in ("tool_calls", "stop"):
            return {}, ["Incomplete or truncated function-call response"]
        if len(calls) != 1:
            return {}, ["Exactly one function call is required per decision"]
        function = calls[0].get("function", {})
        name = function.get("name")
        try:
            payload = json.loads(function["arguments"])
            if not isinstance(payload, dict):
                raise ValueError("Function arguments must be an object")
            if name in ("finish", "stop"):
                if set(payload) != {"reason"}:
                    raise ValueError("finish/stop require only reason")
                decision = {"action":name,"reason":payload["reason"]}
            else:
                if name not in self.tool_names:
                    return {}, [f"unknown tool_name: {name}"]
                if set(payload) - {"arguments","reason","execution"}:
                    raise ValueError("Unexpected function argument fields")
                arguments = payload.get("arguments")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be an object")
                execution = payload.get("execution")
                if execution is not None:
                    if not isinstance(execution,dict) or set(execution)-{"retry","background","job_id"}:
                        raise ValueError('execution accepts retry, background and job_id')
                    for field in ('retry','background'):
                        if field in execution and type(execution[field]) is not bool:raise ValueError(field+' must be boolean')
                    if 'job_id' in execution and (not isinstance(execution['job_id'],str) or not execution['job_id'].strip()):
                        raise ValueError('job_id must be a nonempty string')
                # Validate before a job is launched. Keep original references/arguments.
                self.input_models[name].model_validate(arguments)
                arguments = dict(arguments)
                if execution is not None:
                    arguments["_execution"] = execution
                decision = dict(action="call_tool",tool_name=name,arguments=arguments,reason=payload.get("reason"))
            valid, errors = validate_decision(decision,self.tool_names)
            return decision, errors
        except (KeyError, ValueError, TypeError) as exc:
            return {}, [str(exc)]

    def decide(self, task, state):
        conversation = self._build_prompt(task,state)
        for attempt in range(self.decision_correction_attempts+1):
            with trace_context(decision_attempt=attempt + 1):
                response = self.llm_client.call_functions(conversation,self.functions)
            decision, errors = self._parse_and_validate(response)
            if not errors:
                return decision
            if attempt == self.decision_correction_attempts:
                break
            message = response["message"]
            calls = message.get("tool_calls") or []
            if calls:
                conversation.append(message)
                for call in calls:
                    conversation.append({"role":"tool","tool_call_id":call["id"],
                        "content":json.dumps({"validation_errors":errors,"executed":False},ensure_ascii=False)})
            else:
                continuation = dict(message)
                continuation.pop('tool_calls', None)
                continuation['content'] = message.get('content') or 'No function call.'
                conversation.append(continuation)
            conversation.append({"role":"user","content":
                "No tool was executed. Correct the validation errors and call exactly one declared function. "
                "Use finish/stop with a nonempty reason when appropriate. Errors: "+json.dumps(errors,ensure_ascii=False)})
        return dict(action="error",error_type="decision_validation_failed",
                    validation_errors=errors,reason="Function call failed validation after configured correction attempts")
