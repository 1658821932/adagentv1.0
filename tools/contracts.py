"""Runtime contracts for backend tools; LLM tools retain their typed runner."""
from functools import wraps
from importlib import import_module
from pydantic import BaseModel, ConfigDict, ValidationError
from tools.llm_runtime import NonEmptyStr, validation_details

class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

class OutputModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

def validated(tool_name, module, function):
    schemas = import_module(module + ".schemas")
    @wraps(function)
    def call(arguments):
        try:
            data = schemas.Input.model_validate(arguments)
        except (ValidationError, ValueError, TypeError) as exc:
            return {"status":"error","tool":tool_name,"error":{
                "type":"input_validation_error","message":"Tool input does not match its schema",
                "validation_errors":validation_details(exc)}}
        result = function(data.model_dump(exclude_unset=True))
        if isinstance(result,dict) and result.get("status") == "success":
            try:
                schemas.Output.model_validate(result.get("result"))
            except (ValidationError, ValueError, TypeError) as exc:
                return {"status":"error","tool":tool_name,"error":{
                    "type":"output_validation_error","message":"Backend output does not match its schema",
                    "validation_errors":validation_details(exc)}}
        return result
    return call
