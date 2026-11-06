"""Framework-neutral Pydantic building blocks; no tools or model clients."""
from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]

class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

class OutputModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
