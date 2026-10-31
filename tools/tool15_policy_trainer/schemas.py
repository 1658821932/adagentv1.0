"""Mode-specific inputs for imitation and reinforcement fine-tuning."""
from typing import Annotated, Literal, Any
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]

class TrainingData(InputModel):
    ann_file: NonEmptyStr | None = None
    token_file: NonEmptyStr = Field(validation_alias=AliasChoices("token_file", "nav_filter_path"))
    rollout_dir: NonEmptyStr | None = None
    vocab_path: NonEmptyStr | None = None

class Input(InputModel):
    training_mode: Literal["ilft", "rlft"] = "ilft"
    training_data: TrainingData
    base_checkpoint: NonEmptyStr = Field(validation_alias=AliasChoices("base_checkpoint", "checkpoint_path"))
    config_path: NonEmptyStr | None = None
    epochs: PositiveInt = Field(default=1, validation_alias=AliasChoices("epochs", "total_epochs"))
    worldengine_root: NonEmptyStr | None = None
    python: NonEmptyStr | None = None
    work_dir: NonEmptyStr | None = None
    timeout: PositiveInt = 86400
    seed: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="before")
    @classmethod
    def legacy_data(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        training = data.get("training_data")
        if isinstance(training, str):
            data["training_data"] = {"ann_file": training, "token_file": data.pop("token_file", None)}
        elif not training and "ann_file" in data:
            data["training_data"] = {"ann_file": data.pop("ann_file"), "token_file": data.pop("token_file", None)}
        return data

    @model_validator(mode="after")
    def validate_mode(self):
        t = self.training_data
        if self.training_mode == "ilft":
            if not t.ann_file:
                raise ValueError("ilft requires training_data.ann_file")
            if t.rollout_dir or t.vocab_path:
                raise ValueError("rollout_dir/vocab_path belong to rlft, not ilft")
        else:
            if not t.rollout_dir or not t.vocab_path:
                raise ValueError("rlft requires training_data.rollout_dir and vocab_path")
            if t.ann_file:
                raise ValueError("rlft uses rollout data; ann_file is for ilft")
        return self

class Output(OutputModel):
    checkpoint: NonEmptyStr
    training_mode: Literal["ilft", "rlft"]
    log_path: NonEmptyStr
    training_audit: dict[str, Any] | None = None
    data_preparation: dict[str, Any] | None = None
