"""Mode-specific inputs for imitation and reinforcement fine-tuning."""
from typing import Annotated, Literal, Any
from pydantic import Field, model_validator, AliasChoices
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]

class TrainingData(InputModel):
    ann_file: NonEmptyStr | None = Field(default=None, description='ILFT converted annotation file; expert trajectory CSV alone is not this file.')
    token_file: NonEmptyStr = Field(validation_alias=AliasChoices('token_file', 'nav_filter_path'), description='Token-selection file matching the training scenes.')
    rollout_dir: NonEmptyStr | None = Field(default=None, description='RLFT openscene_format directory with meta_datas/, sensor_blobs/, pdms_pkl/, plan_idx.csv and official all_scenes_pdm_averages_*.csv.')
    vocab_path: NonEmptyStr | None = Field(default=None, description='RLFT (8192,40,3) trajectory vocabulary used to generate the supplied candidate rewards.')

class Input(InputModel):
    training_mode: Literal['ilft', 'rlft'] = Field(default='ilft', description='Post-training objective; required data depends on this mode.')
    training_data: TrainingData = Field(description='Converted ILFT annotations or complete RLFT rollout data, as specified by TrainingData.')
    base_checkpoint: NonEmptyStr = Field(validation_alias=AliasChoices('base_checkpoint', 'checkpoint_path'), description='Existing policy checkpoint to update.')
    config_path: NonEmptyStr | None = Field(default=None, description='Optional mode-appropriate native model/training configuration.')
    epochs: PositiveInt = Field(default=1, validation_alias=AliasChoices('epochs', 'total_epochs'), description='Number of training epochs.')
    worldengine_root: NonEmptyStr | None = Field(default=None, description='Optional WorldEngine source root.')
    python: NonEmptyStr | None = Field(default=None, description='Optional AlgEngine training interpreter.')
    work_dir: NonEmptyStr | None = Field(default=None, description='Optional export parent; execution uses an isolated runtime task.')
    timeout: PositiveInt = Field(default=86400, description='Training process timeout in seconds.')
    seed: Annotated[int, Field(ge=0)] = Field(default=0, description='Random seed for this training run.')

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
    checkpoint: NonEmptyStr = Field(description='Checkpoint produced by training.')
    training_mode: Literal['ilft', 'rlft'] = Field(description='Actual executed mode.')
    log_path: NonEmptyStr = Field(description='Training log path.')
    training_audit: dict[str, Any] | None = Field(default=None, description='Recorded losses, finite-value checks and parameter-update audit.')
    data_preparation: dict[str, Any] | None = Field(default=None, description='Mode-specific training input preparation summary.')
