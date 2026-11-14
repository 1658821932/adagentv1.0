"""Small cross-tool contract; annotations/rewards remain referenced on disk."""
from common.models import InputModel, NonEmptyStr

class RLTrainingData(InputModel):
    rollout_dir: NonEmptyStr
    token_file: NonEmptyStr
    vocab_path: NonEmptyStr
    manifest_path: NonEmptyStr
