"""Executable Agent-tool input and output contracts."""
from typing import Any, Annotated, Literal
from pydantic import ConfigDict, Field, model_validator
from common.rl_training import RLTrainingData
from mcp_servers.worldengine.adapter.evaluation_horizon import DEFAULT_NATIVE_HISTORY, DEFAULT_NATIVE_FRAMES
from tools.contracts import InputModel, OutputModel, NonEmptyStr
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]

class Input(InputModel):
    model_config = ConfigDict(json_schema_extra={'required': []})

    execution_mode: Literal['native', 'pdm_closed_loop'] = Field(default='native', description='Native full simulation or the independent PDM closed-loop challenge.')
    collision_target_track: NonEmptyStr | None = Field(default=None, description='Native mode: exact attack track from tool05. Return actual post-takeover body-contact evidence independently of PDM score, fault attribution and road compliance. Use tool05 simulation_input with original ego history for attack testing.')
    config_path: NonEmptyStr | None = Field(default=None, description='Optional existing AlgEngine configuration override. Omit to use the policy_config configured in the WorldEngine adapter.')
    checkpoint_path: NonEmptyStr | None = Field(default=None, description='Optional existing policy checkpoint override. Omit to use the adapter policy_checkpoint, shared with post-training.')
    model_name: NonEmptyStr | None = Field(default=None, description='Optional experiment label override; the adapter supplies its fixed default when omitted.')
    data_type: NonEmptyStr | None = Field(default=None, description='Optional native scenario data type override; the adapter supplies its fixed default when omitted.')
    challenge_run_baseline_diagnostic: bool = Field(default=False, description='Optionally simulate original scene for diagnosis only; no original-scene safety gate.')
    challenge_baseline_scene_path: NonEmptyStr | None = Field(default=None, description='Read-only original scene for timestamps, map and source integrity. It is not simulated by default.')
    challenge_attack_scene_path: NonEmptyStr | None = Field(default=None, description='Frozen attack scene used for the PDM challenge.')
    challenge_target_track: NonEmptyStr | None = Field(default=None, description='Attack target track present in the frozen scene.')
    challenge_target_is_new: bool = Field(default=False, description='Whether the attack introduced a new target track.')
    challenge_ignore_traffic_lights: bool = Field(default=False, description='Explicit signal-agnostic PDM experiment; original light annotations remain unchanged, signal compliance is not evaluated.')
    challenge_history_s: PositiveFloat = Field(default=3.0, description='Recorded history before PDM takeover.')
    challenge_probe_result_path: NonEmptyStr | None = Field(default=None, description='Optional accepted attack pdm_probe_result_path; reuse only after exact scene/history/target/runtime/map verification.')
    react_type: Literal['R', 'NR'] = Field(default='NR', description='Reactive R or nonreactive NR background traffic.')
    asset_name: NonEmptyStr | None = Field(default=None, description='Optional native renderer asset group.')
    added_name: NonEmptyStr = Field(default='mcp', description='Run-name suffix.')
    num_scenarios: PositiveInt | None = Field(default=None, description='Optional maximum number of scenarios to execute.')
    scenario_data_folder_path: NonEmptyStr | None = Field(default=None, description='Prepared native scenario directory; may come from editor simulation_input.')
    asset_folder_path: NonEmptyStr | None = Field(default=None, description='Prepared native Gaussian/renderer asset directory.')
    scenario_pkl_file_name: NonEmptyStr | None = Field(default=None, description='Scenario pickle name within scenario_data_folder_path.')
    num_history: PositiveInt = Field(default=DEFAULT_NATIVE_HISTORY, description='Input history frame count; normal native runs use 4 for takeover at 1.5s at 2Hz.')
    num_future: PositiveInt | None = Field(default=None, description='Future frame count; derived from the total budget when omitted.')
    max_frame_per_scene: Annotated[int, Field(ge=5)] = Field(default=DEFAULT_NATIVE_FRAMES, description='Normal native score-only and RL collection both use 16 frames: score 1.5-7.5s at 2Hz with num_history=4. Dense collection needs at least 24 real source frames for its 8-frame lookahead. Change only for an explicitly requested other window; num_history + num_future must equal this budget.')
    sampling_poses: PositiveInt | None = Field(default=None, description='Derived automatically from scene frames minus history for full-rollout PDM. Omit this field; a legacy explicit value must equal the derived interval count. Does not change VAD prediction horizon.')
    with_dense_reward_manager: bool = Field(default=True, description='Collect native 8192-candidate PDM rewards for subsequent RLFT. On by default so every native run yields a training package; the last 8 source frames are reserved as the reward lookahead and are not driven. Turn off only for a pure score-only diagnosis.')
    ego_policy: NonEmptyStr | None = Field(default=None, description='Optional native ego policy override.')
    ego_controller: NonEmptyStr | None = Field(default=None, description='Optional native ego controller override.')
    ego_client: NonEmptyStr | None = Field(default=None, description='Optional native planner-client override.')
    ego_navigation: NonEmptyStr | None = Field(default=None, description='Optional native ego navigation override.')
    agent_policy: NonEmptyStr | None = Field(default=None, description='Optional native background-agent policy override.')
    agent_navigation: NonEmptyStr | None = Field(default=None, description='Optional native background-agent navigation override.')
    agent_client: NonEmptyStr | None = Field(default=None, description='Optional native background-agent client override.')
    timeout: PositiveInt = Field(default=86400, description='Simulation process timeout in seconds.')

    @model_validator(mode="after")
    def horizon(self):
        if self.num_history >= self.max_frame_per_scene:
            raise ValueError("max_frame_per_scene must exceed num_history")
        if self.num_future is not None and self.num_history+self.num_future != self.max_frame_per_scene:
            raise ValueError("num_history + num_future must equal max_frame_per_scene")
        if self.execution_mode == 'pdm_closed_loop':
            if self.collision_target_track is not None:
                raise ValueError('collision_target_track is a native closed-loop check')
            if not self.challenge_baseline_scene_path or not self.challenge_attack_scene_path or not self.challenge_target_track:
                raise ValueError('pdm_closed_loop requires baseline/attack scene paths and a target track')
        # Native policy defaults are resolved and their artifacts checked by
        # adapter.config.resolve_policy_inputs before launching the simulator.
        # Validation here must not require model paths from the LLM/user.
        if self.execution_mode=='native' and self.sampling_poses is not None and self.sampling_poses!=self.max_frame_per_scene-self.num_history:
            raise ValueError('Full-rollout PDM derives sampling_poses from scene length; omit the stale override')
        return self

class Output(OutputModel):
    startup_timing_report: NonEmptyStr | None = Field(default=None, description='Per-process cold-start timing report: imports, configuration, model/checkpoint/CUDA setup, Gaussian asset loading, history wait and first plan. Nested and parallel durations overlap.')
    evaluation_scope: dict[str, Any] | None = Field(default=None, description='Verified full-rollout PDM window and actual state/reference/velocity/obstacle counts per scene. Extended-horizon score, not the standard fixed 4-second NAVSIM benchmark.')
    collision_report: dict[str, Any] | None = Field(default=None, description='Actual closed-loop contact with the requested attack track. target_collision_detected=true after policy takeover is the attack criterion; false requires complete coverage; null is unknown. Interpolated possible contact and pre-takeover contact do not establish policy failure.')
    pdm_subscores: dict[str, Any] | None = Field(default=None, description='Official PDM sub-scores read verbatim from all_scenes_pdm_averages_*.csv without altering PDM computation.')
    policy_behavior_summary: dict[str, Any] | None = Field(default=None, description='Closed-loop ego kinematics over the evaluation horizon: absolute progress, speed profile, stop detection, yaw change/turn completion, and comparison against recorded and expert trajectories.')
    result_dir: NonEmptyStr = Field(description='Closed-loop output directory containing official evaluation artifacts.')
    reports: list[Any] = Field(description='Native execution reports.')
    policy_result: NonEmptyStr = Field(description='Policy-run result identifier or path.')
    rollout_dir: NonEmptyStr | None = Field(default=None, description='WorldEngine openscene_format directory for rollout-based training.')
    reward_dir: NonEmptyStr | None = Field(default=None, description='Dense candidate-reward directory including actual plan_idx.csv when collected.')
    candidate_rewards_present: bool = Field(default=False, description='Whether candidate rewards were exported by this run.')
    rlft_training_data: RLTrainingData | None = Field(default=None, description="Ready RL annotation/reward package for tool09.training_data; null when dense rewards were not requested or no PDM-zero scene is eligible.")
    rlft_data_report: dict[str, Any] | None = Field(default=None, description="Annotation export status, counts and manifest_path; no_eligible_scenes is not a failed simulation.")
    pdm_challenge: dict | None = Field(default=None, description='Independent baseline/attack PDM outcomes; not an official native PDM score.')
