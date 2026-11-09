from tools.llm_runtime import make_description
from .schemas import Input, Output
DESCRIPTION=make_description(name='scene_replayer',description='Replay given trajectories through native WorldEngine and export video without running a learned driving policy.',
    when_to_use='Inspect a prepared source or edited scene using its stored trajectories.',input_model=Input,output_model=Output,
    handoff={'input':'Use a non-null simulation_input for the exact intended version and the exact native scene key. An edited scene must be exported first; never substitute its original scene.',
             'output':'videos are native replay artifacts. This tool checks execution and video decoding, not semantic visual quality or driving-policy success.'},
    rules=['Uses trajectory_policy and log_play_controller for ego/background agents, with planner, metrics and dense rewards disabled.',
           'No standalone visual grader is called. Missing native export is a blocker; do not treat frames.json or an expert CSV as an executable package.',
           'Use policy_runner for PDM challenge or VADv2 closed-loop evaluation, and policy_trainer for training.'])
