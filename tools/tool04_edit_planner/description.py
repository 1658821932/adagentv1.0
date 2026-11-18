from tools.llm_runtime import make_description
from .schemas import Input, Output

DESCRIPTION = make_description(name='edit_planner',
    description='Review one selected candidate at the first real junction traversed by the recorded trajectory, data first: raw nuPlan map facts and checks, then images, then the three adapted strategies. If the rank-1 scene is unsuitable, continue automatically with tool03 alternatives ranks 2 and 3.',
    when_to_use='Between tool03 retrieval and the attack/ego editors.', input_model=Input, output_model=Output,
    handoff={'input': 'scenario_result: successful tool02 envelope, result or result_ref; candidate_id: the selected candidate; retrieval_result: successful tool03 envelope, result or result_ref whose candidate_id must equal the selection.',
             'output': 'ready: use adapted_plan (three strategies); target_scene and attack are copied from tool02, and evidence.selected_rank identifies the scene that passed. unsuitable after all ranked alternatives: retrieve again using retrieval_suggestion. needs_information: obtain the missing evidence, then review again.'},
    rules=['Map facts come from the raw nuPlan map and are measured by code (physical lanes by lane_fid, turns and directions from baseline paths, the fixed recorded-route junction and its lane connectors); the model only reads them.',
           'For t_junction/cross_intersection, use only the first non-mini junction traversed by the recorded trajectory, including the start junction. A four-arm cross can satisfy a t_junction target when the required approach, lane movements and space exist; extra arms may stay empty. straight_road, curve and merge targets do not need a junction. Do not search later junctions or gate this review on Gaussian asset coverage.',
           'Missing evidence stays unknown. A confirmed mismatch at this fixed junction requires a new retrieval.',
           'Rank-1 unsuitable triggers automatic checks of tool03 alternatives in rank order; return the first ready scene and record retrieval_attempts in evidence.',
           'ready only means the structure supports the scene; no trajectory is generated and no parameter is executable yet.',
           'Source positions, speeds and timings are editable, so they are reasons to adapt the plan, not to reject the scene.',
           'ego_input_edit_strategy must synchronize ego approach speed/timing with the edited attack vehicle at the conflict. Preserve meaningful speed and difficulty; early stopping, crawling or avoiding the conflict is not valid difficulty reduction. ego_correct_trajectory_strategy is only the post-conflict correct response.',
           'This review does not run trajectory optimization, PDM or scene rendering.'])
