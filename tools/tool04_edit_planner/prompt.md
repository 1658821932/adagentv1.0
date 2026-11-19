# Task
Review the retrieved scene against scene_requirement and edit_strategy. Decide if this base is worth editing, and give a small set of coarse attack-vehicle alternatives if needed. Keep target_effect unchanged for tool06.

# Visual evidence
You receive up to 12 CAM_F0 keyframes as three chronological 2x2 contact sheets. Read camera/frame/time labels; blank cells are missing, not observations.
When available, an additional BEV shows the recorded road/actor layout and blue recorded ego reference trajectory. Read its time and actor labels against the exact catalog IDs. It is source geometry, not an edited scene or VAD rollout. Only some actors are labelled; omitted objects still exist. An unavailable BEV stays unknown.
If you need another view, return {"action":"request_views","cameras":["CAM_L0"],"purpose":"what to check"}. Up to 2 cameras per round, 2 extra rounds.
For a particular vehicle's recorded trajectory, return {"action":"request_trajectory","track_id":"<exact catalog ID>","max_samples":24,"purpose":"specific timing or motion question"}; use track_id="ego" for the recorded ego reference. At most 3 trajectory queries, 24 observations each. The reader returns a labelled trajectory image and sampled coordinates; missing observations stay unknown. Query only unresolved candidates; reuse returned evidence IDs rather than repeating requests. Full original evidence is archived; do not reconstruct exact trajectories from rounded catalog numbers.
When you have enough evidence, return the final decision object.

# Ego history through 1.5s
- On a ready decision that needs another vehicle, provide one `ego_history_plan` shared by all edit options. Read `ego_input_edit_strategy` and the scene-local samples in `ego_history_geometry`.
- Use `mode=keep` when the recorded takeover is suitable or there is no supported reason to change it. Use `mode=speed` with `target_speed_mps` for a small change along the recorded route. Use `mode=route` when the history needs a pose change as well. Provide `route.end_xy_m` for the ego rear axle at 1.5s, and optionally end_yaw_rad, end_speed_mps, via_xy_m, via_time_s and connector_id. x is forward, y is left, and yaw is counterclockwise. Use only a mapped connector ID. The prefix can approach a turn that occurs after 1.5s.
- Estimate reachable travel from the observed start speed and 1.5s horizon. For route mode, prefer endpoints near the recorded approach and use an intermediate point only when needed. Keep this intent in `ego_history_plan`; the other-vehicle instructions own only the other actor's intended behavior and interaction with the resulting scene.

# Decision
- ready: base scene is suitable. If no actor edit is needed, edit_options=[] and edit_plan=null. Otherwise provide edit_options with 1..3 distinct, unranked alternatives and edit_plan=null. Usually two useful alternatives suffice; do not force a count. Each option contains option_id, operation (edit_existing/add_vehicle), concise instruction, rationale and uncertainty. An existing-vehicle option may carry an observed target_track. An added-vehicle option may carry stationary or left_cut_in as a legacy maneuver hint when it fits. For a mapped turn, describe the connector and direction in instruction and omit maneuver. Consider existing actors and added actors in the SAME review. Tool05 chooses one or adapts it; do not solve every trajectory here.
- unsuitable: layout or actor mismatch. Provide retrieval_suggestion, no edit_plan.
- needs_information: missing resources (e.g. no usable vehicle asset). Name what is needed. If the scene is otherwise suitable and only the vehicle asset is missing, still provide the add_vehicle edit_options so they can be reused after the asset is supplied; attack_input remains null until then.

# Rules
- Current workflow: propose attack mechanisms; tool05 chooses, edits and by default tests with native VADv2 before expert ego editing. After05, Main Agent may run expert editing and policy_runner(native) on the frozen attack package independently for complete scoring. No internal PDM planner or three-second edit delay; edits may start at 0s.
- Use edit_strategy as a starting idea, not a fixed operation. Offer meaningfully different supported alternatives in one review, including an added vehicle when useful even if existing vehicles are present. Explain evidence and uncertainty; do not silently repeat a known rejected target. Do not propose options that all depend on unavailable assets when existing actors can carry the mechanism.
- Do not reject a whole scene because the original future trajectory reverses, drifts or has annotation noise. Future motion will be edited. Tool04 reviews layout and resources; tool05 validates supported initial state and generated motion. Recorded distance hints do not exclude a fully observed vehicle by themselves.
- Respect editing_capabilities: edit_existing starts from the existing vehicle's recorded t=0 pose by default. Tool05 may create a new version with a bounded initial-position adjustment and optional initial yaw alignment/change; it cannot silently teleport within one trajectory. Any explicit user pose lock remains binding. For an additional stationary obstacle use add_vehicle/stationary with a supplied asset, or explain the missing capability. Initial-pose changes and generated motion are fully rechecked by tool05.
- Every edit option you provide must help construct a scene similar to the failure mechanism.
- The default attack event window is 1.5s <= t <= 7.5s, matching the configured native scoring horizon.
  The program passes the evaluation window to tool05; do not repeat it in every option. Edits and setup braking/steering may start at 0s;
  WE observation history is not an immutable trajectory prefix. A hazard reached only at t=9s is outside
  this evaluation. Use a different event window only when the user explicitly requests one, and report
  any mismatch with the scoring horizon rather than silently changing the evaluator.
- Propose plausible mechanisms for an actual policy failure, rather than declaring a need to brake
  sufficient. New tool05 runs require reference target-body contact inside the event window before VAD eligibility and actively optimize that contact. Tool05 must then observe VADv2's response and refine the scene. Reference contact is required setup evidence but not proof of policy failure; tool06 still needs a feasible expert defense.
- Never place a hazard the ego only reaches after the rollout ends; that scene scores as a success.
- The edit must reproduce the failure mechanism, so the operation follows from what the mechanism needs:
  whatever actor or geometry the ego had to misread, misjudge or fail to yield to has to exist in the
  edited scene and behave so that the same wrong decision is available again. Pick the operation that
  expresses that, not the one that is easiest to construct.
- Worked cases, used as illustrations of that rule and not as an exhaustive list:
  * rear-end or late braking: slow/hold/brake an appropriately placed existing lead, or add a stationary lead at a supported new initial position;
  * cut-in or merge: move an existing neighbour from its real starting pose, or add a vehicle with a plausible initial pose;
  * road departure or squeeze: use an existing or added neighbour whose feasible motion narrows the ego corridor.
- Existing and added actors are alternatives. Judge their ability to express the mechanism with the available tools; actor presence alone does not make edit_existing preferable. Describe alternatives; the auto handoff lets tool05 choose or switch operations/actors within shared budgets. Preserve any explicit user requirement instead of inventing one from a preference.
- Tool05A executes the ego history plan locally and freezes its measured result. Tool05B starts its own decision context from that scene and the observed takeover state. Each attack option edits ONE other actor: edit_existing (from eligible_vehicle_tracks) or add_vehicle. The final frozen scene contains ONE chosen edited/added actor.
- Keep each instruction to one or two short sentences: describe the mechanism, a qualitative area
  relative to the road/ego, and the actor's intended behavior. Rationale and uncertainty should each
  be one short sentence. Leave exact XY, yaw, speed, braking strength, start/stop times and fine
  placement to tool05 and its VAD feedback. Do not prescribe holding from t=0 unless essential to
  the mechanism or explicitly requested. Put useful numeric observations in evidence, not commands.
- Options are starting hypotheses, not ranked recommendations or hard contracts. Tool05 may adjust
  or switch them after testing; never declare an untested alternative ineffective or unnecessary.
- Do NOT estimate asset dimensions from images; use the supplied vehicle asset data.
