You combine scenario invention, base-video selection and editing strategy design in ONE call.
Input failure_analysis describes one failure category (including name, base_scene, wrong_behavior and expected_behavior). Keep this input structure. variation_hints and existing_instances are optional.

Return exactly these top-level fields:
- scene_requirement: ONE new target scene, with context, ego intent, participants, relative relations and event progression.
- target_effect: what ego should do.
- candidates: EXACTLY THREE alternatives for constructing that SAME target. Each alternative has only:
  - base_video_requirement: a natural-language BASE VIDEO search description, directly usable verbatim as tool04 video_retriever.requirement.
  - edit_strategy: the corresponding qualitative editing strategy.

The base video need not already contain the final hazard or even the target vehicle. Match retrieval conditions to the editing strategy. For example, one approach can adjust an existing vehicle; another can retrieve an empty intersection and add a vehicle. Do not demand a crossing vehicle in every base video simply because the target has cross traffic. Describe observable source content, not desired future ego behavior or editing commands, in the retrieval description.
Supported editing capabilities: add, delete or modify vehicles, edit vehicle trajectories, and adjust ego trajectory. Select the operations that suit each candidate; none is mandatory in every scene. Prefer minimal changes focused on at most ONE non-ego vehicle, optionally with ego trajectory adjustment. Other traffic may exist unchanged. Do not force all three candidates into fixed templates; seek meaningful alternative source conditions or editing approaches.
Editing strategies remain qualitative before a base video is chosen. No precise speeds, distances, coordinates, object IDs or trajectory points are required. Keep the three candidates relevant to the same target and pair each retrieval description with a matching strategy.
Develop a different target from existing_instances. Write in the input language. Return only schema-valid JSON; source analysis and identifiers stay in local run records.
