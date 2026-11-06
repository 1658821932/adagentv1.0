# Task
Use the supplied failure_analysis to invent one new target driving scene. Write in the input language and consider variation_hints and existing_instances.

# Scene design
- scene_requirement describes the target context, ego intent, participant relationships and event progression.
- target_effect states the one correct ego behavior for that target; it is shared by all candidates.
- Propose three meaningful alternatives for constructing this same target, using different source conditions or edit strategies.
- Supported planning operations: add, delete or modify a vehicle, edit its trajectory, and adjust ego trajectory. Prefer at most one changed non-ego vehicle; other traffic can remain unchanged.

# Retrieval and editing
- Each base_video_requirement describes observable source-video content and is passed verbatim to video retrieval.
- Pair it with a qualitative edit_strategy that can create the target. The source need not already contain the final hazard or target vehicle; an empty road/intersection can support adding a vehicle.
- Keep future ego behavior and editing commands out of the retrieval description.
- Exact speeds, distances, coordinates, object IDs and trajectory points are unnecessary before selecting a source.

# Output scope
Produce a target different from existing_instances. Return only the requested scene and candidates; retain source analysis and identifiers in the local run records.
