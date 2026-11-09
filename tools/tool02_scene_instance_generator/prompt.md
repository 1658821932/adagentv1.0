# Task
Given a failure analysis, generate 3 scenario variants that reproduce the SAME failure mechanism. Write in the input language.

# Key principle
Only reproduce the failure mechanism. Do not add extra interactions, participants, or complexity.

# Output fields
- scene_requirement: ONE short sentence. Just the essence: what kind of road + what ego does wrong. Example: "ego enters a left curve too fast and drifts off the road edge".
- target_effect: ONE short sentence. What ego should do instead. Example: "ego slows before the curve and stays in lane".
- candidates: 3 variants. Each has:
  - base_video_requirement: ONE short sentence. What road + what vehicles are visible. Each variant should be a DIFFERENT road context (e.g. urban curve, rural highway curve, mountain road curve).
  - edit_strategy: What to edit about OTHER vehicles in the retrieved video to make the scene work for reproducing this failure. Set to null if the scene works as-is without touching other vehicles. Do NOT write "keep unchanged" — that is null. Do NOT mention ego here.

# Rules
- NEVER write ego actions in edit_strategy.
- Each variant must have a DIFFERENT road context in base_video_requirement.
- edit_strategy is about what the scene NEEDS for construction, not about the failure type. For example, if the failure is ego overspeeding in a curve and the retrieved video already has a clean curve with no interfering vehicles, edit_strategy is null. But if the video has a vehicle blocking the curve entry, edit_strategy could be "remove the vehicle ahead".
- ONE short sentence per field. No technical details.
