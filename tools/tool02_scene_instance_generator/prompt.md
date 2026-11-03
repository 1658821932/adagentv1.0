Expand one saved failure category into ONE NEW driving-scene requirement, before selecting a base scene.

Return exactly four fields:
- scene_requirement: one detailed coherent paragraph covering road context, ego intent, participants, initial relationships, event progression and the risky interaction. This combines the initial situation, actor behaviors and event sequence; do not output them separately.
- required_conditions: a short list of essential observable road/environment conditions for source retrieval.
- required_actors: a short list of required NON-EGO participants.
- target_effect: the correct policy behavior the scene should test.

Use failure_analysis as the source category. variation_hints are optional qualitative directions. existing_instances are previous descriptions; develop a different scene rather than repeating them.
Use relative positions and temporal ordering; exact speeds, distances, coordinates, track IDs and editing trajectories are unnecessary before retrieval. Specify only conditions relevant to the interaction. Lighting/weather are source-retrieval conditions.
Focus on observable traffic interactions and the required driving response. Write in the input language. Return only JSON matching the schema. Source analysis and identifiers are recorded by code; do not repeat them in the output.
