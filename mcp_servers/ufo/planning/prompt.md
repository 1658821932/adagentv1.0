Build a UFO edit plan using only the request and supplied scene resources.
For operation=metric_edit, operations are flat insert/translate objects matching the schema.
Axes are [forward,left,up] in a fixed road frame; right 3 meters is [0,-3,0].
Existing actors use target_id=selected and require a verified target_selection_path.
Inserted actors use unique object_id values and only supplied asset manifests; reference an inserted object only after its insertion.
Never invent physical units, dimensions, paths, coordinates or timing absent from the request/context. Missing critical information must be reported as an unresolved constraint, not invented.
For operation=lane_change use operation,target,parameters with the selected integer instance ID. Visual bbox-width offsets are not metric distances.
Use relative offsets from the original motion. Do not promise lane hold, ego braking, collision safety, complete novel-view geometry or certified absolute distance accuracy.
Keep target_instances equal to the selected instance ID for existing-actor edits; insert-only jobs have no detected targets.
