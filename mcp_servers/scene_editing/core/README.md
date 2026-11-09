# Scene editing backend integration

`adversarial/` contains the independent attack, PDM challenge, ego understanding/editing and WorldEngine rendering backend.

- `editing_backend.py` implements the versioned `editing_stage.v1` JSON protocol used by the adapter.
- `pdm_tool.py` implements the baseline/attack PDM challenge used by `policy_runner`.
- `backend_settings.json` declares PDM/render environments, budgets and the public video root.
- The remaining scripts are the actual editing, checking, rendering and support dependencies.

See `docs/EDITING_INTERFACES.md` for architecture, configuration, output layout and validation status. Backend commands are configured in `scene_editing/adapter/settings.json`; contract tests do not replace real scene validation.

`adversarial/defense_generation_contract.py` declares the structured construction/FOCI requests; `defense_generation.py` routes them through the existing editor graph, shared candidate budget and the default FOCI stage. See `docs/EDITING_INTERFACES.md` for exact fields and verification scope.
