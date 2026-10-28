"""Backend-aware tool contract for DGGT and UFO."""

DESCRIPTION = {'name': 'scene_editor',
 'description': 'Calibrate a UFO scene, insert known-size Gaussian assets, rigidly translate in '
                'meters, or execute visual lane change; retain DGGT Mode 2.',
 'when_to_use': 'For calibrate_metric use after reconstruction; for editing use a saved '
                'configuration.',
 'inputs': {'backend': 'ufo for visual UFO editing, or dggt for legacy scene editing.',
            'operation': 'UFO: lane_change (visual), calibrate_metric (scale/road calibration), '
                         'metric_edit (rigid insertion/translation).',
            'calibration_method': 'calibrate_metric: frontend_metric verifies existing MoGe/GCA '
                                  'estimated meter units and rigid export; depth_fit (default) '
                                  'independently tests dense depth agreement. Unit verification is '
                                  'not physical accuracy certification.',
            'road_model': 'frontend_metric: local_depth for a local rendered patch, or '
                          'motion_near_depth for sufficiently straight driving; fits pitch from '
                          'estimated camera motion and height from near road, not surveyed '
                          'geometry.',
            'reference_path': 'calibrate_metric only: optional user measured-distance JSON, never '
                              'invent a scale.',
            'road_plane_path': 'calibrate_metric only: optional supplied native road plane JSON.',
            'output_path': 'calibrate_metric: optional calibration JSON output.',
            'output_root': 'metric_edit: optional independent output directory.',
            'assets_only': 'metric_edit: CPU asset/audit export without GPU rendering; not visual '
                           'validation.',
            'ufo_reconstruction': 'For UFO, the complete reconstruction result containing '
                                  'workspace, asset_root and frame_count.',
            'target_selection_path': 'For UFO, the target_selector binding; the editor matches its '
                                     'vehicle to a SAM track.',
            'edit_config_path': 'UFO: saved_path from native visual/metric edit_config_generator. '
                                'DGGT: Smart Edit JSON with a selected trajectory_edit.',
            'gpu': 'For UFO, non-negative physical GPU index.',
            'video_path': 'DGGT only: source driving video.',
            'camera_params_path': 'DGGT only: camera params for coordinate conversion.',
            'meters_to_dggt': 'DGGT only: explicit metric-to-DGGT scale.',
            'coordinate_system': 'DGGT only: smart_edit or dggt_world.',
            'ego_trajectory_offsets_path': 'DGGT only: optional ego render-camera offsets.',
            'num_frames': 'Exactly 40 in the Agent pipeline.'},
 'outputs': {'edited_scene': 'DGGT reconstruction result containing edited render, exact '
                             'scene_config/edit_config_path, and scene artifacts.'}}

DESCRIPTION['ufo_rules'] = """For metric distances or inserted Gaussian assets first calibrate_metric (or reuse calibration_path). frontend_metric verifies upstream estimated metric units and rigid export; absolute_accuracy_certified=false. depth_fit is an independent strict test: failures remain rejected. Supply real asset manifests with explicit units/axes/design dimensions. offset_m=[forward,left,up]; right 3m=[0,-3,0]. metric_edit consumes the saved config plus complete reconstruction. Existing actors require target_selector binding; insert-only jobs do not. road_geometry=planar_proxy is optional and changes static road geometry. Numerical precision does not certify real-world distance accuracy. Original-actor ghosts remain possible; UFO visual validation has no DGGT artifact exemption. UFO closed-loop policy integration is separate."""
