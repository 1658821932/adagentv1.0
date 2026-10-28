"""Backend-aware tool contract for DGGT and UFO."""

DESCRIPTION = {'name': 'edit_config_generator',
 'description': 'Generate a UFO visual lane-change or calibrated metric insert/translate config; '
                'use Smart Edit for DGGT.',
 'when_to_use': 'Use after edit_plan_builder and before scene_editor. Set '
                'output_kind=ego_trajectory_offsets for natural-language ego render trajectory '
                'offsets; otherwise use rigid_config for instance edits.',
 'inputs': {'backend': 'ufo (default Agent) or dggt.',
            'operation': 'UFO: lane_change or metric_edit. Metric requires calibration_path and '
                         'operations.',
            'calibration_path': 'Persisted successful metric_scene calibration from scene_editor '
                                'calibrate_metric.',
            'road_geometry': 'Metric: native (default), or explicit planar_proxy to repair static '
                             'road geometry/false occlusion using a local estimated plane. Changes '
                             'road Gaussians and is not measured geometry.',
            'operations': 'Metric: ordered flat insert '
                          '{operation,object_id,asset_manifest_path,position_m:[forward,left,0],yaw_degrees,motion_reference:world_fixed '
                          'or ego_translation} and translate {operation,target_id:selected or '
                          'prior inserted ID,offset_m:[forward,left,0],start_seconds,end_seconds}. '
                          'Right 3m=[0,-3,0].',
            'direction': 'UFO visual lane_change: required left or right from request/plan.',
            'vehicle_widths': 'UFO: image displacement in target bbox widths, default 1.2, maximum '
                              '2; not metres.',
            'start_fraction': 'UFO: default .1.',
            'end_fraction': 'UFO: default .9.',
            'edit_plan': 'Detailed editing plan. Existing instance trajectory waypoints are '
                         'per-frame offsets added to original positions; compensate the source '
                         "vehicle's existing displacement. For cut-in, hold ego-lane position from "
                         'frame 12 through 39 and use zero height offset. For DGGT require '
                         'asset_path and asset_scale; for UFO metric insertion require a real '
                         'asset_manifest_path.',
            'instance_info_path': 'Existing-object edits: selected_instance_info_path from '
                                  'target_selector; not needed for insert-only UFO jobs.',
            'output_path': 'Required .json configuration path.',
            'output_kind': 'rigid_config for an existing traffic instance, or '
                           'ego_trajectory_offsets for ego render-camera motion (not closed-loop '
                           'policy braking).',
            'frame_count': 'For ego offsets, supply 40; Smart Edit must return one vector or '
                           'exactly 40 vectors.',
            'scene_info': 'Optional scene context.',
            'asset_path': 'Required exact vehicle PLY path when adding or replacing an asset.',
            'asset_scale': 'Required explicit positive asset scale for add or replace; no default '
                           'is assumed.'},
 'outputs': {'saved_path': 'Saved UFO native or DGGT Smart Edit configuration path.',
             'tool_calls': 'Smart Edit MCP tools called by the client.'}}
