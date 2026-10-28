"""Backend-aware tool contract for DGGT and UFO."""

DESCRIPTION = {'name': 'reconstruction',
 'description': 'Reconstruct a single-view driving video with the UFO WorldEngine time-v2 pipeline '
                '(default) or legacy DGGT. UFO returns its native asset; default edits are visual '
                'and do not claim metre calibration.',
 'when_to_use': 'Use after video_retriever when the task requires 3D scene reconstruction.',
 'inputs': {'video_path': 'Absolute server path of the single-view driving video.',
            'backend': 'ufo (default) or dggt (legacy).',
            'scene_name': 'Optional unique UFO scene ID.',
            'gpu': 'Optional non-negative physical GPU index for UFO.',
            'num_frames': 'Exactly 40 in the Agent pipeline; UFO trims longer videos to their '
                          'first 40 frames.',
            'export_render_state': 'DGGT only: export full native temporal Gaussians for faithful '
                                   'WorldEngine rendering.'},
 'outputs': {'input_dir': 'DGGT only: parent directory of preprocessed input frames. For UFO pass '
                          'the complete ufo_reconstruction result.',
             'input_scene_name': 'Input scene name for instance_discovery.scene_name, normally '
                                 '000; distinct from output scene 001.',
             'scene_ply': 'DGGT only: combined Gaussian PLY. UFO returns asset_manifest, '
                          'camera_file and workspace instead.',
             'render_state': 'Native temporal Gaussian tensor file when export_render_state=true; '
                             'must be converted to a DGGTScene WorldEngine asset, not treated as a '
                             'PLY.',
             'camera_params': 'DGGT only: camera parameters aligned with PLY. UFO returns '
                              'camera_file; no metric calibration is required for visual lane '
                              'change.',
             'rendered_video': 'Rendered reconstruction video.',
             'output_dir': 'DGGT reconstruction output directory.'}}
