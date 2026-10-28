"""Backend-aware tool contract for DGGT and UFO."""

DESCRIPTION = {'name': 'instance_discovery',
 'description': 'Discover important traffic participants from RGB videos.',
 'when_to_use': 'After reconstruction: UFO reuses SAM tracks; DGGT uses YOLO. Both retain multiple '
                'candidates and visual selection.',
 'inputs': {'backend': 'ufo or dggt_yolo; must match the configured reconstruction backend.',
            'ufo_reconstruction': 'UFO only: complete reconstruction result.',
            'image_dir': 'Use reconstruction.result.input_dir, the parent of the input scene '
                         'folder; do not use reconstruction.result.output_dir or append the scene '
                         'name.',
            'output_path': 'DGGT instance_info.json output path.',
            'scene_name': 'Use reconstruction.result.input_scene_name (normally 000); output scene '
                          '001 is not the input scene name.',
            'sequence_length': 'Exactly 40, matching reconstruction num_frames.'},
 'outputs': {'instances': 'Detected car, truck, bus and motorcycle tracks with candidate IDs, '
                          'original track IDs, boxes and continuity summaries.',
             'instance_info_path': 'Candidate snapshot; run target_selector before editing.',
             'object_cards': 'Per-object native-detail crops and matching cyan-box context, with '
                             'hashes and visible frame IDs.',
             'candidate_count': 'Number of retained vehicle-track candidates.'}}
