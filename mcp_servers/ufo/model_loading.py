"""Pass configured camera count through the original UFO checkpoint loader."""
from __future__ import annotations

import functools
import importlib
from pathlib import Path
import sys


def install_camera_count_adapter(inference) -> None:
    original = inference.build_model
    if getattr(original, '_ufo_camera_count_adapter', False):
        return

    @functools.wraps(original)
    def build_model(args, device):
        count = getattr(args, 'num_max_cameras', None)
        if type(count) is not int or count <= 0:
            raise ValueError('UFO num_max_cameras must be a positive integer')
        registry = inference.UFO_models
        factory = registry[args.model]

        def configured_factory(*positional, **keywords):
            if 'num_cams' in keywords and keywords['num_cams'] != count:
                raise ValueError('UFO constructor camera count conflicts with inference config')
            keywords['num_cams'] = count
            return factory(*positional, **keywords)

        # The source loader still reads and loads the original state_dict.
        # Only its local factory lookup receives the missing constructor arg.
        inference.UFO_models = {**registry, args.model: configured_factory}
        try:
            print(f'UFO_MODEL_CONFIG num_cams={count} (original checkpoint loader)', flush=True)
            return original(args, device)
        finally:
            inference.UFO_models = registry

    build_model._ufo_camera_count_adapter = True
    inference.build_model = build_model


def install_for_source(source: Path) -> None:
    sys.path.insert(0, str(source))
    inference = importlib.import_module('inference')
    if Path(inference.__file__).resolve() != (source / 'inference.py').resolve():
        raise ValueError(f'Unexpected UFO inference module: {inference.__file__}')
    install_camera_count_adapter(inference)
