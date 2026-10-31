import logging
import pickle

import hydra
from omegaconf import DictConfig

from worldengine.runner.builders.worker_pool_builder import build_worker
from worldengine.runner.builders.env_builder import build_envs
from worldengine.runner.executor import run_runners

logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = '../configs'
CONFIG_NAME = 'default_runner'

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2")
def main(cfg: DictConfig) -> None:
    logger.info('WorldEngine is running...')

    # Construct builder
    worker = build_worker(cfg)

    # Construct simulations/environments
    scenes_dict = pickle.load(open(cfg.data_file_path, 'rb'))
    count = cfg.get("num_scenarios")
    if count is not None:
        if type(count) is not int or count < 1:
            raise ValueError("num_scenarios must be a positive integer")
        scenes_dict = dict(list(scenes_dict.items())[:count])
    frame_budget = int(cfg.num_history) + int(cfg.num_future)
    for token, scene in scenes_dict.items():
        length = scene.get("length")
        if length is not None and length < frame_budget:
            raise ValueError(f"Scene {token} has {length} observations; requested {frame_budget}. Padding is not allowed.")
    cfg.max_step = min(int(cfg.max_step), max(1, len(scenes_dict)) * frame_budget)
    envs = build_envs(cfg=cfg, worker=worker, scene_dict=scenes_dict)

    logger.info('Running simulation...')
    try:
        run_runners(envs=envs, worker=worker, cfg=cfg)
    finally:
        for env in envs:
            env.close()
    logger.info('Finished running simulation!')

if __name__ == "__main__":
    main()
