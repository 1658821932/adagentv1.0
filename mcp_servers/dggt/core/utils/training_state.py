from contextlib import contextmanager
import random

import numpy as np
import torch
import torch.distributed as dist

from utils.training_monitor import save_weights


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


@contextmanager
def validation_mode(model):
    module_modes = [(module, module.training) for module in model.modules()]
    rng_state = capture_rng_state()
    model.eval()
    try:
        yield
    finally:
        for module, training in module_modes:
            module.training = training
        restore_rng_state(rng_state)


def save_resume_checkpoint(path, model, optimizer, scheduler, completed_epoch,
                           best_psnr, config):
    rng_states = [None] * dist.get_world_size()
    dist.all_gather_object(rng_states, capture_rng_state())
    if dist.get_rank() == 0:
        save_weights({
            'format_version': 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'completed_epoch': completed_epoch,
            'global_update': scheduler.last_epoch,
            'best_psnr': best_psnr,
            'config': config,
            'rng_states': rng_states,
        }, path)


def restore_training_state(state, model, optimizer, scheduler, config, rank):
    if state['format_version'] != 1:
        raise ValueError(f"Unsupported resume checkpoint version: {state['format_version']}")
    saved_config = dict(state['config'])
    current_config = dict(config)
    saved_max_epoch = saved_config.pop('max_epoch', None)
    current_max_epoch = current_config.pop('max_epoch', None)
    if saved_config != current_config or (
        saved_max_epoch is not None and current_max_epoch is not None and current_max_epoch < saved_max_epoch
    ):
        raise ValueError(f"Resume configuration mismatch: saved={state['config']} current={config}")
    if len(state['rng_states']) != dist.get_world_size():
        raise ValueError(
            f"RNG state world size mismatch: checkpoint has {len(state['rng_states'])}, "
            f"current world size is {dist.get_world_size()}"
        )
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng_state(state['rng_states'][rank])
    return int(state['completed_epoch'])
