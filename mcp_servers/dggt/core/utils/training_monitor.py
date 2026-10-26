import json
import os

import torch
import torch.distributed as dist


LOSS_NAMES = ('loss', 'l1', 'sky', 'lifespan', 'dynamic', 'lpips')


def validation_due(epoch, total_epochs, interval):
    return interval > 0 and (epoch % interval == 0 or epoch == total_epochs)


class BestValidationCheckpoint:
    """Keep the highest named validation metric across runs using one log directory."""

    def __init__(self, checkpoint_dir, metric='validation_psnr'):
        self.checkpoint_dir = checkpoint_dir
        self.metric = metric
        self.value = float('-inf')
        metadata_path = os.path.join(checkpoint_dir, 'best.json')
        if os.path.isfile(metadata_path):
            with open(metadata_path) as handle:
                record = json.load(handle)
            if record['metric'] == self.metric:
                self.value = float(record['value'])

    @property
    def psnr(self):
        """Backward-compatible alias used by the existing resume checkpoint format."""
        return self.value

    @psnr.setter
    def psnr(self, value):
        self.value = float(value)

    def update(self, state_dict, epoch, value):
        if value > self.value:
            save_best_weights(state_dict, self.checkpoint_dir, epoch, self.metric, value)
            self.value = value
            return True
        return False


class EpochLosses:
    """Sample-weighted averages over all batches and distributed ranks."""

    def __init__(self, device):
        self.totals = torch.zeros(len(LOSS_NAMES) + 1, dtype=torch.float64, device=device)

    def add(self, losses, samples):
        self.totals[:-1] += torch.stack([value.detach() for value in losses]) * samples
        self.totals[-1] += samples

    def finish(self):
        dist.all_reduce(self.totals, op=dist.ReduceOp.SUM)
        values = (self.totals[:-1] / self.totals[-1]).tolist()
        return dict(zip(LOSS_NAMES, values)), int(self.totals[-1].item())


def write_epoch_metrics(log_dir, epoch, updates, losses, samples, lr_used, lr_next, lpips_weight):
    record = dict(epoch=epoch, global_update=updates, samples=samples, **losses,
                  lr_used=lr_used, lr_next=lr_next, lpips_weight=lpips_weight)
    with open(os.path.join(log_dir, 'train_metrics.jsonl'), 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    print(f"[Epoch {epoch}] " + ' | '.join(f'{key}={value:.6f}' for key, value in losses.items())
          + f' | LR={lr_used}', flush=True)


def save_weights(state_dict, path):
    """Publish a complete checkpoint without truncating its previous version."""
    torch.save(state_dict, path + '.tmp')
    os.replace(path + '.tmp', path)


def save_best_weights(state_dict, checkpoint_dir, epoch, metric, value):
    save_weights(state_dict, os.path.join(checkpoint_dir, 'best.pt'))
    metadata_path = os.path.join(checkpoint_dir, 'best.json')
    with open(metadata_path + '.tmp', 'w') as handle:
        json.dump(dict(epoch=epoch, metric=metric, value=value), handle, indent=2)
    os.replace(metadata_path + '.tmp', metadata_path)
