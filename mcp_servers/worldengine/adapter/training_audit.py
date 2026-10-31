"""Verify actual optimizer updates and the requested loss family."""
from pathlib import Path
import json
import math
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class AgentTrainingAuditHook(Hook):
    def __init__(self, mode="ilft", **kwargs):
        self.mode = mode
        self.steps = 0
        self.changed = set()
        self.max_delta = 0.

    def before_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        self.before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                       if p.requires_grad and "planning_head" in n}

    def after_train_iter(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        changed = []
        maximum = 0.
        for name, p in model.named_parameters():
            if name in self.before:
                delta = (p.detach().cpu() - self.before[name]).abs().max().item()
                if delta > 0:
                    changed.append(name)
                    maximum = max(maximum, delta)
        losses = {k: float(v) for k, v in runner.outputs.get("log_vars", {}).items()
                  if isinstance(v, (float, int))}
        finite = bool(losses) and all(math.isfinite(v) for v in losses.values())
        required = any("RL" in k for k in losses) if self.mode == "rlft" else "loss.imi" in losses
        self.changed.update(changed)
        self.max_delta = max(self.max_delta, maximum)
        self.steps += 1
        report = dict(training_mode=self.mode, optimizer_steps=self.steps,
            changed_parameter_count=len(self.changed), max_parameter_delta=self.max_delta,
            finite_losses=finite, losses=losses)
        Path(runner.work_dir, "training_audit.json").write_text(json.dumps(report, indent=2))
        if not finite or not required:
            raise RuntimeError("Expected finite mode-specific losses")
