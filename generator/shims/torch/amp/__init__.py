"""torch-lite AMP: autocast (matmul-class ops run in low precision) and GradScaler."""

import numpy as np

import torch
from torch import autocast  # noqa: F401


class GradScaler:
    def __init__(self, device="cuda", init_scale=2.0 ** 16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        self._enabled = enabled
        self._scale = float(init_scale)
        self._growth_factor, self._backoff_factor, self._growth_interval = growth_factor, backoff_factor, growth_interval
        self._growth_tracker = 0
        self._found_inf = {}

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale if self._enabled else 1.0

    def scale(self, outputs):
        if not self._enabled:
            return outputs
        return outputs * self._scale

    def unscale_(self, optimizer):
        if not self._enabled:
            return
        if id(optimizer) in self._found_inf:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        inv, found = 1.0 / self._scale, False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad._a = p.grad._a * np.asarray(inv, dtype=p.grad._a.dtype)
                    found = found or not np.all(np.isfinite(p.grad._a))
        self._found_inf[id(optimizer)] = found

    def step(self, optimizer, *args, **kwargs):
        if not self._enabled:
            return optimizer.step(*args, **kwargs)
        if id(optimizer) not in self._found_inf:
            self.unscale_(optimizer)
        if self._found_inf[id(optimizer)]:
            return None  # skip the step: gradients overflowed
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale)
        elif any(self._found_inf.values()):
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker == self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = {}

    def state_dict(self):
        return {"scale": self._scale, "growth_factor": self._growth_factor, "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval, "_growth_tracker": self._growth_tracker}

    def load_state_dict(self, sd):
        self._scale, self._growth_tracker = sd["scale"], sd["_growth_tracker"]
        self._growth_factor, self._backoff_factor, self._growth_interval = sd["growth_factor"], sd["backoff_factor"], sd["growth_interval"]
