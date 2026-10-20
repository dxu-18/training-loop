"""torch-lite torch.optim.lr_scheduler (closed forms of PyTorch's schedules)."""

import math
from bisect import bisect_right


class LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for g in optimizer.param_groups:
                g.setdefault("initial_lr", g["lr"])
        self.base_lrs = [g["initial_lr"] for g in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._step_count = 0
        self.step()

    def _lr(self, base, t, i):
        raise NotImplementedError

    def get_lr(self):
        return [self._lr(b, self.last_epoch, i) for i, b in enumerate(self.base_lrs)]

    def get_last_lr(self):
        return list(self._last_lr)

    def step(self, epoch=None):
        self._step_count += 1
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        for g, lr in zip(self.optimizer.param_groups, self.get_lr()):
            g["lr"] = lr
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "optimizer" and not callable(v)}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


_LRScheduler = LRScheduler


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return base * self.gamma ** (t // self.step_size)


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones, self.gamma = sorted(milestones), gamma
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return base * self.gamma ** bisect_right(self.milestones, t)


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return base * self.gamma ** t


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * t / self.T_max)) / 2


class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        n = len(optimizer.param_groups)
        self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda] * n
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return base * self.lr_lambdas[i](t)


class MultiplicativeLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        n = len(optimizer.param_groups)
        self.lr_lambdas = list(lr_lambda) if isinstance(lr_lambda, (list, tuple)) else [lr_lambda] * n
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > 0:
            return [g["lr"] * f(self.last_epoch) for g, f in zip(self.optimizer.param_groups, self.lr_lambdas)]
        return [g["lr"] for g in self.optimizer.param_groups]


class LinearLR(LRScheduler):
    def __init__(self, optimizer, start_factor=1.0 / 3, end_factor=1.0, total_iters=5, last_epoch=-1):
        if not 0 < start_factor <= 1:
            raise ValueError("Starting multiplicative factor expected to be greater than 0 and less or equal to 1.")
        self.start_factor, self.end_factor, self.total_iters = start_factor, end_factor, total_iters
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return base * (self.start_factor + (self.end_factor - self.start_factor) * min(self.total_iters, t) / self.total_iters)


class ConstantLR(LRScheduler):
    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1):
        self.factor, self.total_iters = factor, total_iters
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return base * (self.factor if t < self.total_iters else 1.0)


class PolynomialLR(LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1):
        self.total_iters, self.power = total_iters, power
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        return base * (1.0 - min(self.total_iters, t) / self.total_iters) ** self.power


class CosineAnnealingWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        self.T_0, self.T_mult, self.eta_min = T_0, T_mult, eta_min
        super().__init__(optimizer, last_epoch)

    def _cycle(self, t):
        if self.T_mult == 1:
            return t % self.T_0, self.T_0
        n = int(math.log(t / self.T_0 * (self.T_mult - 1) + 1, self.T_mult)) if t >= self.T_0 else 0
        return t - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1), self.T_0 * self.T_mult ** n

    def _lr(self, base, t, i):
        cur, Ti = self._cycle(t)
        return self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * cur / Ti)) / 2


class CyclicLR(LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode="triangular", gamma=1.0,
                 last_epoch=-1, **kw):
        for g in optimizer.param_groups:
            g["lr"] = base_lr
        self.base_lr_, self.max_lr_, self.up = base_lr, max_lr, step_size_up
        self.down = step_size_down or step_size_up
        self.mode, self.gamma = mode, gamma
        super().__init__(optimizer, last_epoch)

    def _lr(self, base, t, i):
        total = self.up + self.down
        cycle = math.floor(1 + t / total)
        x = 1.0 + t / total - cycle
        scale = x / (self.up / total) if x <= self.up / total else (x - 1) / (self.up / total - 1)
        amp = (self.max_lr_ - self.base_lr_) * scale
        if self.mode == "triangular2":
            amp /= 2 ** (cycle - 1)
        elif self.mode == "exp_range":
            amp *= self.gamma ** t
        return self.base_lr_ + amp


class OneCycleLR(LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3,
                 anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4, three_phase=False, last_epoch=-1, **kw):
        self.total_steps = total_steps if total_steps is not None else epochs * steps_per_epoch
        n = len(optimizer.param_groups)
        maxes = list(max_lr) if isinstance(max_lr, (list, tuple)) else [max_lr] * n
        for g, m in zip(optimizer.param_groups, maxes):
            g["initial_lr"] = m / div_factor
            g["max_lr"] = m
            g["min_lr"] = g["initial_lr"] / final_div_factor
            g["lr"] = g["initial_lr"]
        T = self.total_steps
        if three_phase:
            self.phases = [(pct_start * T - 1, "initial_lr", "max_lr"), (2 * pct_start * T - 2, "max_lr", "initial_lr"),
                           (T - 1, "initial_lr", "min_lr")]
        else:
            self.phases = [(pct_start * T - 1, "initial_lr", "max_lr"), (T - 1, "max_lr", "min_lr")]
        self.anneal_strategy = anneal_strategy
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        if t > self.total_steps:
            raise ValueError(f"Tried to step {t} times. The specified number of total steps is {self.total_steps}")
        out = []
        for g in self.optimizer.param_groups:
            start = 0
            for i, (end, a, b) in enumerate(self.phases):
                if t <= end or i == len(self.phases) - 1:
                    pct = (t - start) / (end - start)
                    lo, hi = g[a], g[b]
                    if self.anneal_strategy == "cos":
                        out.append(hi + (lo - hi) / 2.0 * (math.cos(math.pi * pct) + 1))
                    else:
                        out.append((hi - lo) * pct + lo)
                    break
                start = end
        return out


class SequentialLR(LRScheduler):
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        self.optimizer, self._schedulers, self._milestones = optimizer, list(schedulers), list(milestones)
        self.last_epoch = last_epoch + 1
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"]
        self._schedulers[0].last_epoch = -1
        self._schedulers[0].step()
        self._last_lr = self._schedulers[0].get_last_lr()

    def step(self, epoch=None):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        sched = self._schedulers[idx]
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            sched.step(0)
        else:
            sched.step()
        self._last_lr = sched.get_last_lr()


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0,
                 min_lr=0, eps=1e-8):
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.optimizer, self.mode, self.factor, self.patience = optimizer, mode, factor, patience
        self.threshold, self.threshold_mode, self.cooldown, self.eps = threshold, threshold_mode, cooldown, eps
        self.min_lrs = list(min_lr) if isinstance(min_lr, (list, tuple)) else [min_lr] * len(optimizer.param_groups)
        self.best = math.inf if mode == "min" else -math.inf
        self.num_bad_epochs, self.cooldown_counter, self.last_epoch = 0, 0, 0
        self._last_lr = [g["lr"] for g in optimizer.param_groups]

    def _better(self, a):
        rel = self.threshold_mode == "rel"
        if self.mode == "min":
            return a < (self.best * (1 - self.threshold) if rel else self.best - self.threshold)
        return a > (self.best * (self.threshold + 1) if rel else self.best + self.threshold)

    def step(self, metrics, epoch=None):
        current = float(metrics)
        self.last_epoch += 1
        if self._better(current):
            self.best, self.num_bad_epochs = current, 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            for g, lo in zip(self.optimizer.param_groups, self.min_lrs):
                new = max(g["lr"] * self.factor, lo)
                if g["lr"] - new > self.eps:
                    g["lr"] = new
            self.cooldown_counter, self.num_bad_epochs = self.cooldown, 0
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]

    def get_last_lr(self):
        return list(self._last_lr)
