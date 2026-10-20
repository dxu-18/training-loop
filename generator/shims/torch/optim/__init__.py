"""torch-lite torch.optim: the update rules follow PyTorch's formulas (single-tensor, in place, float32)."""

import math

import numpy as np

import torch

from . import lr_scheduler  # noqa: F401


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = dict(defaults)
        self.state = {}
        self.param_groups = []
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(params[0], dict):
            params = [{"params": params}]
        for g in params:
            self.add_param_group(g)

    def add_param_group(self, group):
        group = dict(group)
        ps = group["params"]
        group["params"] = [ps] if isinstance(ps, torch.Tensor) else list(ps)
        for p in group["params"]:
            if not p.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")
        seen = {id(p) for g in self.param_groups for p in g["params"]}
        if any(id(p) in seen for p in group["params"]):
            raise ValueError("some parameters appear in more than one parameter group")
        for k, v in self.defaults.items():
            group.setdefault(k, v)
        self.param_groups.append(group)

    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad._a[...] = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad._a.astype(p._a.dtype)
                if g.get("maximize"):
                    grad = -grad
                st = self.state.setdefault(p, {})
                new = self._update(p._a, grad, st, g)
                p._a[...] = torch._to_bf16(new) if p._bf16 else new
        return loss

    def _update(self, w, grad, st, g):
        raise NotImplementedError

    def state_dict(self):
        ids, groups, k = {}, [], 0
        for g in self.param_groups:
            idx = []
            for p in g["params"]:
                if id(p) not in ids:
                    ids[id(p)] = k
                    k += 1
                idx.append(ids[id(p)])
            groups.append({**{a: b for a, b in g.items() if a != "params"}, "params": idx})
        state = {}
        for g in self.param_groups:
            for p in g["params"]:
                if p in self.state:
                    state[ids[id(p)]] = {a: (torch.Tensor._wrap(np.array(b)) if isinstance(b, np.ndarray) else b)
                                         for a, b in self.state[p].items()}
        return {"state": state, "param_groups": groups}

    def load_state_dict(self, state_dict):
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        params = [p for g in self.param_groups for p in g["params"]]
        for g, saved in zip(self.param_groups, groups):
            if len(g["params"]) != len(saved["params"]):
                raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")
            g.update({k: v for k, v in saved.items() if k != "params"})
        self.state = {}
        for i, st in state_dict["state"].items():
            self.state[params[int(i)]] = {k: (np.array(torch._arr(v)) if isinstance(v, torch.Tensor) else v) for k, v in st.items()}

    def __repr__(self):
        lines = [f"{type(self).__name__} ("]
        for i, g in enumerate(self.param_groups):
            lines.append(f"Parameter Group {i}")
            for k in sorted(k for k in g if k != "params"):
                lines.append(f"    {k}: {g[k]}")
        return "\n".join(lines) + "\n)"


def _f(x, w):
    return np.asarray(x, dtype=w.dtype)


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False, *, maximize=False,
                 foreach=None, differentiable=False, fused=None):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay,
                                      nesterov=nesterov, maximize=maximize))

    def _update(self, w, grad, st, g):
        if g["weight_decay"]:
            grad = grad + _f(g["weight_decay"], w) * w
        mu = g["momentum"]
        if mu:
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = grad.copy()
            else:
                st["momentum_buffer"] = _f(mu, w) * st["momentum_buffer"] + _f(1 - g["dampening"], w) * grad
            buf = st["momentum_buffer"]
            grad = grad + _f(mu, w) * buf if g["nesterov"] else buf
        return w - _f(g["lr"], w) * grad


class Adam(Optimizer):
    _decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, *, maximize=False,
                 foreach=None, capturable=False, differentiable=False, fused=None):
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                                      maximize=maximize))

    def _update(self, w, grad, st, g):
        b1, b2 = g["betas"]
        lr, wd = g["lr"], g["weight_decay"]
        if not st:
            st["step"] = 0.0
            st["exp_avg"] = np.zeros_like(w)
            st["exp_avg_sq"] = np.zeros_like(w)
            if g["amsgrad"]:
                st["max_exp_avg_sq"] = np.zeros_like(w)
        st["step"] = float(st["step"]) + 1
        t = st["step"]
        if wd and self._decoupled:
            w = w * _f(1 - lr * wd, w)
        elif wd:
            grad = grad + _f(wd, w) * w
        st["exp_avg"] = _f(b1, w) * st["exp_avg"] + _f(1 - b1, w) * grad
        st["exp_avg_sq"] = _f(b2, w) * st["exp_avg_sq"] + _f(1 - b2, w) * grad * grad
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        v = st["exp_avg_sq"]
        if g["amsgrad"]:
            st["max_exp_avg_sq"] = np.maximum(st["max_exp_avg_sq"], v)
            v = st["max_exp_avg_sq"]
        denom = np.sqrt(v) / _f(math.sqrt(bc2), w) + _f(g["eps"], w)
        return w - _f(lr / bc1, w) * st["exp_avg"] / denom


class AdamW(Adam):
    _decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, *, maximize=False,
                 foreach=None, capturable=False, differentiable=False, fused=None):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, maximize=maximize)


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, *, maximize=False,
                 foreach=None, capturable=False, differentiable=False):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                                      centered=centered, maximize=maximize))

    def _update(self, w, grad, st, g):
        a = g["alpha"]
        if not st:
            st["step"] = 0.0
            st["square_avg"] = np.zeros_like(w)
            if g["momentum"]:
                st["momentum_buffer"] = np.zeros_like(w)
            if g["centered"]:
                st["grad_avg"] = np.zeros_like(w)
        st["step"] = float(st["step"]) + 1
        if g["weight_decay"]:
            grad = grad + _f(g["weight_decay"], w) * w
        st["square_avg"] = _f(a, w) * st["square_avg"] + _f(1 - a, w) * grad * grad
        if g["centered"]:
            st["grad_avg"] = _f(a, w) * st["grad_avg"] + _f(1 - a, w) * grad
            avg = np.sqrt(st["square_avg"] - st["grad_avg"] ** 2) + _f(g["eps"], w)
        else:
            avg = np.sqrt(st["square_avg"]) + _f(g["eps"], w)
        if g["momentum"]:
            st["momentum_buffer"] = _f(g["momentum"], w) * st["momentum_buffer"] + grad / avg
            return w - _f(g["lr"], w) * st["momentum_buffer"]
        return w - _f(g["lr"], w) * grad / avg


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10, *, maximize=False,
                 foreach=None, differentiable=False, fused=None):
        super().__init__(params, dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps, maximize=maximize,
                                      initial_accumulator_value=initial_accumulator_value))

    def _update(self, w, grad, st, g):
        if not st:
            st["step"] = 0.0
            st["sum"] = np.full_like(w, g["initial_accumulator_value"])
        st["step"] = float(st["step"]) + 1
        if g["weight_decay"]:
            grad = grad + _f(g["weight_decay"], w) * w
        clr = g["lr"] / (1 + (st["step"] - 1) * g["lr_decay"])
        st["sum"] = st["sum"] + grad * grad
        return w - _f(clr, w) * grad / (np.sqrt(st["sum"]) + _f(g["eps"], w))


def __getattr__(name):
    raise AttributeError(f"'torch.optim.{name}' is not available in the browser's torch-lite.")
