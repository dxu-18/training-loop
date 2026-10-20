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


class Adamax(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, *, maximize=False, foreach=None,
                 differentiable=False, capturable=False):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, maximize=maximize))

    def _update(self, w, grad, st, g):
        b1, b2 = g["betas"]
        if not st:
            st["step"], st["exp_avg"], st["exp_inf"] = 0.0, np.zeros_like(w), np.zeros_like(w)
        st["step"] = float(st["step"]) + 1
        if g["weight_decay"]:
            grad = grad + _f(g["weight_decay"], w) * w
        st["exp_avg"] = _f(b1, w) * st["exp_avg"] + _f(1 - b1, w) * grad
        st["exp_inf"] = np.maximum(_f(b2, w) * st["exp_inf"], np.abs(grad) + _f(g["eps"], w))
        clr = g["lr"] / (1 - b1 ** st["step"])
        return w - _f(clr, w) * st["exp_avg"] / st["exp_inf"]


class NAdam(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=4e-3,
                 decoupled_weight_decay=False, *, foreach=None, maximize=False, capturable=False, differentiable=False):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, maximize=maximize,
                                      momentum_decay=momentum_decay, decoupled_weight_decay=decoupled_weight_decay))

    def _update(self, w, grad, st, g):
        b1, b2 = g["betas"]
        lr, wd, md = g["lr"], g["weight_decay"], g["momentum_decay"]
        if not st:
            st["step"], st["mu_product"] = 0.0, 1.0
            st["exp_avg"], st["exp_avg_sq"] = np.zeros_like(w), np.zeros_like(w)
        st["step"] = t = float(st["step"]) + 1
        if wd and g["decoupled_weight_decay"]:
            w = w * _f(1 - lr * wd, w)
        elif wd:
            grad = grad + _f(wd, w) * w
        mu = b1 * (1.0 - 0.5 * 0.96 ** (t * md))
        mu_next = b1 * (1.0 - 0.5 * 0.96 ** ((t + 1) * md))
        st["mu_product"] *= mu
        st["exp_avg"] = _f(b1, w) * st["exp_avg"] + _f(1 - b1, w) * grad
        st["exp_avg_sq"] = _f(b2, w) * st["exp_avg_sq"] + _f(1 - b2, w) * grad * grad
        denom = np.sqrt(st["exp_avg_sq"] / _f(1 - b2 ** t, w)) + _f(g["eps"], w)
        w = w - _f(lr * (1 - mu) / (1 - st["mu_product"]), w) * grad / denom
        return w - _f(lr * mu_next / (1 - st["mu_product"] * mu_next), w) * st["exp_avg"] / denom


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, decoupled_weight_decay=False, *,
                 foreach=None, maximize=False, capturable=False, differentiable=False):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, maximize=maximize,
                                      decoupled_weight_decay=decoupled_weight_decay))

    def _update(self, w, grad, st, g):
        b1, b2 = g["betas"]
        lr, wd = g["lr"], g["weight_decay"]
        if not st:
            st["step"], st["exp_avg"], st["exp_avg_sq"] = 0.0, np.zeros_like(w), np.zeros_like(w)
        st["step"] = t = float(st["step"]) + 1
        if wd and g["decoupled_weight_decay"]:
            w = w * _f(1 - lr * wd, w)
        elif wd:
            grad = grad + _f(wd, w) * w
        st["exp_avg"] = _f(b1, w) * st["exp_avg"] + _f(1 - b1, w) * grad
        st["exp_avg_sq"] = _f(b2, w) * st["exp_avg_sq"] + _f(1 - b2, w) * grad * grad
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        m_hat = st["exp_avg"] / _f(bc1, w)
        rho_inf = 2 / (1 - b2) - 1
        rho_t = rho_inf - 2 * t * (b2 ** t) / bc2
        if rho_t > 5.0:
            rect = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
            adaptive = _f(math.sqrt(bc2), w) / (np.sqrt(st["exp_avg_sq"]) + _f(g["eps"], w))
            return w - m_hat * _f(lr * rect, w) * adaptive
        return w - _f(lr, w) * m_hat


class LBFGS(Optimizer):
    """PyTorch's L-BFGS without a line search (line_search_fn=None): `step(closure)` runs up to max_iter iterations."""

    def __init__(self, params, lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9,
                 history_size=100, line_search_fn=None):
        if line_search_fn is not None:
            raise NotImplementedError("torch-lite LBFGS supports line_search_fn=None only")
        super().__init__(params, dict(lr=lr, max_iter=max_iter, max_eval=max_eval if max_eval is not None else max_iter * 5 // 4,
                                      tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                                      history_size=history_size, line_search_fn=None))
        self._params = self.param_groups[0]["params"]

    def _flat_grad(self):
        return np.concatenate([(p.grad._a if p.grad is not None else np.zeros_like(p._a)).reshape(-1).astype(np.float32)
                               for p in self._params])

    def _add_grad(self, t, d):
        off = 0
        for p in self._params:
            n = p._a.size
            p._a[...] = (p._a.reshape(-1) + np.float32(t) * d[off:off + n]).reshape(p._a.shape)
            off += n

    def step(self, closure):
        g = self.param_groups[0]
        lr, max_iter, max_eval = g["lr"], g["max_iter"], g["max_eval"]
        tol_grad, tol_change, hist = g["tolerance_grad"], g["tolerance_change"], g["history_size"]
        state = self.state.setdefault(self._params[0], {"func_evals": 0, "n_iter": 0})
        with torch.enable_grad():
            orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1
        flat_grad = self._flat_grad()
        if np.abs(flat_grad).max() <= tol_grad:
            return orig_loss
        d, t = state.get("d"), state.get("t")
        old_dirs, old_stps, ro = state.get("old_dirs", []), state.get("old_stps", []), state.get("ro", [])
        H_diag, prev_flat_grad = state.get("H_diag", 1), state.get("prev_flat_grad")
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1
            if state["n_iter"] == 1:
                d, old_dirs, old_stps, ro, H_diag = -flat_grad, [], [], [], 1
            else:
                y = flat_grad - prev_flat_grad
                s = d * np.float32(t)
                ys = float(y @ s)
                if ys > 1e-10:
                    if len(old_dirs) == hist:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    H_diag = ys / float(y @ y)
                al = [0.0] * len(old_dirs)
                q = -flat_grad
                for i in range(len(old_dirs) - 1, -1, -1):
                    al[i] = float(old_stps[i] @ q) * ro[i]
                    q = q - np.float32(al[i]) * old_dirs[i]
                r = q * np.float32(H_diag)
                for i in range(len(old_dirs)):
                    be_i = float(old_dirs[i] @ r) * ro[i]
                    r = r + np.float32(al[i] - be_i) * old_stps[i]
                d = r
            prev_flat_grad = flat_grad.copy()
            prev_loss = loss
            t = min(1.0, 1.0 / float(np.abs(flat_grad).sum())) * lr if state["n_iter"] == 1 else lr
            gtd = float(flat_grad @ d)
            if gtd > -tol_change:
                break
            self._add_grad(t, d)
            ls_evals = 0
            if n_iter != max_iter:
                with torch.enable_grad():
                    loss = float(closure())
                flat_grad = self._flat_grad()
                ls_evals = 1
            current_evals += ls_evals
            state["func_evals"] += ls_evals
            if n_iter == max_iter or current_evals >= max_eval or np.abs(flat_grad).max() <= tol_grad:
                break
            if np.abs(d * np.float32(t)).max() <= tol_change or abs(loss - prev_loss) < tol_change:
                break
        state.update(d=d, t=t, old_dirs=old_dirs, old_stps=old_stps, ro=ro, H_diag=H_diag, prev_flat_grad=prev_flat_grad)
        return orig_loss


def __getattr__(name):
    raise AttributeError(f"'torch.optim.{name}' is not available in the browser's torch-lite.")
