"""torch-lite torch.distributed.checkpoint.state_dict helpers."""

import dataclasses


@dataclasses.dataclass
class StateDictOptions:
    full_state_dict: bool = False
    cpu_offload: bool = False
    ignore_frozen_params: bool = False
    keep_submodule_prefixes: bool = True
    strict: bool = True
    broadcast_from_rank0: bool = False


def _clean(sd):
    return {k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "").replace("module.", "", 1)
            if k.startswith("module.") else k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", ""): v
            for k, v in sd.items()}


def _opts(options):
    return options or StateDictOptions()


def get_model_state_dict(model, *, submodules=None, options=None):
    sd = _clean(model.state_dict())
    if _opts(options).ignore_frozen_params:
        frozen = {k for k, p in _clean(dict(model.named_parameters())).items() if not p.requires_grad}
        sd = {k: v for k, v in sd.items() if k not in frozen}
    return sd


def get_optimizer_state_dict(model, optimizers, *, submodules=None, options=None):
    opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
    names = {id(p): k for k, p in _clean(dict(model.named_parameters())).items()}
    out = {"state": {}, "param_groups": []}
    for opt in opts:
        sd = opt.state_dict()
        params = [p for g in opt.param_groups for p in g["params"]]
        for i, st in sd["state"].items():
            out["state"][names[id(params[int(i)])]] = st
        for g, saved in zip(opt.param_groups, sd["param_groups"]):
            out["param_groups"].append({**{k: v for k, v in saved.items() if k != "params"},
                                        "params": [names[id(p)] for p in g["params"]]})
    return out


def get_state_dict(model, optimizers, *, submodules=None, options=None):
    return get_model_state_dict(model, options=options), get_optimizer_state_dict(model, optimizers, options=options)


def set_model_state_dict(model, model_state_dict, *, options=None):
    own = model.state_dict()
    prefix = next((p for p in ("module.", "_orig_mod.") if all(k.startswith(p) for k in own)), "")
    return model.load_state_dict({prefix + k: v for k, v in model_state_dict.items()}, strict=_opts(options).strict)


def set_optimizer_state_dict(model, optimizers, optim_state_dict, *, options=None):
    opts = optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]
    names = {id(p): k for k, p in _clean(dict(model.named_parameters())).items()}
    for opt in opts:
        params = [p for g in opt.param_groups for p in g["params"]]
        state = {i: optim_state_dict["state"][names[id(p)]] for i, p in enumerate(params) if names[id(p)] in optim_state_dict["state"]}
        groups, off = [], 0
        for g in optim_state_dict["param_groups"][:len(opt.param_groups)]:
            groups.append({**{k: v for k, v in g.items() if k != "params"}, "params": list(range(off, off + len(g["params"])))})
            off += len(g["params"])
        opt.load_state_dict({"state": state, "param_groups": groups})


def set_state_dict(model, optimizers, *, model_state_dict, optim_state_dict, options=None):
    r = set_model_state_dict(model, model_state_dict, options=options)
    set_optimizer_state_dict(model, optimizers, optim_state_dict, options=options)
    return r
