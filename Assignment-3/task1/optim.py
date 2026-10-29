"""Exact AdamW updates with model gradients on GPU 0 and optimizer on GPU 1."""
import torch


class SecondaryGPUAdamW(torch.optim.Optimizer):
    """Keep FP32 AdamW master weights and moments on the second CUDA device.

    GradScaler and gradient clipping see the original model parameters on cuda:0.
    Their unscaled gradients are copied to mirrors on cuda:1, then PyTorch AdamW
    performs the ordinary update there. Updated FP32 weights are copied back.
    The master weights are reconstructed from the model checkpoint on resume.
    """

    def __init__(self, groups, *, lr, betas, eps, device="cuda:1"):
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps))
        self.optimizer_device = torch.device(device)
        mirror_groups = []
        self.pairs = []
        for group in self.param_groups:
            mirrors = []
            for parameter in group["params"]:
                if parameter.dtype != torch.float32:
                    raise ValueError("SecondaryGPUAdamW requires FP32 model parameters")
                mirror = torch.nn.Parameter(parameter.detach().to(self.optimizer_device, copy=True))
                mirrors.append(mirror)
                self.pairs.append((parameter, mirror))
            mirror_groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": mirrors})
        self.mirror_optimizer = torch.optim.AdamW(mirror_groups, lr=lr, betas=betas,
                                                   eps=eps, foreach=False)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Closure is unsupported for secondary-GPU AdamW")
        for source, mirror in zip(self.param_groups, self.mirror_optimizer.param_groups):
            for name in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"):
                if name in source:
                    mirror[name] = source[name]
        for original, mirror in self.pairs:
            mirror.grad = None if original.grad is None else original.grad.detach().to(self.optimizer_device)
            original.grad = None  # Gradients are no longer needed on the model GPU.
        self.mirror_optimizer.step()
        for original, mirror in self.pairs:
            if mirror.grad is not None:
                original.copy_(mirror)
            mirror.grad = None

    def state_dict(self):
        outer = super().state_dict()
        outer["mirror_optimizer"] = self.mirror_optimizer.state_dict()
        return outer

    def load_state_dict(self, state_dict):
        if "mirror_optimizer" in state_dict:
            super().load_state_dict({k: v for k, v in state_dict.items() if k != "mirror_optimizer"})
            self.mirror_optimizer.load_state_dict(state_dict["mirror_optimizer"])
        else:
            # Existing native AdamW checkpoints remain recoverable after offload.
            super().load_state_dict({"state": {}, "param_groups": state_dict["param_groups"]})
            self.mirror_optimizer.load_state_dict(state_dict)
        with torch.no_grad():
            for original, mirror in self.pairs:
                mirror.copy_(original)
                mirror.grad = None


def optimizer_state_count(state):
    return len(state["mirror_optimizer"]["state"] if "mirror_optimizer" in state else state["state"])


def build_optimizer(config, method, model, groups):
    lr = config["full_lr"] if method in ("full", "l2sp") else config["adapter_lr"]
    kwargs = dict(lr=lr, betas=tuple(config["adam_betas"]), eps=config["adam_eps"])
    if method in ("full", "l2sp") and torch.cuda.device_count() >= 2:
        return SecondaryGPUAdamW(groups, device="cuda:1", **kwargs), "secondary_gpu_adamw"
    return torch.optim.AdamW(groups, foreach=False, **kwargs), "single_gpu_adamw"
