import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn


METHODS = {"full": (None, None), "l2sp": (None, None),
           "lora4": (4, 8), "lora16": (16, 32), "oplora16": (16, 32)}


class AdapterLinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout, uk=None, vk=None):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Adapter target must be nn.Linear")
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.register_buffer("uk", uk, persistent=True)
        self.register_buffer("vk", vk, persistent=True)
        for p in base.parameters():
            p.requires_grad_(False)

    def effective_factors(self):
        b, a = self.B, self.A
        if self.uk is not None:
            b = b - self.uk @ (self.uk.T @ b)
            a = a - (a @ self.vk) @ self.vk.T
        return b, a

    def correction(self):
        b, a = self.effective_factors()
        return (self.alpha / self.rank) * (b @ a)

    def forward(self, x):
        original = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            b, a = self.effective_factors()
            correction = torch.nn.functional.linear(
                torch.nn.functional.linear(self.dropout(x.float()), a), b)
            correction = correction * (self.alpha / self.rank)
        return original + correction.to(original.dtype)


def target_modules(model, targets):
    result = []
    for layer_i, layer in enumerate(model.model.layers):
        for name in targets:
            parent = layer.self_attn if hasattr(layer.self_attn, name) else layer.mlp
            if not hasattr(parent, name):
                raise ValueError(f"Missing layer {layer_i} target {name}")
            result.append((layer_i, name, parent, getattr(parent, name)))
    return result


def leading_bases(weight, rank):
    w = weight.detach().float()
    if rank > min(w.shape):
        raise ValueError(f"Protection rank {rank} exceeds shape {tuple(w.shape)}")
    u, _, vh = torch.linalg.svd(w, full_matrices=False)
    return u[:, :rank].contiguous(), vh[:rank, :].T.contiguous()


def load_model(config, method, *, for_inference=False, skip_basis=False):
    from transformers import AutoModelForCausalLM
    if method not in METHODS:
        raise ValueError(method)
    dtype = torch.float16 if method.startswith("lora") or for_inference or skip_basis else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], dtype=dtype, cache_dir=config["cache_dir"],
        revision=config.get("model_revision"), low_cpu_mem_usage=True,
        attn_implementation="sdpa").to("cuda")
    model.config.use_cache = False
    if not for_inference and config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    metadata = {"basis_seconds": 0.0, "basis_bytes": 0}
    rank, alpha = METHODS[method]
    if rank is not None:
        targets = target_modules(model, config["targets"])
        for p in model.parameters():
            p.requires_grad_(False)
        bases = []
        if method == "oplora16":
            start = time.perf_counter()
            for target_index, (_, _, _, module) in enumerate(targets, 1):
                if for_inference or skip_basis:
                    uk = torch.zeros((module.out_features, config["op_rank"]), dtype=torch.float32, device="cuda")
                    vk = torch.zeros((module.in_features, config["op_rank"]), dtype=torch.float32, device="cuda")
                else:
                    uk, vk = leading_bases(module.weight, config["op_rank"])
                metadata["basis_bytes"] += (uk.numel() + vk.numel()) * 4
                bases.append((uk, vk))
                if not for_inference and not skip_basis and target_index % len(config["targets"]) == 0:
                    print(f"OPLoRA SVD: {target_index}/{len(targets)} matrices, "
                          f"{time.perf_counter()-start:.1f} s", flush=True)
            metadata["basis_seconds"] = time.perf_counter() - start
            if not for_inference and not skip_basis:
                model.half()  # FP32 SVD of original weights, then FP16 frozen backbone.
        else:
            bases = [(None, None)] * len(targets)
        # SVD preprocessing must not change adapter initialization.
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        for (_, _, parent, module), (uk, vk) in zip(targets, bases):
            name = next(n for n in config["targets"] if getattr(parent, n, None) is module)
            setattr(parent, name, AdapterLinear(module, rank, alpha, config["lora_dropout"], uk, vk))
    return model, metadata


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def adapter_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith((".A", ".B", ".uk", ".vk"))}


def projected_adapter_state(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, AdapterLinear):
            b, a = module.effective_factors()
            state[name + ".A"] = a.detach().cpu()
            state[name + ".B"] = b.detach().cpu()
    return state


def use_projected_factors(model, state):
    expected = set()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, AdapterLinear):
                ak, bk = name + ".A", name + ".B"
                expected.update((ak, bk))
                module.A.copy_(state[ak].to(module.A.device))
                module.B.copy_(state[bk].to(module.B.device))
                if module.uk is not None:
                    module.uk.zero_()
                    module.vk.zero_()
    if set(state) != expected:
        raise ValueError("Projected factor keys mismatch")


def load_adapter_state(model, state):
    required = {k for k in model.state_dict() if k.endswith((".A", ".B", ".uk", ".vk"))}
    if set(state) != required:
        raise ValueError(f"Adapter keys mismatch: missing={required-set(state)}, extra={set(state)-required}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = set(state)
    if unexpected or any(k in expected for k in missing):
        raise ValueError(f"Adapter state mismatch: {unexpected}")


def merge_adapters(model):
    for _, name, parent, module in target_modules(model, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
        if isinstance(module, AdapterLinear):
            with torch.no_grad():
                merged = nn.Linear(module.base.in_features, module.base.out_features,
                                   bias=module.base.bias is not None, device=module.base.weight.device,
                                   dtype=module.base.weight.dtype)
                merged.weight.copy_(module.base.weight + module.correction().to(module.base.weight.dtype))
                if module.base.bias is not None:
                    merged.bias.copy_(module.base.bias)
                setattr(parent, name, merged)
    return model


def assert_merge_equivalent(unmerged, merged, input_ids, atol=0.1):
    unmerged.eval(); merged.eval()
    with torch.inference_mode():
        a = unmerged(input_ids=input_ids).logits.float()
        b = merged(input_ids=input_ids).logits.float()
    error = (a - b).abs().max().item()
    if error > atol:
        raise AssertionError(f"Merge logits max error {error:.6f} > {atol}")
    return error
