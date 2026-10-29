import time
import torch


def unique_named_parameters(model):
    seen = set()
    for name, p in model.named_parameters():
        if id(p) not in seen:
            seen.add(id(p))
            yield name, p


class CPUReference:
    """Exact FP32 L2-SP reference; stream one tensor at a time to bound GPU memory."""
    def __init__(self, model):
        self.weights = {name: (p.detach().cpu().clone().pin_memory() if torch.cuda.is_available()
                               else p.detach().cpu().clone())
                        for name, p in unique_named_parameters(model)}
        self.bytes = sum(w.numel() * w.element_size() for w in self.weights.values())
        self.transfer_seconds = 0.0

    @torch.no_grad()
    def omega_and_gradient(self, model, coefficient=0.0):
        omega = 0.0
        start = time.perf_counter()
        for name, p in unique_named_parameters(model):
            ref = self.weights[name].to(p.device, non_blocking=False)
            diff = p.detach().float() - ref
            omega += 0.5 * torch.sum(diff.square()).item()
            if coefficient:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(diff, alpha=coefficient)
            del ref, diff
        self.transfer_seconds += time.perf_counter() - start
        return omega
