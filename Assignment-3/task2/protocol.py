"""Immutable scientific protocol; integer updates count successful optimizer steps."""
from dataclasses import dataclass, asdict
import math
import hashlib
import json
import numpy as np

@dataclass(frozen=True)
class Protocol:
    seed: int = 6304
    vision: str = 'google/siglip-base-patch16-224'
    language: str = 'Qwen/Qwen3-0.6B'
    vision_revision: str = '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'
    language_revision: str = 'c1899de289a04d12100db370d81485cdf75e47ca'
    visual_tokens: int = 196
    microbatch: int = 8
    accumulation: int = 2
    updates: tuple = (2250, 1500, 750)
    connector_lr: tuple = (1e-3, 1e-4, 5e-5)
    lora_lr: tuple = (0., 2e-5, 1e-5)
    question_cap: int = 80
    caption_cap: int = 64
    answer_cap: int = 16
    bootstrap: int = 4000
    weight_decay: float = .01
    clip: float = 1.
    rank: int = 16
    alpha: int = 32
    dropout: float = .05

P = Protocol()
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def stage_at(update):
    if not 0 <= update < 4500:
        raise ValueError('Update outside main training budget')
    return 0 if update < 2250 else 1 if update < 3750 else 2

def lr_factor(step, total):
    # ceil avoids an undocumented fractional warmup endpoint (113,75,38).
    warmup = math.ceil(.05 * total)
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / (total - warmup)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))

def schedule(data):
    """Precommit exact example IDs; two independently shuffled caption passes.

    Each later draw contributes a whole pair; probabilities are not quotas.
    Persisting this small plan makes mid-stage recovery independent of workers.
    """
    rng = np.random.default_rng(P.seed)
    captions = data['caption_train']
    if len(captions) != 18000:
        raise ValueError('Stage 1 requires 18,000 caption examples')
    plan = []
    for _ in range(2):
        ids = rng.permutation(len(captions))
        plan.extend([[captions[i] for i in ids[j:j+16]] for j in range(0, len(ids), 16)])
    for updates, names, probs in [(1500, ['presence','counting','caption_train'], [.5,.25,.25]),
                                   (750, ['natural_train','presence','counting','caption_train'], [.7,.1,.1,.1])]:
        for _ in range(updates):
            batch = []
            for _ in range(8):
                name = str(rng.choice(names, p=probs))
                pool = data[name]
                if name == 'caption_train':
                    batch.extend(pool[int(i)] for i in rng.integers(len(pool), size=2))
                else:
                    batch.extend(pool[int(rng.integers(len(pool)))])
            plan.append(batch)
    assert len(plan) == 4500 and sum(map(len,plan)) == 72000
    return plan
