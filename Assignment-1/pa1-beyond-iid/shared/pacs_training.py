#! Shared source-side training utilities for PACS Tasks 2 and 3
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from pacs import SOURCE_DOMAINS, freeze_bn_running_stats


def set_epoch_loaders(source_loaders, epoch, seed=6304):
    #! Reset source-shuffle/augmentation seeds at the start of every epoch.
    for index, domain in enumerate(SOURCE_DOMAINS):
        source_loaders[domain].generator.manual_seed(seed + epoch * 1000 + index)


def train_source_epoch(model, source_loaders, optimizer, device, epoch, scaler=None, seed=6304):
    #! A source epoch spans the longest domain loader; cycle shorter domains.
    model.train()
    freeze_bn_running_stats(model)
    set_epoch_loaders(source_loaders, epoch, seed)
    iterators = {domain: iter(source_loaders[domain]) for domain in SOURCE_DOMAINS}
    steps = max(len(source_loaders[domain]) for domain in SOURCE_DOMAINS)
    if not steps:
        raise ValueError("A source loader contains no full batches")

    total_loss, total_correct, total_seen = 0.0, 0, 0
    domain_correct = {domain: 0 for domain in SOURCE_DOMAINS}
    domain_seen = {domain: 0 for domain in SOURCE_DOMAINS}
    amp_enabled = device.type == "cuda" and scaler is not None and scaler.is_enabled()

    for _ in range(steps):
        batches = []
        for domain in SOURCE_DOMAINS:
            try:
                batch = next(iterators[domain])
            except StopIteration:
                iterators[domain] = iter(source_loaders[domain])
                batch = next(iterators[domain])
            batches.append(batch)

        images = torch.cat([batch[0] for batch in batches], dim=0).to(device, non_blocking=True)
        labels = torch.cat([batch[1] for batch in batches], dim=0).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp_enabled else nullcontext()
        with context:
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = logits.detach().argmax(dim=1)
        n = len(labels)
        total_loss += float(loss.detach()) * n
        total_correct += int((predictions == labels).sum().item())
        total_seen += n
        offset = 0
        for domain, (domain_images, _) in zip(SOURCE_DOMAINS, batches):
            count = len(domain_images)
            domain_correct[domain] += int((predictions[offset:offset + count] == labels[offset:offset + count]).sum().item())
            domain_seen[domain] += count
            offset += count

    result = {
        "train_loss": total_loss / total_seen,
        "train_accuracy": total_correct / total_seen,
        "updates": steps,
        "source_examples_processed": total_seen,
    }
    result.update({f"train_{domain}_accuracy": domain_correct[domain] / domain_seen[domain]
                   for domain in SOURCE_DOMAINS})
    return result


@torch.inference_mode()
def evaluate_sources(model, validation_loaders, device, num_classes=7):
    #! All three source validation sets; never access Sketch here.
    model.eval()
    results = {}
    for domain in SOURCE_DOMAINS:
        total_loss, total_seen = 0.0, 0
        targets, predictions = [], []
        for images, labels in validation_loaders[domain]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_seen += len(labels)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
        if not total_seen:
            raise ValueError(f"Validation domain {domain} is empty")
        results[domain] = {
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, labels=list(range(num_classes)),
                                       average="macro", zero_division=0)),
            "loss": total_loss / total_seen,
            "n_images": total_seen,
        }

    results["mean_accuracy"] = float(np.mean([results[d]["accuracy"] for d in SOURCE_DOMAINS]))
    results["mean_macro_f1"] = float(np.mean([results[d]["macro_f1"] for d in SOURCE_DOMAINS]))
    results["worst_source_accuracy"] = float(min(results[d]["accuracy"] for d in SOURCE_DOMAINS))
    results["worst_source_macro_f1"] = float(min(results[d]["macro_f1"] for d in SOURCE_DOMAINS))
    return results


def atomic_torch_save(payload, destination):
    #! Prevent an interrupted Drive write from destroying a valid checkpoint.
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    torch.save(payload, partial)
    os.replace(partial, destination)


def cpu_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
