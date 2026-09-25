#! Shared adaptation loop for PACS: reused by DAN, DANN and CDAN
from contextlib import nullcontext

import torch

from pacs import SOURCE_DOMAINS, freeze_bn_running_stats
from pacs_training import set_epoch_loaders


def forward_features_logits(model, images):
    #! Exactly torchvision ResNet-18's forward, exposing its 512-D pooled features.
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    features = torch.flatten(model.avgpool(x), 1)
    return features, model.fc(features)


def _take_next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_adaptation_epoch(model, source_loaders, target_loader, optimizer, device,
                           epoch, objective, scaler=None, seed=6304, extra_module=None):
    #! Identical source batching and source-epoch definition to Source-only ERM.
  
    model.train()
    freeze_bn_running_stats(model)
  
    if extra_module is not None:
        extra_module.train()
  
    set_epoch_loaders(source_loaders, epoch, seed)
    target_loader.generator.manual_seed(seed + epoch * 1000 + len(SOURCE_DOMAINS))

    source_iters = {domain: iter(source_loaders[domain]) for domain in SOURCE_DOMAINS}
    target_iter = iter(target_loader)
    steps = max(len(source_loaders[domain]) for domain in SOURCE_DOMAINS)
  
    if steps == 0 or len(target_loader) == 0:
        raise ValueError('Training requires nonempty full source and target batches')

    sums = {'train_total_loss': 0., 'train_cls_loss': 0., 'train_alignment_loss': 0.}
  
    correct, count = 0, 0
    per_domain_correct = {d: 0 for d in SOURCE_DOMAINS}
    per_domain_count = {d: 0 for d in SOURCE_DOMAINS}
    amp_enabled = device.type == 'cuda' and scaler is not None and scaler.is_enabled()

    for step in range(steps):
        batches = []
        for domain in SOURCE_DOMAINS:
            batch, source_iters[domain] = _take_next(source_iters[domain], source_loaders[domain])
            batches.append(batch)
        target_batch, target_iter = _take_next(target_iter, target_loader)
  
        #! The target loader returns images only; it contains no class labels.
        if not isinstance(target_batch, torch.Tensor):
            raise TypeError('Unlabeled Sketch loader must return image tensors only')

        source_images = torch.cat([batch[0] for batch in batches], dim=0).to(device, non_blocking=True)
        labels = torch.cat([batch[1] for batch in batches], dim=0).to(device, non_blocking=True)
        target_images = target_batch.to(device, non_blocking=True)
       
        if len(source_images) != 24 or len(target_images) != 24:
            raise ValueError('Expected 8 samples from each source and 24 unlabeled target samples')
       
        images = torch.cat((source_images, target_images), dim=0)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast('cuda', dtype=torch.float16) if amp_enabled else nullcontext()
       
        with context:
            features, logits = forward_features_logits(model, images)
            total_loss, components = objective(features[:24], logits[:24], labels,
                                               features[24:], logits[24:], step, steps)

        if not bool(torch.isfinite(total_loss).item()):
            raise FloatingPointError('Non-finite adaptation loss; inspect this batch and the MMD kernel')
       
        if amp_enabled:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        sums['train_total_loss'] += float(total_loss.detach())
        sums['train_cls_loss'] += float(components['classification'].detach())
        sums['train_alignment_loss'] += float(components['alignment'].detach())
        preds = logits[:24].detach().argmax(1)
        correct += int((preds == labels).sum().item())
        count += len(labels)
       
        for index, domain in enumerate(SOURCE_DOMAINS):
            start = index * 8
            per_domain_correct[domain] += int((preds[start:start+8] == labels[start:start+8]).sum().item())
            per_domain_count[domain] += 8

    result = {name: value / steps for name, value in sums.items()}
    result.update({'train_accuracy': correct / count, 'updates': steps,
                   'source_examples_processed': count, 'target_examples_processed': 24 * steps})
    result.update({f'train_{domain}_accuracy': per_domain_correct[domain] / per_domain_count[domain]
                   for domain in SOURCE_DOMAINS})
    return result
