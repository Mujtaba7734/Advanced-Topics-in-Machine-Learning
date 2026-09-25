#! Task 4 shared V1: isolated from Tasks 1-3 and stable for GCSC/PROSER
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_resnet18(num_classes=10):
    #! Random initialization; CIFAR stem does not use the ImageNet 7x7 stem or max-pool.
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def forward_features_logits(model, images):
    #! Exact ResNet-18 forward with accessible 512-dimensional pooled representation.
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


def training_transform(gcsc=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if gcsc:
        #! The sole augmentation difference in GCSC, before ToTensor and Normalize.
        ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops.extend([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    return transforms.Compose(ops)


def evaluation_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def extract_known_outputs(model, loader, device):
    #! Sequential, unaugmented known-only loader; index order comes from its Subset.
    model.eval()
    features, logits, labels = [], [], []
    with torch.inference_mode():
        for images, targets in loader:
            f, z = forward_features_logits(model, images.to(device, non_blocking=True))
            features.append(f.float().cpu().numpy())
            logits.append(z.float().cpu().numpy())
            labels.append(targets.numpy())
    return (np.concatenate(features), np.concatenate(logits), np.concatenate(labels))
