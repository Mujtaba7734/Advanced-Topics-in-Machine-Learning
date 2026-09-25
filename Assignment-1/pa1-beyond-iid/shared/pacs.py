#! Shared PACS protocol for Tasks 2 and 3

import random
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 6304
SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_safe(archive_path, destination):
    root = Path(destination).resolve()

    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            output = (root / member.name).resolve()

            if not output.is_relative_to(root) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe archive member: {member.name}")

        archive.extractall(root)


def prepare_local_data(storage_root, local_root, include_target=False):
    storage_root, local_root = Path(storage_root), Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    source_file = local_root / "images" / "photo"

    if not source_file.is_dir() or not any(source_file.iterdir()):
        _extract_safe(storage_root / "pacs_sources.tar", local_root)

    if include_target:
        target_file = local_root / "images" / TARGET_DOMAIN
        if not target_file.is_dir() or not any(target_file.iterdir()):
            _extract_safe(storage_root / "pacs_target_unlabeled.tar", local_root)
    return local_root


def make_transform(train=False):
    stats = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    geometry = [transforms.Resize((256, 256))]

    if train:
        geometry += [transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
    else:
        geometry += [transforms.CenterCrop(224)]
    return transforms.Compose(geometry + [
        transforms.ToTensor(), transforms.Normalize(mean=stats.mean, std=stats.std)
    ])


class PACSDataset(Dataset):
    def __init__(self, manifest, image_root, transform, domain=None, unlabeled=False):
        self.records = pd.read_csv(manifest)

        if domain is not None:
            self.records = self.records[self.records["domain"] == domain]

        self.records = self.records.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.unlabeled = unlabeled

        if unlabeled and ("label" in self.records or "label_name" in self.records):
            raise ValueError("Unlabeled target manifest contains labels")

        if not unlabeled and "label" not in self.records:
            raise ValueError("Source manifest is missing labels")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        path = self.image_root / row["relative_path"]

        with Image.open(path) as image:
            rgb = image.convert("RGB")
        tensor = self.transform(rgb)

        if self.unlabeled:
            return tensor

        return tensor, int(row["label"])


def make_loaders(storage_root, local_root, include_target=False, workers=2):
    storage_root = Path(storage_root)
    image_root = prepare_local_data(storage_root, local_root, include_target=include_target)
    source = {}
    validation = {}

    for i, domain in enumerate(SOURCE_DOMAINS):
        train_ds = PACSDataset(storage_root / "source_train.csv", image_root, make_transform(True), domain=domain)
        val_ds = PACSDataset(storage_root / "source_val.csv", image_root, make_transform(False), domain=domain)
        gen = torch.Generator().manual_seed(SEED + i)

        source[domain] = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True,
                                    num_workers=workers, pin_memory=torch.cuda.is_available(), generator=gen)
        validation[domain] = DataLoader(val_ds, batch_size=64, shuffle=False,
                                        num_workers=workers, pin_memory=torch.cuda.is_available())

    if not include_target:
        return source, validation

    target_ds = PACSDataset(storage_root / "target_unlabeled.csv", image_root,
                            make_transform(True), unlabeled=True)

    target_gen = torch.Generator().manual_seed(SEED + 100)
    target_loader = DataLoader(target_ds, batch_size=24, shuffle=True, drop_last=True,
                               num_workers=workers, pin_memory=torch.cuda.is_available(), generator=target_gen)

    return source, validation, target_loader


def forever(loader):
    while True:
        yield from loader


def build_resnet18(num_classes=7):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone


def freeze_bn_running_stats(model):
    #! Call AFTER model.train(); preserve trainable BatchNorm affine parameters.
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
