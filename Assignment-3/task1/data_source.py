from pathlib import Path


def load_split(source, split, cache_dir, config_name=None):
    from datasets import DatasetDict, load_dataset, load_from_disk
    path = Path(source)
    if path.exists():
        data = load_from_disk(str(path))
        if isinstance(data, DatasetDict):
            if split not in data:
                raise KeyError(f"Local dataset {path} has no {split} split")
            return data[split]
        if split != "train":
            raise ValueError(f"Local Dataset {path} has no named {split} split")
        return data
    return load_dataset(source, config_name, split=split, cache_dir=cache_dir)
