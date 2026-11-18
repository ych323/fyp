from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_config(path="configs/pilot.yaml"):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    config = yaml.safe_load(p.read_text(encoding="utf-8"))
    if config["precision"] != "fp16" or config["num_workers"] != 0:
        raise ValueError("This Windows pilot currently supports precision=fp16 and num_workers=0.")
    if config["source_language"] != "en" or config["target_language"] != "zh" or config["num_labels"] != 5:
        raise ValueError("This study implements English-to-Chinese, five-star classification.")
    if config["selection_samples"] + config["calibration_samples"] != 5000:
        raise ValueError("Selection and calibration must partition the original 5,000 English validation examples.")
    return config


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def experiment_dir(config):
    return ROOT / "results" / config["study"]


def processed_dir(config):
    return ROOT / "data" / "processed" / config["study"]


def model_path(name):
    return ROOT / "models" / "pretrained" / ("mbert" if name == "mbert" else "xlmr")
