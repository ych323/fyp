import hashlib
import json

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .common import ROOT, fingerprint, model_path, processed_dir, read_jsonl, save_json


class EncodedReviews(Dataset):
    def __init__(self, config, model_name, split):
        raw_path = processed_dir(config) / (split + ".jsonl")
        self.rows = read_jsonl(raw_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path(model_name), local_files_only=True, use_fast=True)
        source_hash = fingerprint(raw_path)
        signature = {"source_sha256": source_hash, "max_length": config["max_length"],
                     "body_max_tokens": config["body_max_tokens"], "input_fields": config["input_fields"],
                     "model": model_name, "tokenizer_sha256": fingerprint(model_path(model_name) / "tokenizer.json")}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        cache = ROOT / "data" / "tokenized" / config["study"] / model_name / f"{split}_{key}.pt"
        if cache.exists():
            self.encodings = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            bodies = [r["body"] for r in self.rows]
            tokens = tokenizer(bodies, add_special_tokens=False, truncation=True,
                               max_length=config["body_max_tokens"], return_attention_mask=False)["input_ids"]
            if config["input_fields"] == "body_title_category":
                pairs = tokenizer([r["title"] + " " + r["category"] for r in self.rows],
                                  add_special_tokens=False, truncation=True,
                                  max_length=config["max_length"] - 4, return_attention_mask=False)["input_ids"]
            else:
                pairs = [None] * len(tokens)
            encoded = [tokenizer.prepare_for_model(body, pair_ids=pair, max_length=config["max_length"],
                       truncation="longest_first", padding="max_length", return_attention_mask=True, verbose=False)
                       for body, pair in zip(tokens, pairs)]
            self.encodings = {k: torch.tensor([r[k] for r in encoded], dtype=torch.long)
                              for k in ("input_ids", "attention_mask", "token_type_ids") if k in encoded[0]}
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.encodings, cache)
            save_json(cache.with_suffix(".json"), signature)
        self.labeled = "label" in self.rows[0]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = {k: v[index] for k, v in self.encodings.items()}
        if self.labeled:
            row["labels"] = torch.tensor(self.rows[index]["label"], dtype=torch.long)
        return row
