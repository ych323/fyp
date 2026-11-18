"""Build fixed splits and prove that selection/calibration/test IDs are disjoint."""
import collections
import gzip
import json

import numpy as np
from sklearn.model_selection import train_test_split

from .common import ROOT, fingerprint, processed_dir, save_json, write_jsonl


def load_raw(language, split):
    path = ROOT / "data" / "raw" / "marc" / language / (split + ".jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(row):
    return {"review_id": row["review_id"], "body": row.get("review_body") or "",
            "title": row.get("review_title") or "", "category": row.get("product_category") or "other",
            "language": row["language"], "label": int(row["stars"]) - 1}


def prepare(config):
    out = processed_dir(config)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])
    audit = {"config": config, "raw_splits": {}, "processed_splits": {}}
    raw = {}
    for lang in ("en", "zh"):
        raw[lang] = {}
        for split in ("train", "validation", "test"):
            rows = load_raw(lang, split)
            expected = 200000 if split == "train" else 5000
            assert len(rows) == expected, (lang, split, len(rows))
            assert len({r["review_id"] for r in rows}) == expected
            assert all(r["language"] == lang and 1 <= int(r["stars"]) <= 5 for r in rows)
            counts = dict(sorted(collections.Counter(int(r["stars"]) for r in rows).items()))
            assert len(counts) == 5 and len(set(counts.values())) == 1
            audit["raw_splits"][f"{lang}_{split}"] = {"count": len(rows), "stars": counts,
                "null_fields": {key: sum(r.get(key) is None for r in rows)
                                for key in ("review_body", "review_title", "product_category")}}
            raw[lang][split] = rows
        id_sets = [set(r["review_id"] for r in raw[lang][s]) for s in ("train", "validation", "test")]
        assert not (id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2])
    en = [normalize(r) for r in raw["en"]["train"]]
    n = config["train_samples"]
    assert n % 5 == 0 and n <= len(en)
    ids = []
    for label in range(5):
        candidates = np.array([i for i, r in enumerate(en) if r["label"] == label])
        ids.extend(rng.choice(candidates, size=n // 5, replace=False).tolist())
    rng.shuffle(ids)
    source = [en[i] for i in ids]
    dev = [normalize(r) for r in raw["en"]["validation"]]
    sel, cal = train_test_split(np.arange(len(dev)), test_size=config["calibration_samples"],
                               random_state=config["seed"], stratify=[r["label"] for r in dev])
    assert len(sel) == config["selection_samples"]
    # Target sampling is uniform WITHOUT inspecting labels; training files exclude labels.
    target_indices = rng.choice(len(raw["zh"]["train"]), size=config["target_unlabeled_samples"], replace=False)
    target = []
    for index in target_indices:
        r = raw["zh"]["train"][int(index)]
        target.append({"review_id": r["review_id"], "body": r.get("review_body") or "",
                       "title": r.get("review_title") or "", "category": r.get("product_category") or "other", "language": "zh"})
    splits = {"en_train": source, "zh_unlabeled": target,
              "en_selection": [dev[int(i)] for i in sel],
              "en_calibration": [dev[int(i)] for i in cal],
              "en_test": [normalize(r) for r in raw["en"]["test"]],
              "zh_test": [normalize(r) for r in raw["zh"]["test"]]}
    assert not any("label" in r or "stars" in r for r in target)
    sets = {k: {r["review_id"] for r in v} for k, v in splits.items()}
    keys = list(sets)
    for i, key in enumerate(keys):
        for other in keys[i + 1:]:
            assert not sets[key] & sets[other], (key, other)
    for name, rows in splits.items():
        assert all(isinstance(r[k], str) for r in rows for k in ("body", "title", "category"))
        path = out / (name + ".jsonl")
        write_jsonl(path, rows)
        ids_path = ROOT / "data" / "splits" / config["study"] / (name + "_ids.json")
        save_json(ids_path, [r["review_id"] for r in rows])
        audit["processed_splits"][name] = {"count": len(rows), "sha256": fingerprint(path),
                                          "label_counts": dict(collections.Counter(r["label"] for r in rows)) if "label" in rows[0] else None}
    audit["checks"] = {"counts_verified": True, "all_review_ids_disjoint": True,
                       "target_training_labels_absent": True,
                       "chinese_validation_used": False,
                       "source_sample_shared_across_models": True}
    save_json(ROOT / "data" / "provenance" / (config["study"] + "_audit.json"), audit)
    print("Prepared fixed splits: " + str({k: len(v) for k, v in splits.items()}), flush=True)
