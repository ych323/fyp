import csv
import json

import numpy as np

from .common import ROOT, experiment_dir, fingerprint, processed_dir, read_jsonl, save_json
from .metrics import classification_metrics


def verify(config):
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    audit = json.loads((ROOT / "data" / "provenance" / (config["study"] + "_audit.json")).read_text(encoding="utf-8"))
    sets = []
    for name, item in audit["processed_splits"].items():
        path = processed_dir(config) / (name + ".jsonl")
        rows = read_jsonl(path)
        check(name + " data checksum", fingerprint(path) == item["sha256"])
        check(name + " count", len(rows) == item["count"])
        ids = {r["review_id"] for r in rows}
        check(name + " no duplicate IDs", len(ids) == len(rows))
        check(name + " disjoint from other processed splits", all(not ids & s for s in sets))
        sets.append(ids)
        if name == "zh_unlabeled":
            check("target labels removed", all("label" not in r and "stars" not in r for r in rows))
    source_hashes = set()
    for name in ("mbert", "xlmr", "xlmr_dann"):
        folder = experiment_dir(config) / name
        meta = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        check(name + " training completed", meta["status"] == "trained")
        check(name + " configuration retained", meta["config"] == config)
        source_hashes.add(meta["source_data_sha256"])
        weights = ROOT / "models" / "finetuned" / config["study"] / name / "best.safetensors"
        check(name + " checkpoint checksum", fingerprint(weights) == meta["checkpoint_sha256"])
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        check(name + " no target calibration labels", not report["calibration"]["uses_target_labels"])
        for split in ("en_test", "zh_test"):
            z = np.load(folder / (split + "_logits.npz"), allow_pickle=False)
            for stage, temperature in [("before", 1.), ("after", report["calibration"]["temperature"])]:
                calculated = classification_metrics(z["logits"], z["labels"], temperature, config["ece_bins"])
                for metric in ("accuracy", "macro_f1", "mae", "ece", "nll", "brier"):
                    check(f"{name} {split} {stage} {metric} recomputes", abs(calculated[metric] - report["splits"][split][stage][metric]) < 1e-10)
            check(name + " " + split + " full test count", len(z["labels"]) == 5000)
            check(name + " " + split + " predictions unchanged by temperature",
                  report["splits"][split]["before"]["accuracy"] == report["splits"][split]["after"]["accuracy"])
            rows = list(csv.DictReader(open(folder / (split + "_predictions.csv"), encoding="utf-8-sig", newline="")))
            check(name + " " + split + " exported prediction count", len(rows) == 5000)
    check("all models used the same source sample", len(source_hashes) == 1)
    save_json(ROOT / "reports" / config["study"] / "verification.json", {"status": "passed", "checks_passed": len(checks), "checks": checks})
    print(f"Verification passed: {len(checks)} checks.", flush=True)
