import csv
import json
import time

import numpy as np
from safetensors.torch import load_file
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
import torch

from .common import ROOT, experiment_dir, fingerprint, model_path, save_json, seed_all
from .data import EncodedReviews
from .metrics import classification_metrics
from .model import SentimentModel
from .train import predict


def fit_temperature(logits, labels):
    logits = logits.astype(np.float64)

    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

    solution = minimize_scalar(objective, bounds=(-4.605170186, 4.605170186), method="bounded",
                               options={"xatol": 1e-8})
    if not solution.success:
        raise RuntimeError("Temperature optimization failed")
    logt = float(solution.x) if solution.fun <= objective(0.0) else 0.0
    return {"temperature": float(np.exp(logt)), "calibration_nll_before": objective(0.0),
            "calibration_nll_after": objective(logt), "fit_split": "en_calibration",
            "uses_target_labels": False, "optimizer": "scipy bounded scalar optimization over log(T)",
            "temperature_bounds": [0.01, 100.0]}


def evaluate(config, name):
    seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = "mbert" if name == "mbert" else "xlmr"
    checkpoint = ROOT / "models" / "finetuned" / config["study"] / name / "best.safetensors"
    out = experiment_dir(config) / name
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["status"] == "trained" and meta["config"] == config
    assert fingerprint(checkpoint) == meta["checkpoint_sha256"]
    model = SentimentModel(model_path(base), 5, dann=name == "xlmr_dann", load_pretrained=False)
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    model.to(device)
    started = time.perf_counter()
    calibration = EncodedReviews(config, base, "en_calibration")
    cal_logits, cal_labels = predict(model, calibration, config, device)
    temperature = fit_temperature(cal_logits, cal_labels)
    np.savez_compressed(out / "en_calibration_logits.npz", logits=cal_logits, labels=cal_labels,
                        review_ids=np.array([r["review_id"] for r in calibration.rows]))
    save_json(out / "temperature.json", temperature)
    results = {"model": name, "protocol": "pilot_single_seed" if config["train_samples"] < 200000 else "adapted_full_data_single_seed",
               "calibration": temperature, "splits": {}}
    for split in ("en_test", "zh_test"):
        data = EncodedReviews(config, base, split)
        logits, labels = predict(model, data, config, device)
        before = classification_metrics(logits, labels, bins=config["ece_bins"])
        after = classification_metrics(logits, labels, temperature["temperature"], bins=config["ece_bins"])
        assert before["accuracy"] == after["accuracy"]
        results["splits"][split] = {"before": before, "after": after}
        np.savez_compressed(out / (split + "_logits.npz"), logits=logits, labels=labels,
                            review_ids=np.array([r["review_id"] for r in data.rows]))
        probs, calibrated = softmax(logits.astype(float), axis=1), softmax(logits.astype(float) / temperature["temperature"], axis=1)
        with open(out / (split + "_predictions.csv"), "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["review_id", "true_stars", "predicted_stars", "confidence_before", "confidence_after"])
            writer.writerows((r["review_id"], int(y) + 1, int(p.argmax()) + 1, float(p.max()), float(q.max()))
                             for r, y, p, q in zip(data.rows, labels, probs, calibrated))
        print(f"{name} {split}: accuracy={before['accuracy']:.4f}, MAE={before['mae']:.4f}, ECE={before['ece']:.4f}->{after['ece']:.4f}", flush=True)
    results["evaluation_seconds"] = time.perf_counter() - started
    save_json(out / "metrics.json", results)
