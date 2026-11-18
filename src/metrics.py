import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error


def classification_metrics(logits, labels, temperature=1.0, bins=15):
    probs = softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)
    labels = np.asarray(labels, dtype=np.int64)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == labels).astype(float)
    reliability, ece = [], 0.0
    for i in range(bins):
        lower, upper = i / bins, (i + 1) / bins
        mask = (conf >= lower) & ((conf < upper) if i + 1 < bins else (conf <= upper))
        count = int(mask.sum())
        acc, avg_conf = (float(correct[mask].mean()), float(conf[mask].mean())) if count else (0.0, 0.0)
        ece += count / len(labels) * abs(acc - avg_conf)
        reliability.append({"lower": lower, "upper": upper, "count": count, "accuracy": acc, "confidence": avg_conf})
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=np.arange(5), average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(labels, pred)),
            "ece": float(ece),
            "nll": float(-np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-300)).mean()),
            # Multiclass Brier: mean over examples of SUM over the five classes.
            "brier": float(np.square(probs - np.eye(5)[labels]).sum(axis=1).mean()),
            "temperature": float(temperature),
            "confusion_matrix": confusion_matrix(labels, pred, labels=np.arange(5)).tolist(),
            "reliability_bins": reliability}
