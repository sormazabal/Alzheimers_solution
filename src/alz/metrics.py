"""Shared binary classification metrics, used by all three models now that
MRI has been collapsed to binary (Non Demented vs Demented) alongside the
already-binary clinical and EEG models."""
import json
import os

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob_pos, threshold: float = 0.5) -> dict:
    """Full clinical metric set for a binary classifier.

    y_prob_pos: predicted probability of the positive (worse-outcome) class.
    Returns accuracy, balanced_accuracy, auroc, auprc, f1, sensitivity,
    specificity, confusion_matrix ({tn, fp, fn, tp}). auroc/auprc are None
    when y_true has only one class (can't be computed).
    """
    y_pred = [1 if p >= threshold else 0 for p in y_prob_pos]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None

    try:
        auroc = roc_auc_score(y_true, y_prob_pos)
        auprc = average_precision_score(y_true, y_prob_pos)
    except ValueError:
        auroc = auprc = None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_metrics(name: str, metrics: dict, out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def demo():
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.6, 0.4, 0.7, 0.8, 0.9]  # 1 false pos, 1 false neg at 0.5
    m = binary_metrics(y_true, y_prob)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 3}
    assert abs(m["sensitivity"] - 3 / 4) < 1e-9
    assert abs(m["specificity"] - 2 / 3) < 1e-9
    assert 0 <= m["auroc"] <= 1


demo()
