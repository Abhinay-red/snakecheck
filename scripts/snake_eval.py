"""
snake_eval.py - evaluation for SnakeCheck. Plain numpy, so it runs anywhere.

The model outputs one probability per class. The app's real question is
"venomous or not", so we also collapse the classes:

    p_venomous = sum of the probabilities of the venomous classes

A photo is called venomous when p_venomous >= threshold. Then the species is
the most likely venomous class; otherwise it is the most likely of
non_venomous / not_a_snake.

The threshold is chosen on the VALIDATION set only: the highest threshold that
still catches at least `target_recall` of venomous snakes. That keeps missed
venomous snakes rare (the dangerous mistake) while calling as few harmless
snakes venomous as possible. The test sets are then scored with that fixed
threshold, never tuned on.
"""

from __future__ import annotations

import numpy as np


def venomous_mask(class_names, venomous_classes) -> np.ndarray:
    return np.array([c in set(venomous_classes) for c in class_names])


def p_venomous(probs: np.ndarray, vmask: np.ndarray) -> np.ndarray:
    return probs[:, vmask].sum(axis=1)


def predict(probs: np.ndarray, vmask: np.ndarray, threshold: float) -> np.ndarray:
    """Class index per image using the collapsed venomous threshold."""
    pv = p_venomous(probs, vmask)
    ven = np.where(vmask, probs, -1).argmax(axis=1)
    non = np.where(~vmask, probs, -1).argmax(axis=1)
    return np.where(pv >= threshold, ven, non)


def snake_rows(y_true: np.ndarray, class_names, snake_classes) -> np.ndarray:
    """True for images whose true label is a snake (venomous or not)."""
    idx = [class_names.index(c) for c in snake_classes if c in class_names]
    return np.isin(y_true, idx)


def pick_threshold(y_true, probs, class_names, venomous_classes, target_recall=0.95) -> float:
    """Highest threshold whose venomous recall on these images is >= target_recall."""
    vmask = venomous_mask(class_names, venomous_classes)
    is_ven = vmask[y_true]
    pv = p_venomous(probs, vmask)
    if not is_ven.any():
        return 0.5
    # Recall at threshold t is the share of venomous images with pv >= t.
    # The best t is the k-th largest venomous score, k = ceil(target * n).
    scores = np.sort(pv[is_ven])[::-1]
    k = int(np.ceil(target_recall * len(scores)))
    return float(scores[max(k, 1) - 1])


def binary_auc(is_pos: np.ndarray, score: np.ndarray) -> float:
    pos, neg = score[is_pos], score[~is_pos]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(y_true, probs, class_names, venomous_classes, threshold, snake_classes=None) -> dict:
    """Per-class recall/precision, a confusion matrix, and the venomous vs not view."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    vmask = venomous_mask(class_names, venomous_classes)
    y_pred = predict(probs, vmask, threshold)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for i, c in enumerate(class_names):
        support = int(cm[i].sum())
        tp = int(cm[i, i])
        predicted = int(cm[:, i].sum())
        per_class[c] = {
            "support": support,
            "recall": tp / support if support else float("nan"),
            "precision": tp / predicted if predicted else float("nan"),
        }
    f1s = [2 * m["recall"] * m["precision"] / (m["recall"] + m["precision"])
           for m in per_class.values()
           if m["support"] and m["recall"] + m["precision"] > 0]

    snake_classes = snake_classes or [c for c in class_names if c != "not_a_snake"]
    snakes = snake_rows(y_true, class_names, snake_classes)
    is_ven = vmask[y_true]
    pred_ven = vmask[y_pred]
    pv = p_venomous(probs, vmask)
    s_ven, s_pred = is_ven[snakes], pred_ven[snakes]
    binary = {
        "venomous_images": int(s_ven.sum()),
        "harmless_snake_images": int((~s_ven).sum()),
        "venomous_recall": float((s_pred & s_ven).sum() / s_ven.sum()) if s_ven.any() else float("nan"),
        "harmless_recall": float((~s_pred & ~s_ven).sum() / (~s_ven).sum()) if (~s_ven).any() else float("nan"),
        "missed_venomous": int((~s_pred & s_ven).sum()),
        "harmless_called_venomous": int((s_pred & ~s_ven).sum()),
        "auc": binary_auc(s_ven, pv[snakes]),
    }
    return {
        "threshold": float(threshold),
        "images": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class": per_class,
        "venomous_vs_not": binary,
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }


def gate_check(metrics: dict, venomous_classes, gates: dict) -> tuple[bool, list[str]]:
    """Launch gates. Classes with no images in this split are reported, not passed."""
    lines, ok = [], True

    def check(name, value, gate):
        nonlocal ok
        if value != value:  # nan
            lines.append(f"  {name:<28} no images            [SKIP]")
            return
        passed = value >= gate
        ok &= passed
        lines.append(f"  {name:<28} {value:.3f}  gate >= {gate:.2f}  [{'PASS' if passed else 'FAIL'}]")

    for c in venomous_classes:
        check(f"{c} recall", metrics["per_class"][c]["recall"], gates["venomous_class_recall"])
    check("non_venomous recall", metrics["per_class"]["non_venomous"]["recall"], gates["non_venomous_recall"])
    b = metrics["venomous_vs_not"]
    check("venomous vs not: recall", b["venomous_recall"], gates["binary_venomous_recall"])
    check("venomous vs not: AUC", b["auc"], gates["binary_auc"])
    check("macro F1", metrics["macro_f1"], gates["macro_f1"])
    return ok, lines


def format_report(name: str, m: dict) -> str:
    out = [f"== {name}: {m['images']} images, threshold {m['threshold']:.3f} ==",
           f"{'class':<18}{'n':>6}{'recall':>9}{'precision':>11}"]
    for c, v in m["per_class"].items():
        out.append(f"{c:<18}{v['support']:>6}{v['recall']:>9.3f}{v['precision']:>11.3f}")
    b = m["venomous_vs_not"]
    out += [f"accuracy {m['accuracy']:.3f}   macro F1 {m['macro_f1']:.3f}",
            f"venomous vs not (snakes only): venomous recall {b['venomous_recall']:.3f}, "
            f"harmless recall {b['harmless_recall']:.3f}, AUC {b['auc']:.3f}",
            f"  missed venomous: {b['missed_venomous']} of {b['venomous_images']}; "
            f"harmless called venomous: {b['harmless_called_venomous']} of {b['harmless_snake_images']}"]
    return "\n".join(out)
