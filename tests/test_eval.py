"""Tests for the evaluation helpers (numpy only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import snake_eval as se  # noqa: E402

CLASSES = ["non_venomous", "cobra", "russells_viper", "common_krait", "sawscaled_viper", "not_a_snake"]
VEN = ["cobra", "russells_viper", "common_krait", "sawscaled_viper"]


def test_collapsed_prediction_sums_venomous_probabilities():
    # No single venomous class beats non_venomous, but together they do.
    p = np.array([[0.40, 0.20, 0.20, 0.10, 0.10, 0.0]])
    vm = se.venomous_mask(CLASSES, VEN)
    assert se.predict(p, vm, 0.5)[0] == 1          # called venomous, species = cobra
    assert se.predict(p, vm, 0.7)[0] == 0          # stricter threshold -> non_venomous


def test_threshold_meets_recall_target_on_val():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 600)
    p = rng.dirichlet(np.ones(6), 600)
    p[np.isin(y, [1, 2, 3, 4]), 1] += 0.5         # venomous images lean venomous
    p /= p.sum(1, keepdims=True)
    t = se.pick_threshold(y, p, CLASSES, VEN, 0.95)
    m = se.evaluate(y, p, CLASSES, VEN, t)
    assert m["venomous_vs_not"]["venomous_recall"] >= 0.95
    # a slightly higher threshold would drop below the target
    m2 = se.evaluate(y, p, CLASSES, VEN, t + 1e-6)
    assert m2["venomous_vs_not"]["venomous_recall"] < 0.95


def test_not_a_snake_is_left_out_of_the_venomous_view():
    y = np.array([5, 5, 1, 0])
    p = np.eye(6)[[5, 1, 1, 0]]
    m = se.evaluate(y, p, CLASSES, VEN, 0.5)
    b = m["venomous_vs_not"]
    assert b["venomous_images"] == 1 and b["harmless_snake_images"] == 1
    assert m["per_class"]["not_a_snake"]["recall"] == 0.5


def test_auc_matches_simple_definition():
    pos, neg = np.array([0.9, 0.8, 0.3]), np.array([0.1, 0.4, 0.3])
    is_pos = np.array([1, 1, 1, 0, 0, 0], bool)
    auc = se.binary_auc(is_pos, np.concatenate([pos, neg]))
    pairs = [(a > b) + 0.5 * (a == b) for a in pos for b in neg]
    assert abs(auc - np.mean(pairs)) < 1e-9


def test_gates_skip_classes_with_no_images():
    y = np.array([0, 1])
    p = np.eye(6)[[0, 1]]
    ok, lines = se.gate_check(se.evaluate(y, p, CLASSES, VEN, 0.5), VEN, {
        "venomous_class_recall": 0.9, "non_venomous_recall": 0.9, "binary_venomous_recall": 0.9,
        "binary_auc": 0.9, "macro_f1": 0.5})
    assert ok and any("SKIP" in l for l in lines)
