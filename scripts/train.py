"""
train.py - two-phase EfficientNetB0 training for SnakeCheck on the frozen splits.

Reads the folders written by
    python scripts/build_splits.py --materialize data/splits
which look like data/splits/<split>/<class>/<image>, with splits
train, val, test and test_external.

  * Trains on `train` only. `val` picks the best epoch and the venomous threshold.
  * `test` (India, unseen observers) and `test_external` (Telangana, held out
    completely, plus your own photos) are scored ONCE at the end with the
    threshold fixed on val. Nothing is ever evaluated on training images.
  * Classes come from config/dataset.yaml (5 snake classes + not_a_snake).

Phase 1: backbone frozen, train the head.
Phase 2: unfreeze the top layers of EfficientNetB0 (BatchNorm stays frozen).

Usage
    python scripts/train.py
    python scripts/train.py --epochs_phase1 2 --epochs_phase2 2        # quick check
    python scripts/train.py --eval_only models/<run>/snakecheck_final.keras
    python scripts/train.py --venomous_recall_target 0.97              # stricter safety

The notebook imports this file, so the two always train the same way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import snake_eval

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config" / "dataset.yaml"
SPLITS = ("train", "val", "test", "test_external")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
P1_LR, P1_EPOCHS = 1e-3, 15
P2_LR, P2_EPOCHS, P2_UNFREEZE_TOP = 5e-5, 25, 40
DROPOUT_1, DROPOUT_2, DENSE_UNITS = 0.4, 0.3, 256
EARLY_STOP_PATIENCE = 6

# Pick the threshold on val so this share of venomous snakes is caught.
VENOMOUS_RECALL_TARGET = 0.95

# Launch gates, checked on the test sets (never on val or train).
GATES = {
    "venomous_class_recall": 0.90,
    "non_venomous_recall": 0.92,
    "binary_venomous_recall": 0.90,
    "binary_auc": 0.92,
    "macro_f1": 0.88,
}


def load_classes(config_path: Path = CONFIG_PATH) -> tuple[list[str], list[str]]:
    """Class order and the venomous classes, from config/dataset.yaml."""
    import yaml
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = list(cfg["classes"])
    venomous = [c for c, v in cfg["classes"].items() if v.get("venomous") is True]
    return names, venomous


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_split(data_dir: Path, split: str, class_names: list[str]) -> tuple[list[str], np.ndarray]:
    paths, labels = [], []
    for i, c in enumerate(class_names):
        d = data_dir / split / c
        if d.is_dir():
            files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
            paths += [str(f) for f in files]
            labels += [i] * len(files)
    return paths, np.array(labels, dtype=np.int64)


def make_dataset(paths, labels, num_classes, batch_size=BATCH_SIZE, shuffle=False):
    """tf.data pipeline. Images stay in 0-255; EfficientNet rescales inside the model."""
    import tensorflow as tf

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return tf.cast(img, tf.float32), tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_splits(data_dir: Path, class_names, batch_size=BATCH_SIZE) -> dict:
    out = {}
    for s in SPLITS:
        paths, labels = list_split(data_dir, s, class_names)
        counts = np.bincount(labels, minlength=len(class_names)) if len(labels) else np.zeros(len(class_names), int)
        print(f"[data] {s:<14}" + "  ".join(f"{c} {n}" for c, n in zip(class_names, counts)))
        out[s] = {"paths": paths, "labels": labels, "counts": counts,
                  "ds": make_dataset(paths, labels, len(class_names), batch_size, shuffle=(s == "train"))
                  if paths else None}
    if out["train"]["ds"] is None or out["val"]["ds"] is None:
        raise SystemExit(f"No train/val images under {data_dir}. Run build_splits.py --materialize first.")
    return out


def class_weights(counts: np.ndarray) -> dict:
    """Inverse-frequency weights from the TRAIN counts."""
    counts = np.maximum(counts, 1)
    w = counts.sum() / (len(counts) * counts)
    return {i: float(x) for i, x in enumerate(w)}


def eval_fingerprint() -> str:
    """Hash of the frozen eval manifest, saved with every model so runs are comparable."""
    for p in (ROOT / "manifests" / "eval_frozen.csv", DATA_DIR.parent / "manifests" / "eval_frozen.csv"):
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return "unknown"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def macro_recall_metric(num_classes):
    from tensorflow import keras
    import tensorflow as tf

    @keras.utils.register_keras_serializable(package="snakecheck")
    class MacroRecall(keras.metrics.Metric):
        def __init__(self, num_classes=num_classes, name="macro_recall", **kw):
            super().__init__(name=name, **kw)
            self.num_classes = num_classes
            self.tp = self.add_weight(shape=(num_classes,), initializer="zeros", name="tp")
            self.fn = self.add_weight(shape=(num_classes,), initializer="zeros", name="fn")

        def update_state(self, y_true, y_pred, sample_weight=None):
            t = tf.one_hot(tf.argmax(y_true, -1), self.num_classes)
            p = tf.one_hot(tf.argmax(y_pred, -1), self.num_classes)
            self.tp.assign_add(tf.reduce_sum(t * p, 0))
            self.fn.assign_add(tf.reduce_sum(t * (1 - p), 0))

        def result(self):
            seen = self.tp + self.fn
            r = self.tp / (seen + keras.backend.epsilon())
            return tf.reduce_sum(r) / tf.maximum(tf.reduce_sum(tf.cast(seen > 0, tf.float32)), 1.0)

        def reset_state(self):
            self.tp.assign(tf.zeros(self.num_classes))
            self.fn.assign(tf.zeros(self.num_classes))

        def get_config(self):
            return {**super().get_config(), "num_classes": self.num_classes}

    return MacroRecall


def build_model(num_classes: int, weights="imagenet"):
    from tensorflow import keras
    from tensorflow.keras import layers

    aug = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(35 / 360),
        layers.RandomZoom((-0.25, 0.25)),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomBrightness(0.25, value_range=(0, 255)),
        layers.RandomContrast(0.25),
    ], name="augmentation")
    base = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                             input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))   # 0-255 pixels
    x = aug(inputs)
    x = base(x, training=False)          # keeps BatchNorm in inference mode in both phases
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_1)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT_2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="snakecheck"), base


def unfreeze_top(base, n: int) -> None:
    from tensorflow.keras import layers
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model, lr, num_classes):
    from tensorflow import keras
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                  metrics=["accuracy", macro_recall_metric(num_classes)()])


def make_callbacks(ckpt: Path, log: Path):
    from tensorflow.keras import callbacks
    m = "val_macro_recall"
    return [
        callbacks.ModelCheckpoint(str(ckpt), monitor=m, mode="max", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor=m, mode="max", patience=EARLY_STOP_PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor=m, mode="max", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        callbacks.CSVLogger(str(log)),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def predict_split(model, split: dict) -> np.ndarray:
    return model.predict(split["ds"], verbose=0)


def evaluate_all(model, splits, class_names, venomous, target_recall=VENOMOUS_RECALL_TARGET,
                 threshold=None) -> dict:
    """Pick the threshold on val, then score val, test and test_external with it."""
    probs = {s: predict_split(model, splits[s]) for s in ("val", "test", "test_external") if splits[s]["ds"]}
    if threshold is None:
        threshold = snake_eval.pick_threshold(splits["val"]["labels"], probs["val"], class_names,
                                              venomous, target_recall)
    print(f"\n[threshold] {threshold:.3f}  (chosen on val for venomous recall >= {target_recall})")
    results = {"threshold": threshold, "venomous_recall_target": target_recall}
    for s, p in probs.items():
        m = snake_eval.evaluate(splits[s]["labels"], p, class_names, venomous, threshold)
        results[s] = m
        print("\n" + snake_eval.format_report(s, m))
    for s in ("test", "test_external"):
        if s in results:
            ok, lines = snake_eval.gate_check(results[s], venomous, GATES)
            results[s]["gates_passed"] = ok
            print(f"\nLaunch gates on {s}:\n" + "\n".join(lines))
            print("  ALL PASS" if ok else "  NOT PASSED: do not ship this model yet")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(data_dir=DATA_DIR, models_dir=MODELS_DIR, epochs_phase1=P1_EPOCHS, epochs_phase2=P2_EPOCHS,
          batch_size=BATCH_SIZE, weights="imagenet", target_recall=VENOMOUS_RECALL_TARGET):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    class_names, venomous = load_classes()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(models_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_id} -> {run_dir}\n[classes] {class_names}\n[venomous] {venomous}")

    splits = load_splits(Path(data_dir), class_names, batch_size)
    cw = class_weights(splits["train"]["counts"])
    print("[weights] " + ", ".join(f"{c} {cw[i]:.2f}" for i, c in enumerate(class_names)))

    model, base = build_model(len(class_names), weights)
    histories = {}
    if epochs_phase1:
        print("\n=== Phase 1: head only ===")
        compile_model(model, P1_LR, len(class_names))
        h = model.fit(splits["train"]["ds"], validation_data=splits["val"]["ds"], epochs=epochs_phase1,
                      class_weight=cw, callbacks=make_callbacks(run_dir / "best_phase1.keras",
                                                                run_dir / "history_phase1.csv"))
        histories["phase1"] = h.history
    if epochs_phase2:
        print(f"\n=== Phase 2: top {P2_UNFREEZE_TOP} backbone layers unfrozen ===")
        unfreeze_top(base, P2_UNFREEZE_TOP)
        compile_model(model, P2_LR, len(class_names))
        h = model.fit(splits["train"]["ds"], validation_data=splits["val"]["ds"], epochs=epochs_phase2,
                      class_weight=cw, callbacks=make_callbacks(run_dir / "best_phase2.keras",
                                                                run_dir / "history_phase2.csv"))
        histories["phase2"] = h.history

    final = run_dir / "snakecheck_final.keras"
    model.save(final)
    results = evaluate_all(model, splits, class_names, venomous, target_recall)
    meta = {
        "run_id": run_id, "class_names": class_names, "venomous_classes": venomous,
        "img_size": IMG_SIZE, "input": "RGB 0-255 float, resized to 224x224",
        "threshold": results["threshold"], "venomous_recall_target": target_recall,
        "eval_manifest_sha": eval_fingerprint(),
        "train_counts": dict(zip(class_names, map(int, splits["train"]["counts"]))),
        "class_weights": {class_names[i]: w for i, w in cw.items()},
        "phase1": {"lr": P1_LR, "epochs": epochs_phase1},
        "phase2": {"lr": P2_LR, "epochs": epochs_phase2, "unfreeze_top": P2_UNFREEZE_TOP},
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] model {final}\n[done] metrics {run_dir / 'metrics.json'}")
    return model, results, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--models_dir", type=Path, default=MODELS_DIR)
    ap.add_argument("--epochs_phase1", type=int, default=P1_EPOCHS)
    ap.add_argument("--epochs_phase2", type=int, default=P2_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--venomous_recall_target", type=float, default=VENOMOUS_RECALL_TARGET)
    ap.add_argument("--weights", default="imagenet", help="'imagenet', or 'none' for an offline smoke test")
    ap.add_argument("--eval_only", type=Path, help="score a saved .keras model instead of training")
    a = ap.parse_args()
    weights = None if a.weights == "none" else a.weights
    if a.eval_only:
        import tensorflow as tf
        class_names, venomous = load_classes()
        macro_recall_metric(len(class_names))
        model = tf.keras.models.load_model(a.eval_only)
        splits = load_splits(a.data_dir, class_names, a.batch_size)
        evaluate_all(model, splits, class_names, venomous, a.venomous_recall_target)
        return
    train(a.data_dir, a.models_dir, a.epochs_phase1, a.epochs_phase2, a.batch_size, weights,
          a.venomous_recall_target)


if __name__ == "__main__":
    main()
