"""
train.py

Two-phase EfficientNetB0 fine-tuning for SnakeCheck Hyderabad.
Produces a 5-class softmax model (Non-Venomous + Big Four).

Phase 1 — head only (backbone frozen):   fast convergence, ~10 epochs
Phase 2 — top-40 base layers unfrozen:   fine-tuning, ~20 epochs

Usage:
    python scripts/train.py
    python scripts/train.py --epochs_phase1 5 --epochs_phase2 10  # quick test
    python scripts/train.py --data_dir data/processed             # custom data root
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config (matches PRD §10.3)
# ---------------------------------------------------------------------------

CLASS_NAMES = ["non_venomous", "cobra", "russells_viper", "common_krait", "sawscaled_viper"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# Phase 1
P1_LR = 1e-3
P1_EPOCHS = 15

# Phase 2
P2_LR = 5e-5
P2_UNFREEZE_TOP = 40
P2_EPOCHS = 25

DROPOUT_1 = 0.4
DROPOUT_2 = 0.3
DENSE_UNITS = 256
EARLY_STOP_PATIENCE = 6

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"


# ---------------------------------------------------------------------------
# Custom metric — macro recall (primary training signal per PRD §10.3)
# ---------------------------------------------------------------------------

class MacroRecall(keras.metrics.Metric):
    def __init__(self, num_classes=NUM_CLASSES, name="macro_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            shape=(num_classes,), initializer="zeros", name="tp"
        )
        self.false_negatives = self.add_weight(
            shape=(num_classes,), initializer="zeros", name="fn"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        y_true_classes = tf.argmax(y_true, axis=-1)
        for c in range(self.num_classes):
            true_c = tf.cast(tf.equal(y_true_classes, c), tf.float32)
            pred_c = tf.cast(tf.equal(y_pred_classes, c), tf.float32)
            self.true_positives[c].assign_add(tf.reduce_sum(true_c * pred_c))
            self.false_negatives[c].assign_add(tf.reduce_sum(true_c * (1 - pred_c)))

    def result(self):
        recalls = self.true_positives / (
            self.true_positives + self.false_negatives + keras.backend.epsilon()
        )
        return tf.reduce_mean(recalls)

    def reset_state(self):
        self.true_positives.assign(tf.zeros(self.num_classes))
        self.false_negatives.assign(tf.zeros(self.num_classes))


# ---------------------------------------------------------------------------
# Augmentation pipeline (PRD §10.3)
# ---------------------------------------------------------------------------

def build_augmentation() -> keras.Sequential:
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(35 / 360),
        layers.RandomZoom((-0.25, 0.25)),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomBrightness(0.25),
        layers.RandomContrast(0.25),
    ], name="augmentation")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_datasets(data_dir: Path):
    """
    Loads train/val splits from data_dir.
    Expects subdirectories named after CLASS_NAMES.
    Falls back to raw/ if processed/ doesn't exist.
    """
    if not data_dir.exists():
        raw_dir = data_dir.parent / "raw"
        if raw_dir.exists():
            print(f"[data] processed/ not found — using raw/ ({raw_dir})")
            data_dir = raw_dir
        else:
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                "Run dataset_download.py first, then preprocess.py to split into train/val."
            )

    print(f"[data] Loading from {data_dir}")

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir / "train" if (data_dir / "train").exists() else data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        validation_split=0.2,
        subset="training",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir / "train" if (data_dir / "train").exists() else data_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        validation_split=0.2,
        subset="validation",
    )

    return train_ds, val_ds


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


def compute_weights(data_dir: Path) -> dict:
    """Compute inverse-frequency class weights for imbalanced Big Four dataset."""
    counts = []
    for cls in CLASS_NAMES:
        d = data_dir / cls
        if d.exists():
            counts.append(len(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))))
        else:
            counts.append(1)

    total = sum(counts)
    weights = {i: total / (NUM_CLASSES * max(c, 1)) for i, c in enumerate(counts)}
    print("[weights]", {CLASS_NAMES[i]: f"{w:.2f}" for i, w in weights.items()})
    return weights


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(trainable_base: bool = False, unfreeze_top: int = 0) -> keras.Model:
    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = trainable_base
    if trainable_base and unfreeze_top > 0:
        for layer in base.layers[:-unfreeze_top]:
            layer.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augmentation()(inputs)
    # EfficientNetB0 expects pixel values in [0, 255] — preprocess internally
    x = keras.applications.efficientnet.preprocess_input(x * 255.0)
    x = base(x, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_1)(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT_2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def make_callbacks(run_dir: Path, monitor: str = "val_macro_recall", phase: int = 1):
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        callbacks.ModelCheckpoint(
            filepath=str(run_dir / f"best_phase{phase}.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.CSVLogger(str(run_dir / f"history_phase{phase}.csv")),
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(history, phase: int, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"Phase {phase} — Loss")
    axes[0].legend()

    recall_key = "macro_recall"
    if recall_key in history.history:
        axes[1].plot(history.history[recall_key], label="train")
        axes[1].plot(history.history[f"val_{recall_key}"], label="val")
        axes[1].set_title(f"Phase {phase} — Macro Recall")
        axes[1].legend()

    plt.tight_layout()
    out = out_dir / f"phase{phase}_curves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[plot] Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SnakeCheck EfficientNetB0 model")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs_phase1", type=int, default=P1_EPOCHS)
    parser.add_argument("--epochs_phase2", type=int, default=P2_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip_phase1", action="store_true",
                        help="Load best_phase1.keras and jump straight to phase 2.")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODELS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {run_id}  →  {run_dir}")

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[tf] GPUs available: {len(gpus)}")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    # Data
    train_ds, val_ds = load_datasets(args.data_dir)
    train_ds = train_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    class_weights = compute_weights(
        args.data_dir / "train" if (args.data_dir / "train").exists() else args.data_dir
    )

    # -----------------------------------------------------------------------
    # Phase 1 — frozen backbone, train head only
    # -----------------------------------------------------------------------
    if not args.skip_phase1:
        print("\n" + "=" * 60)
        print("PHASE 1 — Head training (backbone frozen)")
        print("=" * 60)

        model = build_model(trainable_base=False)
        model.compile(
            optimizer=keras.optimizers.Adam(P1_LR),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                MacroRecall(name="macro_recall"),
                keras.metrics.AUC(multi_label=False, name="auc"),
            ],
        )
        model.summary(show_trainable=True)

        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs_phase1,
            class_weight=class_weights,
            callbacks=make_callbacks(run_dir, phase=1),
        )
        plot_history(h1, phase=1, out_dir=run_dir)

    else:
        ckpt = run_dir / "best_phase1.keras"
        if not ckpt.exists():
            # look for most recent run
            runs = sorted(MODELS_DIR.glob("*/best_phase1.keras"))
            if not runs:
                raise FileNotFoundError("No phase 1 checkpoint found. Run without --skip_phase1 first.")
            ckpt = runs[-1]
        print(f"[phase1] Loading checkpoint: {ckpt}")
        model = keras.models.load_model(ckpt, custom_objects={"MacroRecall": MacroRecall})

    # -----------------------------------------------------------------------
    # Phase 2 — unfreeze top-40 base layers, fine-tune
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PHASE 2 — Fine-tuning (top {P2_UNFREEZE_TOP} base layers unfrozen)")
    print("=" * 60)

    # Rebuild with unfrozen layers so augmentation layer stays in graph
    model_p2 = build_model(trainable_base=True, unfreeze_top=P2_UNFREEZE_TOP)

    # Copy head weights from phase 1
    for layer in model_p2.layers:
        try:
            src = model.get_layer(layer.name)
            layer.set_weights(src.get_weights())
        except (ValueError, AttributeError):
            pass

    model_p2.compile(
        optimizer=keras.optimizers.Adam(P2_LR),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            MacroRecall(name="macro_recall"),
            keras.metrics.AUC(multi_label=False, name="auc"),
        ],
    )

    h2 = model_p2.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_phase2,
        class_weight=class_weights,
        callbacks=make_callbacks(run_dir, phase=2),
    )
    plot_history(h2, phase=2, out_dir=run_dir)

    # -----------------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------------
    final_path = run_dir / "snakecheck_final.keras"
    model_p2.save(final_path)
    print(f"\n[done] Final model saved → {final_path}")

    # Save run config for reproducibility
    config = {
        "run_id": run_id,
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "batch_size": args.batch_size,
        "phase1": {"lr": P1_LR, "epochs": args.epochs_phase1},
        "phase2": {"lr": P2_LR, "epochs": args.epochs_phase2, "unfreeze_top": P2_UNFREEZE_TOP},
        "class_weights": {CLASS_NAMES[i]: w for i, w in class_weights.items()},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[done] Config saved → {run_dir / 'config.json'}")

    # Quick summary of val macro recall
    val_recall = max(h2.history.get("val_macro_recall", [0]))
    print(f"\n[result] Best val macro recall: {val_recall:.4f}")
    gate = 0.90
    if val_recall >= gate:
        print(f"  PASS — exceeds PRD launch gate ({gate})")
    else:
        print(f"  BELOW gate ({gate}) — expand dataset or tune augmentation before TFLite export")


if __name__ == "__main__":
    main()
