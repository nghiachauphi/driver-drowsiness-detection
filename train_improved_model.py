"""Train and evaluate a subject-independent MobileNetV2 drowsiness model.

This pipeline fixes the main weaknesses of the original notebook:
- subjects never overlap across train/validation/test;
- validation and test folds are stratified by label and grouped by subject;
- preprocessing is serializable and identical at training/inference time;
- phase-specific checkpoints are compared and the best one is reloaded;
- the deployed threshold is selected only from validation predictions;
- image-level and subject-macro metrics are exported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "outputs"
MODEL_DIR = OUTPUT_ROOT / "models"
RESULT_DIR = OUTPUT_ROOT / "results"
TABLE_DIR = OUTPUT_ROOT / "tables"
DEFAULT_DATASET = Path(
    r"C:\Users\NGHIACP\.cache\kagglehub\datasets\ismailnasri20"
    r"\driver-drowsiness-dataset-ddd\versions\1"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--extra-data",
        type=Path,
        action="append",
        default=[],
        help="Optional additional dataset root; may be supplied more than once.",
    )
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test the pipeline with at most 120 images per subject/class.",
    )
    return parser.parse_args()


def discover_records(dataset_roots: list[Path]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for root_index, root in enumerate(dataset_roots):
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset không tồn tại: {root}")
        for path in root.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            class_directory = next(
                (
                    ancestor
                    for ancestor in path.parents
                    if ancestor != root and "drowsy" in ancestor.name.lower()
                ),
                None,
            )
            if class_directory is None:
                continue
            class_name = class_directory.name.lower()
            label = 0 if "non" in class_name or "not" in class_name else 1
            relative_parts = path.relative_to(class_directory).parts
            if len(relative_parts) > 1:
                # Supports class/person/frame.jpg for locally collected data.
                raw_subject = relative_parts[0]
            elif "_" in path.stem or "-" in path.stem:
                # Supports names such as P01_000123.jpg.
                raw_subject = re.split(r"[_-]", path.stem, maxsplit=1)[0]
            else:
                # The original DDD names are A0001.png, B0001.png, ..., ZA0001.
                match = re.match(r"^([A-Za-z]+)", path.stem)
                raw_subject = match.group(1) if match else path.stem
            raw_subject = str(raw_subject).strip().upper()
            # Prefix extra datasets so coincidentally identical filenames do not
            # create false subject overlap with the original Kaggle dataset.
            subject = f"R{root_index}_{raw_subject}"
            records.append(
                {"path": str(path), "label": label, "subject": subject}
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("Không tìm thấy ảnh trong thư mục Drowsy/Non-Drowsy.")
    return frame.drop_duplicates("path").reset_index(drop=True)


def limit_for_quick_run(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    sampled_parts = [
        part.sample(min(len(part), 120), random_state=seed)
        for _, part in frame.groupby(["subject", "label"], sort=False)
    ]
    return pd.concat(sampled_parts, ignore_index=True)


def make_subject_split(
    frame: pd.DataFrame, fold: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    outer_splits = list(
        outer.split(frame, y=frame["label"], groups=frame["subject"])
    )
    train_val_index, test_index = outer_splits[fold]
    train_val = frame.iloc[train_val_index].reset_index(drop=True)
    test = frame.iloc[test_index].reset_index(drop=True)

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + fold)
    train_index, validation_index = next(
        inner.split(
            train_val,
            y=train_val["label"],
            groups=train_val["subject"],
        )
    )
    train = train_val.iloc[train_index].reset_index(drop=True)
    validation = train_val.iloc[validation_index].reset_index(drop=True)

    subject_sets = [
        set(train["subject"]),
        set(validation["subject"]),
        set(test["subject"]),
    ]
    if subject_sets[0] & subject_sets[1] or subject_sets[0] & subject_sets[2]:
        raise AssertionError("Rò rỉ người giữa train và validation/test.")
    if subject_sets[1] & subject_sets[2]:
        raise AssertionError("Rò rỉ người giữa validation và test.")
    return train, validation, test


def decode_image(path: tf.Tensor, label: tf.Tensor, image_size: int):
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize_with_pad(image, image_size, image_size)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03, fill_mode="nearest"),
            layers.RandomZoom(0.10, fill_mode="nearest"),
            layers.RandomContrast(0.20),
            layers.RandomBrightness(0.15, value_range=(0, 255)),
        ],
        name="training_augmentation",
    )


def make_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(
        (frame["path"].values, frame["label"].values.astype("float32"))
    )
    if training:
        dataset = dataset.shuffle(
            min(len(frame), 10_000), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda path, label: decode_image(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size)
    if training:
        augmentation = build_augmentation()
        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(image_size: int) -> tuple[Model, Model]:
    inputs = layers.Input((image_size, image_size, 3), name="image")
    # Equivalent to MobileNetV2 preprocess_input for [0,255] images, but unlike
    # Lambda(preprocess_input) this serializes without custom_objects.
    normalized = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(
        inputs
    )
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    features = backbone(normalized, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(0.45)(features)
    features = layers.Dense(
        128,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(features)
    features = layers.Dropout(0.35)(features)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(features)
    return Model(inputs, outputs, name="ImprovedMobileNetV2"), backbone


def compile_model(model: Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=1e-4
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.04),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def callbacks_for(checkpoint_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.4,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def predict(model: Model, dataset: tf.data.Dataset) -> np.ndarray:
    return model.predict(dataset, verbose=1).reshape(-1)


def validation_auc(model_path: Path, dataset: tf.data.Dataset) -> float:
    model = tf.keras.models.load_model(model_path, compile=False)
    labels = np.concatenate([label.numpy() for _, label in dataset]).astype(int)
    return float(roc_auc_score(labels, predict(model, dataset)))


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities
    )
    finite = np.isfinite(thresholds)
    score = true_positive_rate[finite] - false_positive_rate[finite]
    threshold = float(thresholds[finite][int(np.argmax(score))])
    # A low Youden threshold can make a safety UI trigger on most awake frames
    # when score calibration shifts between subjects. Keep the conventional
    # sigmoid cut-off as a conservative floor; the temporal layer adds the
    # second-stage evidence required for an actual video/webcam alert.
    return float(np.clip(threshold, 0.50, 0.90))


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def subject_macro_metrics(
    frame: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> tuple[dict[str, float], pd.DataFrame]:
    scored = frame[["subject", "label", "path"]].copy()
    scored["probability"] = probabilities
    scored["prediction"] = (probabilities >= threshold).astype(int)
    rows = []
    for subject, part in scored.groupby("subject"):
        labels = part["label"].to_numpy()
        predictions = part["prediction"].to_numpy()
        probabilities_for_subject = part["probability"].to_numpy()
        row = {
            "subject": subject,
            "images": len(part),
            "accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, zero_division=0),
            "roc_auc": (
                roc_auc_score(labels, probabilities_for_subject)
                if len(np.unique(labels)) == 2
                else np.nan
            ),
        }
        rows.append(row)
    subject_table = pd.DataFrame(rows)
    macro = {
        column: float(subject_table[column].mean(skipna=True))
        for column in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    }
    return macro, subject_table


def split_summary(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "name": name,
        "images": len(frame),
        "subjects": sorted(frame["subject"].unique().tolist()),
        "subject_count": int(frame["subject"].nunique()),
        "drowsy_ratio": float(frame["label"].mean()),
    }


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    frame = discover_records([args.dataset, *args.extra_data])
    if args.quick:
        frame = limit_for_quick_run(frame, args.seed)
    train, validation, test = make_subject_split(frame, args.fold, args.seed)

    split_table = pd.DataFrame(
        [
            split_summary("train", train),
            split_summary("validation", validation),
            split_summary("test", test),
        ]
    )
    split_table.to_json(
        TABLE_DIR / "improved_subject_split.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    print(split_table[["name", "images", "subject_count", "drowsy_ratio"]])

    train_dataset = make_dataset(
        train, args.image_size, args.batch_size, True, args.seed
    )
    validation_dataset = make_dataset(
        validation, args.image_size, args.batch_size, False, args.seed
    )
    test_dataset = make_dataset(
        test, args.image_size, args.batch_size, False, args.seed
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train["label"].to_numpy(),
    )
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    phase1_path = MODEL_DIR / "improved_mobilenetv2_phase1.keras"
    phase2_path = MODEL_DIR / "improved_mobilenetv2_phase2.keras"
    final_path = MODEL_DIR / "improved_mobilenetv2.keras"

    model, backbone = build_model(args.image_size)
    compile_model(model, learning_rate=3e-4)
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs_head if not args.quick else 1,
        class_weight=class_weight,
        callbacks=callbacks_for(phase1_path),
        verbose=1,
    )

    phase1_model = tf.keras.models.load_model(phase1_path, compile=False)
    backbone = next(
        layer
        for layer in phase1_model.layers
        if isinstance(layer, Model) and layer.name.startswith("mobilenetv2")
    )
    backbone.trainable = True
    for layer in backbone.layers[:-25]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(phase1_model, learning_rate=8e-6)
    phase1_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs_finetune if not args.quick else 1,
        class_weight=class_weight,
        callbacks=callbacks_for(phase2_path),
        verbose=1,
    )

    candidate_paths = [phase1_path, phase2_path]
    candidate_scores = {
        str(path): validation_auc(path, validation_dataset)
        for path in candidate_paths
    }
    best_path = max(candidate_paths, key=lambda path: candidate_scores[str(path)])
    shutil.copy2(best_path, final_path)
    best_model = tf.keras.models.load_model(final_path, compile=False)

    validation_labels = validation["label"].to_numpy().astype(int)
    validation_probabilities = predict(best_model, validation_dataset)
    threshold = choose_threshold(validation_labels, validation_probabilities)

    test_labels = test["label"].to_numpy().astype(int)
    test_probabilities = predict(best_model, test_dataset)
    image_metrics = classification_metrics(
        test_labels, test_probabilities, threshold
    )
    macro_metrics, subject_table = subject_macro_metrics(
        test, test_probabilities, threshold
    )
    # A new checkpoint is not automatically a better checkpoint. Only mark it
    # deployable when a full run generalizes beyond chance on unseen subjects.
    deployable = bool(
        not args.quick
        and image_metrics["roc_auc"] >= 0.55
        and image_metrics["balanced_accuracy"] >= 0.55
        and macro_metrics["balanced_accuracy"] >= 0.55
    )
    subject_table.to_csv(
        TABLE_DIR / "improved_subject_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "model_name": "Improved MobileNetV2",
        "model_path": str(final_path.relative_to(ROOT)),
        "input_size": args.image_size,
        "class_names": ["Non-Drowsy", "Drowsy"],
        "positive_class": 1,
        "recommended_threshold": threshold,
        "threshold_policy": "validation_youden_with_0.50_safety_floor",
        "quick_run": args.quick,
        "deployable": deployable,
        "fold": args.fold,
        "candidate_validation_auc": candidate_scores,
        "selected_checkpoint": best_path.name,
        "image_metrics": image_metrics,
        "subject_macro_metrics": macro_metrics,
        "splits": {
            name: split_summary(name, split)
            for name, split in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]
        },
    }
    metadata_path = RESULT_DIR / "improved_model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["image_metrics"], indent=2))
    print(json.dumps(metadata["subject_macro_metrics"], indent=2))
    print(f"Saved model: {final_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
