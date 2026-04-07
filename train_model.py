"""
Network Traffic Classifier — CNN + BiLSTM
==========================================
This is the version that successfully trained your model.
Saves: best_model.keras, final_model.keras, scaler.pkl,
       label_encoder.pkl, metadata.json,
       confusion_matrix.png, training_curves.png

Usage:
    python train_model.py --dataset dataset.csv --epochs 40
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ── Config ────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "pkt_len", "ip_len", "trans_len",
    "direction", "iat",
    "protocol", "tcp_flags", "win_size",
    "burst_size", "flow_bytes",
    "roll_mean", "roll_std",
]
N_FEATURES  = 12
WINDOW_SIZE = 100

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_metric(names, *candidates):
    for c in candidates:
        if c in names:
            return names.index(c)
    raise ValueError(f"None of {candidates} found in {names}")

def find_history_key(keys, *candidates):
    for c in candidates:
        if c in keys:
            return c
    raise ValueError(f"None of {candidates} found in {keys}")

# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(csv_path):
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Shape : {df.shape}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32).reshape(-1, WINDOW_SIZE, N_FEATURES)

    enc = LabelEncoder()
    y   = enc.fit_transform(df["label"].values)

    print(f"  Classes: {list(enc.classes_)}")
    return X, y, enc

# ── Oversampling ──────────────────────────────────────────────────────────────

def oversample(X, y, random_state=42):
    rng = np.random.default_rng(random_state)
    max_count = int(np.bincount(y).max())
    indices_all = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < max_count:
            extra = rng.choice(idx, size=max_count - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        indices_all.append(idx)
    indices = np.concatenate(indices_all)
    rng.shuffle(indices)
    print(f"  After oversampling: {len(indices)} samples "
          f"({max_count}/class × {len(np.unique(y))} classes)")
    return X[indices], y[indices]

# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(X_train, X_val, X_test):
    N, W, F = X_train.shape
    scaler    = StandardScaler()
    X_train_n = scaler.fit_transform(X_train.reshape(-1, F)).reshape(N, W, F)
    X_val_n   = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape[0], W, F)
    X_test_n  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], W, F)
    return X_train_n, X_val_n, X_test_n, scaler

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(n_classes):
    inp = keras.Input(shape=(WINDOW_SIZE, N_FEATURES), name="packets")

    x = layers.Conv1D(64, 5, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    x = layers.Bidirectional(
            layers.LSTM(32, dropout=0.2))(x)

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="category")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ── Training ──────────────────────────────────────────────────────────────────

def train(csv_path, epochs, batch_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    X, y, enc = load_dataset(csv_path)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    print("\nOversampling training set...")
    X_tr, y_tr = oversample(X_tr, y_tr)

    X_tr, X_val, X_te, scaler = normalize(X_tr, X_val, X_te)

    model = build_model(len(enc.classes_))
    model.summary()

    # Auto-detect metric name (handles all Keras versions)
    metric_names = model.metrics_names
    print(f"\n  Keras metric names: {metric_names}")
    acc_key = next(m for m in metric_names if "acc" in m)
    MON = f"val_{acc_key}"
    print(f"  Monitoring: {MON}\n")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=MON, mode="max", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor=MON, mode="max", factor=0.5, patience=4, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best_model.keras"),
            monitor=MON, mode="max", save_best_only=True),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model separately
    model.save(os.path.join(out_dir, "final_model.keras"))

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n── Test Evaluation ──────────────────────────────────────────────")
    results  = model.evaluate(X_te, y_te, verbose=0)
    names    = model.metrics_names
    print(f"  Available metric names: {names}")
    acc_idx  = find_metric(names, "accuracy", "category_accuracy")
    accuracy = results[acc_idx]
    print(f"  Test accuracy : {accuracy:.4f}  ({accuracy*100:.1f}%)")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    print("\nClassification report:")
    print(classification_report(y_te, y_pred, target_names=enc.classes_))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm      = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=enc.classes_, yticklabels=enc.classes_)
    plt.title("Confusion Matrix — Website Category")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ confusion_matrix.png saved")

    # ── Training curves ───────────────────────────────────────────────────────
    hkeys   = list(history.history.keys())
    trn_key = find_history_key(hkeys, "accuracy", "category_accuracy")
    val_key = find_history_key(hkeys, "val_accuracy", "val_category_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history[trn_key],  label="Train")
    axes[0].plot(history.history[val_key],  label="Val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    loss_key     = find_history_key(hkeys, "loss")
    val_loss_key = find_history_key(hkeys, "val_loss")
    axes[1].plot(history.history[loss_key],     label="Train")
    axes[1].plot(history.history[val_loss_key], label="Val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ training_curves.png saved")

    # ── Save artefacts ────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "scaler.pkl"),        "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f: pickle.dump(enc, f)
    with open(os.path.join(out_dir, "metadata.json"),     "w") as f:
        json.dump({
            "classes":      list(enc.classes_),
            "n_classes":    len(enc.classes_),
            "window_size":  WINDOW_SIZE,
            "n_features":   N_FEATURES,
            "feature_names": FEATURE_NAMES,
            "test_accuracy": round(float(accuracy), 4),
        }, f, indent=2)

    print(f"\n✓ All artefacts saved to {out_dir}/")
    print(f"  best_model.keras · final_model.keras · scaler.pkl")
    print(f"  label_encoder.pkl · metadata.json")
    print(f"  confusion_matrix.png · training_curves.png")
    print(f"\n  Final test accuracy: {accuracy*100:.1f}%")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs",  type=int, default=40)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--out_dir", default="model_artifacts")
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.batch, args.out_dir)