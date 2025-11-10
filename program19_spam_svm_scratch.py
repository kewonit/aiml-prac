"""SVM from scratch on the full emails dataset.
Loads all word-count features, performs a manual train/test split, trains a
linear SVM with hinge-loss updates, then evaluates on the held-out emails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

DATASET_PATH = Path("datasets/emails_16_17_18_19.csv")


def load_email_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.drop(columns=["Email No.", "Prediction"]).to_numpy(dtype=float)
    y = df["Prediction"].map({0: -1, 1: 1}).to_numpy(dtype=int)
    return X, y


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 11,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    cut = int(len(indices) * (1 - test_ratio))
    train_idx = indices[:cut]
    test_idx = indices[cut:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def standardise(train_X: np.ndarray, test_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    return (train_X - mean) / std, (test_X - mean) / std


def train_svm(
    rows: Sequence[Tuple[np.ndarray, float]],
    steps: int = 35,
    lr: float = 0.0008,
    reg: float = 0.01,
) -> Tuple[np.ndarray, float]:
    feature_count = rows[0][0].shape[0]
    weights = np.zeros(feature_count)
    bias = 0.0
    rows_list = list(rows)
    for _ in range(steps):
        np.random.shuffle(rows_list)
        for feats, label in rows_list:
            margin = label * (np.dot(weights, feats) + bias)
            if margin < 1:
                weights = (1 - lr * reg) * weights + lr * label * feats
                bias += lr * label
            else:
                weights = (1 - lr * reg) * weights
    return weights, bias


def predict(weights: np.ndarray, bias: float, feats: np.ndarray) -> int:
    return 1 if np.dot(weights, feats) + bias >= 0 else -1


def evaluate(
    rows: Iterable[Tuple[np.ndarray, float]], weights: np.ndarray, bias: float
) -> Tuple[float, float, float, float]:
    tp = fp = tn = fn = 0
    for feats, label in rows:
        guess = predict(weights, bias, feats)
        if guess == 1 and label == 1:
            tp += 1
        elif guess == 1 and label == -1:
            fp += 1
        elif guess == -1 and label == -1:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    X, y = load_email_dataset(DATASET_PATH)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)
    X_train, X_test = standardise(X_train, X_test)

    train_rows = [(feat, lbl) for feat, lbl in zip(X_train, y_train)]
    test_rows = [(feat, lbl) for feat, lbl in zip(X_test, y_test)]

    weights, bias = train_svm(train_rows)
    acc, prec, rec, f1 = evaluate(test_rows, weights, bias)

    print("Train size:", len(train_rows))
    print("Test size:", len(test_rows))
    print("Accuracy:", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall:", round(rec, 3))
    print("F1:", round(f1, 3))
