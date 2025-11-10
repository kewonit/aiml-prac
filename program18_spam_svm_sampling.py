"""Linear SVM with manual oversampling on the emails dataset.
Loads the real word-count features, balances the train split by cloning the
spam minority class, then optimises a linear SVM using hinge-loss updates.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("datasets/emails_16_17_18_19.csv")


def load_email_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    features = df.drop(columns=["Email No.", "Prediction"]).to_numpy(dtype=float)
    labels = df["Prediction"].map({0: -1, 1: 1}).to_numpy(dtype=int)
    return features, labels


def oversample_minority(
    X: np.ndarray, y: np.ndarray, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    counts = Counter(y)
    if len(counts) < 2:
        return X, y
    ((maj_label, maj_count), (min_label, min_count)) = sorted(
        counts.items(), key=lambda item: item[1], reverse=True
    )
    if maj_count == min_count:
        return X, y

    rng = np.random.default_rng(seed)
    minority_indices = np.where(y == min_label)[0]
    extra_indices = rng.choice(minority_indices, size=maj_count - min_count, replace=True)
    X_extra = X[extra_indices]
    y_extra = y[extra_indices]
    X_balanced = np.vstack((X, X_extra))
    y_balanced = np.concatenate((y, y_extra))
    shuffle_idx = rng.permutation(len(X_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]


def train_svm(
    rows: Sequence[Tuple[np.ndarray, float]],
    steps: int = 35,
    lr: float = 0.0008,
    reg: float = 0.01,
) -> Tuple[np.ndarray, float]:
    feature_count = rows[0][0].shape[0]
    weights = np.zeros(feature_count)
    bias = 0.0

    rows_list: List[Tuple[np.ndarray, float]] = list(rows)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_balanced, y_balanced = oversample_minority(X_train, y_train)

    train_rows = [(feat, lbl) for feat, lbl in zip(X_balanced, y_balanced)]
    test_rows = [(feat, lbl) for feat, lbl in zip(X_test, y_test)]

    weights, bias = train_svm(train_rows)
    acc, prec, rec, f1 = evaluate(test_rows, weights, bias)

    print("Train class counts:", Counter(int(lbl) for lbl in y_train))
    print("Balanced train counts:", Counter(int(lbl) for lbl in y_balanced))
    print("Test class counts:", Counter(int(lbl) for lbl in y_test))
    print("Accuracy:", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall:", round(rec, 3))
    print("F1:", round(f1, 3))
