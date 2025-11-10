from __future__ import annotations

"""Sales revenue regression using pandas + scikit-learn cross-validation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FILE_PATH = Path("datasets/15 ad spends.csv")


FEATURES = ["Ad_Spend", "Discount_Applied", "Clicks"]
TARGET = "Revenue"


def load_dataset(limit: int | None = 180):
    df = pd.read_csv(FILE_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    if limit:
        df = df.head(limit)
    X = df[FEATURES]
    y = df[TARGET].astype(float)
    return X, y


def build_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("reg", LinearRegression()),
    ])


def run_kfold(X, y, folds: int = 5):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=33)
    scores = []
    final_snapshot = None
    for train_idx, test_idx in splitter.split(X):
        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(mean_squared_error(y.iloc[test_idx], preds))
        final_snapshot = (y.iloc[test_idx], preds)
    return sum(scores) / len(scores), final_snapshot


def plot_preds(actual, predicted):
    plt.figure(figsize=(6, 4))
    plt.scatter(actual, predicted, alpha=0.6, color="maroon")
    diag_min = min(actual.min(), predicted.min())
    diag_max = max(actual.max(), predicted.max())
    plt.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="black")
    plt.title("Sales revenue predictions (last fold)")
    plt.xlabel("Actual revenue")
    plt.ylabel("Predicted revenue")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    features, targets = load_dataset()
    avg_mse, final = run_kfold(features, targets)
    print("5 fold MSE:", round(avg_mse, 2))
    if final:
        plot_preds(final[0], final[1])
