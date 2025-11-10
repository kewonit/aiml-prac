from __future__ import annotations

"""Student score regression via pandas + scikit-learn with K-fold."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FILE_PATH = Path("datasets/student_exam_scores_12_13.csv")


FEATURES = [
    "hours_studied",
    "attendance_percent",
    "Internal_marks",
]
TARGET = "exam_score"


def load_dataset(limit: int | None = 150):
    df = pd.read_csv(FILE_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    if limit:
        df = df.head(limit)
    X = df[FEATURES]
    y = df[TARGET].astype(float)
    return X, y


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("scale", StandardScaler(), FEATURES),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("reg", LinearRegression()),
    ])


def run_kfold(X, y, folds: int = 5):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=21)
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
    plt.scatter(actual, predicted, alpha=0.6, color="darkorange")
    diag_min = min(actual.min(), predicted.min())
    diag_max = max(actual.max(), predicted.max())
    plt.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="black")
    plt.title("Student score predictions (last fold)")
    plt.xlabel("Actual score")
    plt.ylabel("Predicted score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    features, labels = load_dataset()
    mse_avg, snapshot = run_kfold(features, labels)
    print("5 fold MSE:", round(mse_avg, 3))
    if snapshot:
        plot_preds(snapshot[0], snapshot[1])
