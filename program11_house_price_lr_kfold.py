from __future__ import annotations

"""House price regression with scikit-learn pipelines and shuffled K-fold."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FILE_PATH = Path("datasets/House_Price_Dataset 11.csv")


NUMERIC_FEATURES = ["Area", "Bedrooms"]
CATEGORICAL_FEATURES = ["Location"]


def load_dataset(limit: int | None = 120):
    df = pd.read_csv(FILE_PATH)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["Price"])
    if limit:
        df = df.head(limit)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Price"].astype(float)
    return X, y


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("prep", preprocessor),
        ("reg", LinearRegression()),
    ])


def run_kfold(X, y, splits: int = 4):
    kfold = KFold(n_splits=splits, shuffle=True, random_state=42)
    scores = []
    final_plot = None
    for train_idx, test_idx in kfold.split(X):
        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        score = mean_squared_error(y.iloc[test_idx], preds)
        scores.append(score)
        final_plot = (y.iloc[test_idx], preds)
    return sum(scores) / len(scores), final_plot


def plot_predictions(actual, predicted):
    plt.figure(figsize=(6, 4))
    plt.scatter(actual, predicted, alpha=0.6, color="seagreen")
    diag_min = min(actual.min(), predicted.min())
    diag_max = max(actual.max(), predicted.max())
    plt.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="black")
    plt.title("House price predictions (last fold)")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    features, targets = load_dataset()
    mse_avg, final = run_kfold(features, targets)
    print("Average MSE over folds:", round(mse_avg, 2))
    if final:
        plot_predictions(final[0], final[1])
