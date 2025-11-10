from __future__ import annotations

"""IT salary regression using a tidy pandas + scikit-learn workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FILE_PATH = Path("datasets/salary_data_14_converted.csv")


NUMERIC_FEATURES = ["Age", "Years of Experience"]
CATEGORICAL_FEATURES = ["Education Level", "Job Title", "Gender"]
TARGET = "Salary"


def load_dataset(limit: int | None = 200):
    df = pd.read_csv(FILE_PATH)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    if limit:
        df = df.head(limit)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(float)
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


def run_kfold(X, y, folds: int = 5):
    splitter = KFold(n_splits=folds, shuffle=True, random_state=19)
    scores = []
    final_snapshot = None
    for train_idx, test_idx in splitter.split(X):
        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        scores.append(mean_squared_error(y.iloc[test_idx], preds))
        final_snapshot = (y.iloc[test_idx], preds)
    return sum(scores) / len(scores), final_snapshot


def plot_predictions(actual, predicted):
    plt.figure(figsize=(6, 4))
    plt.scatter(actual, predicted, alpha=0.55, color="purple")
    diag_min = min(actual.min(), predicted.min())
    diag_max = max(actual.max(), predicted.max())
    plt.plot([diag_min, diag_max], [diag_min, diag_max], color="black", linestyle="--")
    plt.title("Predicted vs actual salary")
    plt.xlabel("Actual salary")
    plt.ylabel("Predicted salary")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    features, targets = load_dataset()
    avg_mse, final = run_kfold(features, targets)
    print("5 fold MSE:", round(avg_mse, 2))
    if final:
        plot_predictions(final[0], final[1])
