from __future__ import annotations

"""Uber fare PCA comparison with full error metrics and a diagnostic plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FILE_PATH = Path("datasets/uber_9_10.csv")


def load_features(limit: int | None = 600):
    df = pd.read_csv(FILE_PATH)
    df = df[df["fare_amount"] > 0].dropna(subset=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ])
    if limit:
        df = df.head(limit)
    df = df.assign(
        lng_jump=lambda d: (d["dropoff_longitude"] - d["pickup_longitude"]).abs(),
        lat_jump=lambda d: (d["dropoff_latitude"] - d["pickup_latitude"]).abs(),
    )
    return df[["lng_jump", "lat_jump"]], df["fare_amount"].astype(float)


def build_pipeline(use_pca: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=1)))
    steps.append(("reg", LinearRegression()))
    return Pipeline(steps)


def compare_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
    plain = build_pipeline(use_pca=False)
    pca = build_pipeline(use_pca=True)
    plain.fit(X_train, y_train)
    pca.fit(X_train, y_train)

    plain_preds = plain.predict(X_test)
    pca_preds = pca.predict(X_test)

    def collect_metrics(actual, preds):
        return (
            mean_absolute_error(actual, preds),
            root_mean_squared_error(actual, preds, squared=False),
            r2_score(actual, preds),
        )

    return (
        collect_metrics(y_test, plain_preds),
        collect_metrics(y_test, pca_preds),
        y_test,
        plain_preds,
    )


def show_pred_plot(actual, preds):
    plt.figure(figsize=(6, 4))
    plt.scatter(actual, preds, alpha=0.55, color="navy")
    diag_min = min(actual.min(), preds.min())
    diag_max = max(actual.max(), preds.max())
    plt.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="red")
    plt.xlabel("Actual fare")
    plt.ylabel("Predicted fare")
    plt.title("Uber fare predictions (no PCA model)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = load_features()
    plain_metrics, pca_metrics, actual, preds = compare_models(X, y)
    print("Without PCA (MAE, RMSE, R2):", tuple(round(x, 3) for x in plain_metrics))
    print("With PCA (MAE, RMSE, R2):", tuple(round(x, 3) for x in pca_metrics))
    show_pred_plot(actual, preds)
