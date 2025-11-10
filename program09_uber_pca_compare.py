from __future__ import annotations

"""Uber fare predictor using pandas and scikit-learn helpers.
We engineer coordinate jumps, plot them, and compare LinearRegression
with and without a 1D PCA projection on a held-out split.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FILE_PATH = Path("datasets/uber_9_10.csv")


def load_features(limit: int | None = 400):
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
    features = df[["lng_jump", "lat_jump"]]
    target = df["fare_amount"].astype(float)
    return features, target, df


def eda_summary(df: pd.DataFrame) -> None:
    stats = df[["fare_amount", "lng_jump", "lat_jump"]].describe().loc[["count", "mean", "std", "min", "max"]]
    print("EDA snapshot (fare + rough jumps):")
    print(stats.round(3))


def scatter_plot(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="lng_jump", y="fare_amount", alpha=0.5, label="Longitude jump")
    sns.scatterplot(data=df, x="lat_jump", y="fare_amount", alpha=0.5, label="Latitude jump")
    plt.title("Uber fare vs coordinate jump")
    plt.xlabel("Coordinate jump (degrees)")
    plt.ylabel("Fare amount")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    direct_model = Pipeline([
        ("scale", StandardScaler()),
        ("reg", LinearRegression()),
    ])

    pca_model = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=1)),
        ("reg", LinearRegression()),
    ])

    direct_model.fit(X_train, y_train)
    pca_model.fit(X_train, y_train)

    direct_mae = mean_absolute_error(y_test, direct_model.predict(X_test))
    pca_mae = mean_absolute_error(y_test, pca_model.predict(X_test))
    return direct_mae, pca_mae


if __name__ == "__main__":
    X, y, raw_df = load_features()
    eda_summary(raw_df)
    scatter_plot(raw_df)
    plain_mae, pca_mae = evaluate_models(X, y)
    print("MAE without PCA:", round(plain_mae, 3))
    print("MAE with 1D PCA:", round(pca_mae, 3))
