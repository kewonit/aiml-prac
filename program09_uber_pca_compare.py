"""Ride price prediction with PCA dimensionality reduction and model comparison.
Performs EDA, implements PCA manually, trains models, compares with/without PCA.
Supports Uber dataset and Iris dataset (classification -> regression via labels).
"""

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error, r2_score

FILE_PATH = "datasets/uber_9_10.csv"


def load_dataset(dataset_type="uber", limit=200):
    """Load Uber or Iris dataset. Returns list of ([features], target)."""
    if dataset_type == "iris":
        iris = load_iris()
        X, y = iris.data[:, :2], iris.target  # Use first 2 features, target as continuous
        return [([float(x[0]), float(x[1])], float(y_val)) for x, y_val in zip(X, y)][:limit]
    
    # Uber dataset
    rows = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fare = float(row["fare_amount"])
                px, py = float(row["pickup_longitude"]), float(row["pickup_latitude"])
                dx, dy = float(row["dropoff_longitude"]), float(row["dropoff_latitude"])
            except (ValueError, KeyError):
                continue
            if fare <= 0:
                continue
            rows.append(([abs(dx - px), abs(dy - py)], fare))
            if len(rows) >= limit:
                break
    return rows


def eda_summary(rows):
    """Print exploratory data analysis statistics."""
    fares = [fare for _, fare in rows]
    dists = [feat[0] + feat[1] for feat, _ in rows]
    print(f"Samples: {len(rows)} | Avg fare: {np.mean(fares):.2f} | "
          f"Distance: {np.mean(dists):.4f} | Fare range: [{min(fares):.2f}, {max(fares):.2f}]")


def normalize(vecs):
    """Center features to zero mean (manual standardization)."""
    vecs_arr = np.array(vecs)
    means = np.mean(vecs_arr, axis=0)
    centered = vecs_arr - means
    return centered.tolist(), means


def covariance_2d(centered):
    """Compute 2D covariance matrix elements manually."""
    xs = [row[0] for row in centered]
    ys = [row[1] for row in centered]
    n = len(centered) - 1
    if n < 1:
        return 0.0, 0.0, 0.0
    xx = sum(x * x for x in xs) / n
    yy = sum(y * y for y in ys) / n
    xy = sum(xs[i] * ys[i] for i in range(len(xs))) / n
    return xx, xy, yy


def principal_component(centered):
    """Compute eigenvector for PCA (manual eigendecomposition)."""
    a, b, c = covariance_2d(centered)
    trace, det = a + c, a * c - b * b
    disc = max(trace * trace - 4 * det, 0)
    eig1 = (trace + math.sqrt(disc)) / 2
    vec = (eig1 - c, b) if b != 0 else ((1, 0) if a >= c else (0, 1))
    length = math.sqrt(vec[0]**2 + vec[1]**2) or 1
    return (vec[0] / length, vec[1] / length)


def project(centered, vec):
    """Project data onto principal component."""
    return [[row[0] * vec[0] + row[1] * vec[1]] for row in centered]


def gradient_descent(X, y, steps=600, lr=0.5):
    """Manual gradient descent for linear regression."""
    w = [0.0] * (len(X[0]) + 1)
    for _ in range(steps):
        g = [0.0] * len(w)
        for x_i, y_i in zip(X, y):
            pred = w[0] + sum(w[i+1] * x_i[i] for i in range(len(x_i)))
            err = pred - y_i
            g[0] += err
            for i, val in enumerate(x_i):
                g[i+1] += err * val
        w = [w[i] - lr * g[i] / len(X) for i in range(len(w))]
    return w


def predict(w, x):
    """Predict using trained weights."""
    return w[0] + sum(w[i+1] * x[i] for i in range(len(x)))


def visualize_comparison(X, y, w_raw, w_pca, vec):
    """Visualize EDA scatter and model predictions comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # EDA scatter plot
    X_arr = np.array(X)
    axes[0].scatter(X_arr[:, 0], y, alpha=0.5, c="teal", label="Long diff")
    axes[0].scatter(X_arr[:, 1], y, alpha=0.5, c="orange", label="Lat diff")
    axes[0].set_xlabel("Distance component (deg)")
    axes[0].set_ylabel("Fare ($)")
    axes[0].set_title("EDA: Fare vs Features")
    axes[0].legend()
    
    # Model comparison
    centered, _ = normalize(X)
    pred_raw = [predict(w_raw, x) for x in centered]
    reduced = project(centered, vec)
    pred_pca = [predict(w_pca, x) for x in reduced]
    
    axes[1].scatter(range(len(y)), y, alpha=0.5, label="Actual", c="gray")
    axes[1].plot(pred_raw, label="Raw features", linestyle="--", linewidth=2)
    axes[1].plot(pred_pca, label="PCA features", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Fare ($)")
    axes[1].set_title("Model Predictions Comparison")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load dataset (change "uber" to "iris" to use Iris dataset)
    dataset = load_dataset(dataset_type="uber", limit=200)
    
    # EDA
    eda_summary(dataset)
    
    # Extract features and targets
    X = [feat for feat, _ in dataset]
    y = [fare for _, fare in dataset]
    
    # Raw model: train on original features
    centered, _ = normalize(X)
    w_raw = gradient_descent(centered, y)
    pred_raw = [predict(w_raw, x) for x in centered]
    mse_raw, r2_raw = mean_squared_error(y, pred_raw), r2_score(y, pred_raw)
    
    # PCA model: train on reduced features
    vec = principal_component(centered)
    reduced = project(centered, vec)
    w_pca = gradient_descent(reduced, y)
    pred_pca = [predict(w_pca, x) for x in reduced]
    mse_pca, r2_pca = mean_squared_error(y, pred_pca), r2_score(y, pred_pca)
    
    # Results
    print(f"\n{'='*50}")
    print(f"{'Raw Features (2D)':<25} MSE: {mse_raw:.4f} | R²: {r2_raw:.4f}")
    print(f"{'PCA Features (1D)':<25} MSE: {mse_pca:.4f} | R²: {r2_pca:.4f}")
    print(f"{'='*50}")
    
    visualize_comparison(X, y, w_raw, w_pca, vec)
