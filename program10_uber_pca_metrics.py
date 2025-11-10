"""Uber metrics run that mirrors the sample assignment but stays barebones.
Loads the csv, squeezes coordinates with PCA, compares metrics, and plots
the predictions so we can see how off they are.
"""

import csv
import math
import matplotlib.pyplot as plt

FILE_PATH = "datasets/uber_9_10.csv"


def mean(vals):
    return sum(vals) / len(vals)


def load_rows(limit=250):
    rows = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fare = float(row["fare_amount"])
                px = float(row["pickup_longitude"])
                py = float(row["pickup_latitude"])
                dx = float(row["dropoff_longitude"])
                dy = float(row["dropoff_latitude"])
            except (ValueError, KeyError):
                continue
            if fare <= 0:
                continue
            run = abs(dx - px)
            rise = abs(dy - py)
            rows.append(([run, rise], fare))
            if len(rows) >= limit:
                break
    return rows


def normalize(vecs):
    cols = list(zip(*vecs))
    means = [mean(col) for col in cols]
    centered = []
    for row in vecs:
        centered.append([row[i] - means[i] for i in range(len(row))])
    return centered, means


def covariance_2d(centered):
    xs = [row[0] for row in centered]
    ys = [row[1] for row in centered]
    n = len(centered)
    if n < 2:
        return 0.0, 0.0, 0.0
    xx = sum(x * x for x in xs) / (n - 1)
    yy = sum(y * y for y in ys) / (n - 1)
    xy = sum(xs[i] * ys[i] for i in range(n)) / (n - 1)
    return xx, xy, yy


def principal_component(centered):
    a, b, c = covariance_2d(centered)
    trace = a + c
    det = a * c - b * b
    disc = max(trace * trace - 4 * det, 0)
    eig1 = (trace + math.sqrt(disc)) / 2
    if b != 0:
        vec = (eig1 - c, b)
    elif a >= c:
        vec = (1, 0)
    else:
        vec = (0, 1)
    length = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) or 1
    return (vec[0] / length, vec[1] / length)


def project(centered, vec):
    return [[row[0] * vec[0] + row[1] * vec[1]] for row in centered]


def gradient_descent(features, targets, steps=600, lr=0.4):
    weights = [0.0 for _ in range(len(features[0]) + 1)]
    for _ in range(steps):
        grad = [0.0 for _ in weights]
        for row, y in zip(features, targets):
            pred = weights[0]
            for i, val in enumerate(row):
                pred += weights[i + 1] * val
            error = pred - y
            grad[0] += error
            for i, val in enumerate(row):
                grad[i + 1] += error * val
        m = len(features)
        if m == 0:
            return weights
        for i in range(len(weights)):
            weights[i] -= lr * grad[i] / m
    return weights


def predict(weights, row):
    out = weights[0]
    for i, val in enumerate(row):
        out += weights[i + 1] * val
    return out


def metrics(weights, features, targets):
    if not targets:
        return 0.0, 0.0, 0.0
    preds = [predict(weights, row) for row in features]
    errors = [preds[i] - targets[i] for i in range(len(targets))]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    avg_y = mean(targets)
    ss_tot = sum((y - avg_y) ** 2 for y in targets)
    ss_res = sum(e * e for e in errors)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return mae, rmse, r2, preds


def show_pred_plot(actual, preds):
    plt.figure(figsize=(6, 4))
    plt.scatter(actual, preds, alpha=0.5, color="navy")
    line_min = min(actual + preds)
    line_max = max(actual + preds)
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="red")
    plt.title("Uber Fare Predictions")
    plt.xlabel("Actual Fare")
    plt.ylabel("Predicted Fare")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = load_rows()
    X = [feat for feat, _ in dataset]
    y = [fare for _, fare in dataset]

    centered, _ = normalize(X)
    raw_w = gradient_descent(centered, y)
    raw_mae, raw_rmse, raw_r2, raw_preds = metrics(raw_w, centered, y)

    vec = principal_component(centered)
    reduced = project(centered, vec)
    pca_w = gradient_descent(reduced, y)
    pca_mae, pca_rmse, pca_r2, pca_preds = metrics(pca_w, reduced, y)

    print("Raw features -> MAE, RMSE, R2:", tuple(round(x, 3) for x in (raw_mae, raw_rmse, raw_r2)))
    print("PCA feature -> MAE, RMSE, R2:", tuple(round(x, 3) for x in (pca_mae, pca_rmse, pca_r2)))

    show_pred_plot(y, raw_preds)
