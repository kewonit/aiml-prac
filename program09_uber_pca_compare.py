"""Uber fare predictor that actually reads the csv from the repo.
I use basic longitude/latitude jumps as features, shrink them with PCA,
then compare mean absolute errors and throw up a scatter plot.
"""

import csv
import math
import matplotlib.pyplot as plt

FILE_PATH = "datasets/uber_9_10.csv"


def mean(values):
    return sum(values) / len(values)


def load_rows(limit=200):
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


def eda_summary(rows):
    fares = [fare for _, fare in rows]
    dists = [feat[0] + feat[1] for feat, _ in rows]
    print("Trips used:", len(rows))
    print("Avg fare:", round(mean(fares), 2))
    print("Avg rough distance (deg):", round(mean(dists), 4))
    print("Min/Max fare:", round(min(fares), 2), round(max(fares), 2))


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


def gradient_descent(features, targets, steps=600, lr=0.5):
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


def mae(weights, features, targets):
    if not features:
        return 0.0
    errors = [abs(predict(weights, row) - y) for row, y in zip(features, targets)]
    return sum(errors) / len(errors)


def plot_scatter(rows):
    runs = [feat[0] for feat, _ in rows]
    rises = [feat[1] for feat, _ in rows]
    fares = [fare for _, fare in rows]
    plt.figure(figsize=(6, 4))
    plt.scatter(runs, fares, alpha=0.5, c="teal", label="longitude jump")
    plt.scatter(rises, fares, alpha=0.5, c="orange", label="latitude jump")
    plt.xlabel("Rough distance component (degrees)")
    plt.ylabel("Fare ($)")
    plt.title("Uber Fare vs Coordinate Jump")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = load_rows()
    eda_summary(dataset)
    plot_scatter(dataset)

    X = [feat for feat, _ in dataset]
    y = [fare for _, fare in dataset]
    centered, _ = normalize(X)
    raw_weights = gradient_descent(centered, y)
    raw_mae = mae(raw_weights, centered, y)

    vec = principal_component(centered)
    reduced = project(centered, vec)
    reduced_weights = gradient_descent(reduced, y)
    reduced_mae = mae(reduced_weights, reduced, y)

    print("Raw feature MAE:", round(raw_mae, 3))
    print("PCA feature MAE:", round(reduced_mae, 3))
