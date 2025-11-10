"""Second uber price run where I spit out MAE, RMSE and R2.
Same toy dataset, but I only keep one PCA component to show off the metrics.
"""
import math

rides = [
    {"distance": 3.2, "minutes": 10, "price": 8.5},
    {"distance": 5.0, "minutes": 16, "price": 12.4},
    {"distance": 1.8, "minutes": 6, "price": 5.9},
    {"distance": 7.3, "minutes": 21, "price": 15.6},
    {"distance": 4.1, "minutes": 14, "price": 10.3},
    {"distance": 9.0, "minutes": 25, "price": 19.9},
    {"distance": 2.5, "minutes": 8, "price": 7.1},
    {"distance": 6.4, "minutes": 20, "price": 14.8},
    {"distance": 3.9, "minutes": 12, "price": 9.6},
    {"distance": 8.2, "minutes": 23, "price": 17.2}
]


def mean(vals):
    return sum(vals) / len(vals)


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


def gradient_descent(features, targets, steps=600, lr=0.01):
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
        for i in range(len(weights)):
            weights[i] -= lr * grad[i] / m
    return weights


def predict(weights, row):
    out = weights[0]
    for i, val in enumerate(row):
        out += weights[i + 1] * val
    return out


def metrics(weights, features, targets):
    preds = [predict(weights, row) for row in features]
    errors = [preds[i] - targets[i] for i in range(len(targets))]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    avg_y = mean(targets)
    ss_tot = sum((y - avg_y) ** 2 for y in targets)
    ss_res = sum(e * e for e in errors)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0
    return mae, rmse, r2


if __name__ == "__main__":
    X = [[r["distance"], r["minutes"]] for r in rides]
    y = [r["price"] for r in rides]
    centered, _ = normalize(X)
    raw_w = gradient_descent(centered, y)
    raw_scores = metrics(raw_w, centered, y)

    vec = principal_component(centered)
    reduced = project(centered, vec)
    pca_w = gradient_descent(reduced, y)
    pca_scores = metrics(pca_w, reduced, y)

    print("Raw features -> MAE, RMSE, R2:", tuple(round(x, 3) for x in raw_scores))
    print("PCA feature -> MAE, RMSE, R2:", tuple(round(x, 3) for x in pca_scores))
