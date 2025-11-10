"""Mini uber price regressor.
First part prints a couple EDA stats so the dataset doesn’t feel mysterious.
Then I train a dumb gradient descent model on raw features and on a PCA squished feature.
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


def mean(values):
    return sum(values) / len(values)


def eda_summary():
    prices = [r["price"] for r in rides]
    print("Avg price:", round(mean(prices), 2))
    print("Min price:", round(min(prices), 2))
    print("Max price:", round(max(prices), 2))


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
    y = weights[0]
    for i, val in enumerate(row):
        y += weights[i + 1] * val
    return y


def mae(weights, features, targets):
    errors = [abs(predict(weights, row) - y) for row, y in zip(features, targets)]
    return sum(errors) / len(errors)


if __name__ == "__main__":
    eda_summary()
    X = [[r["distance"], r["minutes"]] for r in rides]
    y = [r["price"] for r in rides]
    centered, means = normalize(X)
    raw_weights = gradient_descent(centered, y)
    raw_mae = mae(raw_weights, centered, y)

    vec = principal_component(centered)
    reduced = project(centered, vec)
    reduced_weights = gradient_descent(reduced, y)
    reduced_mae = mae(reduced_weights, reduced, y)

    print("Raw feature MAE:", round(raw_mae, 3))
    print("PCA feature MAE:", round(reduced_mae, 3))
