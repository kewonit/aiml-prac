"""Sales forecast using ad spend csv.
I grab a small slice of the file, pull ad spend, discount and clicks to guess revenue with 5-fold CV.
"""
import csv

FILE_PATH = "datasets/15 ad spends.csv"


def load_rows(limit=150):
    rows = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spend = float(row["Ad_Spend"])
            discount = float(row["Discount_Applied"])
            footfall = float(row["Clicks"])
            revenue = float(row["Revenue"])
            rows.append(([spend, discount, footfall], revenue))
            if len(rows) >= limit:
                break
    return rows


def gradient_descent(features, targets, steps=400, lr=0.00001):
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


def predict(w, row):
    out = w[0]
    for i, val in enumerate(row):
        out += w[i + 1] * val
    return out


def mse(weights, features, targets):
    return sum((predict(weights, row) - y) ** 2 for row, y in zip(features, targets)) / len(features)


def kfold(rows, k=5):
    fold_size = len(rows) // k
    scores = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        test = rows[start:end]
        train = rows[:start] + rows[end:]
        x_train = [r[0] for r in train]
        y_train = [r[1] for r in train]
        x_test = [r[0] for r in test]
        y_test = [r[1] for r in test]
        w = gradient_descent(x_train, y_train)
        scores.append(mse(w, x_test, y_test))
    return sum(scores) / len(scores)


if __name__ == "__main__":
    data = load_rows()
    print("5 fold MSE:", round(kfold(data), 2))
