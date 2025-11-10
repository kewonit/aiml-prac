"""House price regressor with baby K-fold cross validation.
I read a handful of rows from the csv, encode location crudely, run gradient descent, and average fold scores.
"""
import csv

FILE_PATH = "datasets/House_Price_Dataset 11.csv"


def grab_rows(limit=80):
    rows = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            area = float(row["Area"])
            beds = float(row["Bedrooms"])
            loc = row["Location"]
            price = float(row["Price"])
            loc_num = {"Rural": 0.0, "Suburban": 1.0, "Urban": 2.0}.get(loc, 1.0)
            rows.append(([area, beds, loc_num], price))
            if len(rows) >= limit:
                break
    return rows


def gradient_descent(features, targets, steps=400, lr=0.00000001):
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


def predict(w, x):
    out = w[0]
    for i, val in enumerate(x):
        out += w[i + 1] * val
    return out


def mse(weights, features, targets):
    errors = [(predict(weights, row) - y) ** 2 for row, y in zip(features, targets)]
    return sum(errors) / len(errors)


def kfold(rows, k=4):
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
    data = grab_rows()
    score = kfold(data, k=4)
    print("Average MSE over folds:", round(score, 2))
