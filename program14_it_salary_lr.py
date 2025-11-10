"""Fake IT salary predictor using three quick features.
Data is totally made up but shaped like experience, education tier, and skill score.
I do 5-fold CV and print the mean squared error.
"""

data = [
    {"years": 1, "education": 0, "skill": 4, "salary": 32000},
    {"years": 3, "education": 1, "skill": 6, "salary": 45000},
    {"years": 5, "education": 1, "skill": 7, "salary": 52000},
    {"years": 2, "education": 0, "skill": 5, "salary": 36000},
    {"years": 7, "education": 2, "skill": 8, "salary": 72000},
    {"years": 4, "education": 1, "skill": 6, "salary": 48000},
    {"years": 6, "education": 2, "skill": 9, "salary": 68000},
    {"years": 8, "education": 2, "skill": 9, "salary": 78000},
    {"years": 10, "education": 2, "skill": 10, "salary": 88000},
    {"years": 3, "education": 0, "skill": 5, "salary": 41000},
    {"years": 9, "education": 2, "skill": 9, "salary": 82000},
    {"years": 1, "education": 0, "skill": 3, "salary": 30000}
]


def gradient_descent(features, targets, steps=600, lr=0.00001):
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
    ans = w[0]
    for i, val in enumerate(row):
        ans += w[i + 1] * val
    return ans


def mse(weights, features, targets):
    return sum((predict(weights, row) - y) ** 2 for row, y in zip(features, targets)) / len(features)


def kfold(k=5):
    rows = [([item["years"], item["education"], item["skill"]], item["salary"]) for item in data]
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
    print("5 fold MSE:", round(kfold(), 2))
