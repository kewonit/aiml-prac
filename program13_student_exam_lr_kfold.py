"""Multi feature student score regression with basic K-fold.
I use hours, attendance and internal marks from the csv, average the fold MSE,
and plot how the last fold predictions compare to reality.
"""
import csv
import matplotlib.pyplot as plt

FILE_PATH = "datasets/student_exam_scores_12_13.csv"


def load_rows(limit=120):
    rows = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = [
                float(row["hours_studied"]),
                float(row["attendance_percent"]),
                float(row["Internal_marks"])
            ]
            target = float(row["exam_score"])
            rows.append((feat, target))
            if len(rows) >= limit:
                break
    return rows


def gradient_descent(features, targets, steps=500, lr=0.0001):
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


def mse(weights, features, targets):
    return sum((predict(weights, row) - y) ** 2 for row, y in zip(features, targets)) / len(features)


def kfold(rows, k=5):
    fold_size = len(rows) // k
    scores = []
    last = None
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
        fold_score = mse(w, x_test, y_test)
        scores.append(fold_score)
        last = (w, x_test, y_test)
    return sum(scores) / len(scores), last


def plot_preds(w, feats, targets):
    preds = [predict(w, row) for row in feats]
    plt.figure(figsize=(6, 4))
    plt.scatter(targets, preds, alpha=0.6, color="darkorange")
    line_min = min(targets + preds)
    line_max = max(targets + preds)
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="black")
    plt.title("Student Score Predictions (Fold)")
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data = load_rows()
    mse_avg, final = kfold(data)
    print("5 fold MSE:", round(mse_avg, 3))
    if final:
        plot_preds(final[0], final[1], final[2])
