"""IT salary guesser trained on the provided spreadsheet (converted to csv).
I keep it basic: experience, age, job, education and gender all become numeric.
Then I run goofy gradient descent and plot how the predictions line up.
"""

import csv
import matplotlib.pyplot as plt

FILE_PATH = "datasets/salary_data_14_converted.csv"


def encode(cache, key):
    key = key.strip().lower()
    if key not in cache:
        cache[key] = float(len(cache))
    return cache[key]


def load_rows(limit=180):
    rows = []
    edu_map = {}
    job_map = {}
    gender_map = {}
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                age = float(row["Age"]) if row["Age"] else None
                exp = float(row["Years of Experience"]) if row["Years of Experience"] else None
                salary = float(row["Salary"]) if row["Salary"] else None
            except ValueError:
                continue
            if age is None or exp is None or salary is None:
                continue
            edu = encode(edu_map, row.get("Education Level", "unknown"))
            job = encode(job_map, row.get("Job Title", "other"))
            gender = encode(gender_map, row.get("Gender", "missing"))
            feat = [age, exp, edu, job, gender]
            rows.append((feat, salary))
            if len(rows) >= limit:
                break
    return rows


def gradient_descent(features, targets, steps=650, lr=0.0000005):
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
    rows = load_rows()
    fold_size = len(rows) // k
    scores = []
    last_fold = None
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
        last_fold = (w, x_test, y_test)
    return sum(scores) / len(scores), last_fold


def plot_predictions(w, feats, targets):
    preds = [predict(w, row) for row in feats]
    plt.figure(figsize=(6, 4))
    plt.scatter(targets, preds, alpha=0.5, color="purple")
    line_min = min(targets + preds)
    line_max = max(targets + preds)
    plt.plot([line_min, line_max], [line_min, line_max], color="black", linestyle="--")
    plt.title("Predicted vs Actual Salary")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    avg_mse, final = kfold()
    print("5 fold MSE:", round(avg_mse, 2))
    if final:
        plot_predictions(final[0], final[1], final[2])
