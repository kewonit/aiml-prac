"""Polynomial kernel SVM-style classifier for breast cancer data.
Training uses a kernel perceptron (degree 3) and evaluation reports confusion matrix and a few ROC points.
"""
import csv

FILE_PATH = "datasets/Breast Cancer Wisconsin (Diagnostic)_21.csv"


def load_rows(limit=220):
    rows = []
    with open(FILE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [
                float(row["radius_mean"]),
                float(row["texture_mean"]),
                float(row["perimeter_mean"])
            ]
            label = 1 if row["diagnosis"].strip().upper() == "M" else -1
            rows.append((feats, label))
            if len(rows) >= limit:
                break
    return rows


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def poly_kernel(a, b, degree=3):
    return (dot(a, b) + 1) ** degree


def train_kernel_perceptron(rows, epochs=3):
    alphas = [0 for _ in rows]
    for _ in range(epochs):
        for idx, (x_i, y_i) in enumerate(rows):
            score = 0
            for j, (x_j, y_j) in enumerate(rows):
                if alphas[j]:
                    score += alphas[j] * y_j * poly_kernel(x_j, x_i)
            if y_i * score <= 0:
                alphas[idx] += 1
    return alphas


def score_point(rows, alphas, x):
    total = 0
    for alpha, (x_i, y_i) in zip(alphas, rows):
        if alpha:
            total += alpha * y_i * poly_kernel(x_i, x)
    return total


def predict(rows, alphas, x):
    return 1 if score_point(rows, alphas, x) >= 0 else -1


def confusion(test, rows, alphas):
    TP = FP = TN = FN = 0
    scores = []
    for feats, label in test:
        s = score_point(rows, alphas, feats)
        guess = 1 if s >= 0 else -1
        scores.append((s, label))
        if guess == 1 and label == 1:
            TP += 1
        elif guess == 1 and label == -1:
            FP += 1
        elif guess == -1 and label == -1:
            TN += 1
        else:
            FN += 1
    return (TP, FP, TN, FN), scores


def roc_points(scores, cuts=5):
    scores_sorted = sorted(scores, key=lambda x: x[0])
    thresholds = []
    if not scores_sorted:
        return []
    step = max(1, len(scores_sorted) // cuts)
    for idx in range(0, len(scores_sorted), step):
        thresholds.append(scores_sorted[idx][0])
    thresholds.append(scores_sorted[-1][0] + 1)
    out = []
    for t in thresholds:
        TP = FP = TN = FN = 0
        for score, label in scores:
            guess = 1 if score >= t else -1
            if guess == 1 and label == 1:
                TP += 1
            elif guess == 1 and label == -1:
                FP += 1
            elif guess == -1 and label == -1:
                TN += 1
            else:
                FN += 1
        TPR = TP / (TP + FN) if TP + FN else 0
        FPR = FP / (FP + TN) if FP + TN else 0
        out.append((round(t, 3), round(TPR, 3), round(FPR, 3)))
    return out


if __name__ == "__main__":
    data = load_rows()
    split = int(len(data) * 0.7)
    train = data[:split]
    test = data[split:split + 100]
    alphas = train_kernel_perceptron(train)
    cm, scores = confusion(test, train, alphas)
    print("Confusion matrix (TP, FP, TN, FN):", cm)
    for t, tpr, fpr in roc_points(scores):
        print("Threshold", t, "TPR", tpr, "FPR", fpr)
