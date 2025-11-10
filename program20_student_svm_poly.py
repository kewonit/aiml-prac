"""Kernelized SVM-ish classifier for student pass/fail.
I treat it like a perceptron with a polynomial kernel (degree 2) because coding SMO is pain.
"""
import csv

FILE_PATH = "datasets/student_performance_dataset_20.csv"


def load_rows(limit=160):
    rows = []
    with open(FILE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feats = [
                float(row["Study_Hours_per_Week"]),
                float(row["Attendance_Rate"]),
                float(row["Internal_Scores"])
            ]
            label = 1 if row["Pass_Fail"].strip().lower() == "pass" else -1
            rows.append((feats, label))
            if len(rows) >= limit:
                break
    return rows


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def poly_kernel(a, b, degree=2):
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


def predict(rows, alphas, x):
    score = 0
    for alpha, (x_i, y_i) in zip(alphas, rows):
        if alpha:
            score += alpha * y_i * poly_kernel(x_i, x)
    return 1 if score >= 0 else -1


if __name__ == "__main__":
    data = load_rows()
    split = int(len(data) * 0.7)
    train = data[:split]
    test = data[split:split + 40]
    alphas = train_kernel_perceptron(train)

    TP = FP = TN = FN = 0
    for feats, label in test:
        guess = predict(train, alphas, feats)
        if guess == 1 and label == 1:
            TP += 1
        elif guess == 1 and label == -1:
            FP += 1
        elif guess == -1 and label == -1:
            TN += 1
        else:
            FN += 1
    accuracy = (TP + TN) / len(test) if test else 0
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
