"""Kernelized SVM-ish classifier for student pass/fail.
I treat it like a perceptron with a polynomial kernel (degree 2) because coding SMO is pain.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FILE_PATH = "datasets/student_performance_dataset_20.csv"


def load_rows(limit=160):
    df = pd.read_csv(FILE_PATH, encoding="utf-8")
    df = df.head(limit)
    rows = []
    for _, row in df.iterrows():
        feats = [
            float(row["Study_Hours_per_Week"]),
            float(row["Attendance_Rate"]),
            float(row["Internal_Scores"])
        ]
        label = 1 if row["Pass_Fail"].strip().lower() == "pass" else -1
        rows.append((feats, label))
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

    y_true = [label for _, label in test]
    y_pred = [predict(train, alphas, feats) for feats, _ in test]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
