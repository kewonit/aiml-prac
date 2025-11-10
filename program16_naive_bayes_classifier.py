"""Multinomial naive bayes that actually touches the giant email csv.
I cherry pick a few columns (free, money, etc.), learn simple counts with Laplace,
measure the metrics, and plot the confusion matrix as bars.
"""

import csv
import math
import matplotlib.pyplot as plt

FILE_PATH = "datasets/emails_16_17_18_19.csv"
WORDS = ["free", "money", "enron", "viagra", "credit", "project", "meeting", "cash", "offer", "prize"]


def load_rows(limit=400):
    feats = []
    labels = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("Prediction")
            if label not in {"0", "1"}:
                continue
            vector = []
            ok = True
            for word in WORDS:
                try:
                    vector.append(float(row.get(word, 0)))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            feats.append(vector)
            labels.append(int(label))
            if len(feats) >= limit:
                break
    return feats, labels


def split_train_test(features, labels, ratio=0.7):
    cut = int(len(features) * ratio)
    return (features[:cut], labels[:cut]), (features[cut:], labels[cut:])


def train_nb(features, labels):
    priors = {}
    counts = {}
    totals = {}
    for label in set(labels):
        priors[label] = labels.count(label) / len(labels)
        counts[label] = [1.0 for _ in WORDS]
        totals[label] = float(len(WORDS))
    for vec, lab in zip(features, labels):
        for i, val in enumerate(vec):
            counts[lab][i] += val
            totals[lab] += val
    return priors, counts, totals


def predict_nb(vec, priors, counts, totals):
    best_class = None
    best_score = None
    for lab in priors:
        score = math.log(priors[lab])
        for i, val in enumerate(vec):
            prob = counts[lab][i] / totals[lab]
            if val > 0:
                score += val * math.log(prob)
        if best_score is None or score > best_score:
            best_score = score
            best_class = lab
    return best_class


def eval_model(features, labels, priors, counts, totals):
    TP = FP = TN = FN = 0
    for vec, lab in zip(features, labels):
        guess = predict_nb(vec, priors, counts, totals)
        if guess == 1 and lab == 1:
            TP += 1
        elif guess == 1 and lab == 0:
            FP += 1
        elif guess == 0 and lab == 0:
            TN += 1
        else:
            FN += 1
    total = len(labels) or 1
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (TP, FP, TN, FN), accuracy, precision, recall, f1


def show_confusion_bars(tp, fp, tn, fn):
    plt.figure(figsize=(5, 4))
    plt.bar(["TP", "FP", "TN", "FN"], [tp, fp, tn, fn], color=["green", "red", "blue", "orange"])
    plt.title("Naive Bayes Confusion Counts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    feats, labels = load_rows()
    (train_X, train_y), (test_X, test_y) = split_train_test(feats, labels)
    priors, counts, totals = train_nb(train_X, train_y)
    (TP, FP, TN, FN), acc, prec, rec, f1 = eval_model(test_X, test_y, priors, counts, totals)

    print("Confusion matrix (TP, FP, TN, FN):", TP, FP, TN, FN)
    print("Accuracy:", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall:", round(rec, 3))
    print("F1:", round(f1, 3))

    show_confusion_bars(TP, FP, TN, FN)
