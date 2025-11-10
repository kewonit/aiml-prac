"""Naive Bayes from scratch for disease labels using symptom columns.
Counts are pulled straight from the csv; everything uses Laplace smoothing.
Train on first chunk, test on the next chunk, then show accuracy.
"""
import math
import pandas as pd
from sklearn.metrics import accuracy_score

FILE_PATH = "datasets/disease_diagnosis_16_17.csv"


def load_rows(limit=200):
    df = pd.read_csv(FILE_PATH)
    df = df.head(limit)
    rows = []
    for _, row in df.iterrows():
        feat = (row["Symptom_1"], row["Symptom_2"], row["Symptom_3"])
        label = row["Diagnosis"]
        rows.append((feat, label))
    return rows


def train(nb_rows):
    priors = {}
    cond = {}
    vocab = set()
    for feats, label in nb_rows:
        priors[label] = priors.get(label, 0) + 1
        bucket = cond.setdefault(label, {})
        for feat in feats:
            vocab.add(feat)
        for idx, feat in enumerate(feats):
            key = (idx, feat)
            bucket[key] = bucket.get(key, 0) + 1
    return priors, cond, vocab


def predict(example, priors, cond, vocab):
    total = sum(priors.values())
    best = None
    best_score = None
    for label, count in priors.items():
        score = math.log(count / total)
        bucket = cond[label]
        for idx, feat in enumerate(example):
            key = (idx, feat)
            hit = bucket.get(key, 0)
            score += math.log((hit + 1) / (priors[label] + len(vocab)))
        if best_score is None or score > best_score:
            best_score = score
            best = label
    return best


if __name__ == "__main__":
    rows = load_rows()
    split = len(rows) // 2
    train_rows = rows[:split]
    test_rows = rows[split:]
    priors, cond, vocab = train(train_rows)
    
    y_true = [label for _, label in test_rows]
    y_pred = [predict(feats, priors, cond, vocab) for feats, _ in test_rows]
    
    accuracy = accuracy_score(y_true, y_pred)
    print("Test accuracy:", round(accuracy, 3))
