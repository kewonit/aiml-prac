"""Naive Bayes from scratch for disease labels using symptom columns.
Counts are pulled straight from the csv; everything uses Laplace smoothing.
Train on first chunk, test on the next chunk, then show accuracy.
"""
import csv

FILE_PATH = "datasets/disease_diagnosis_16_17.csv"


def load_rows(limit=200):
    rows = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = (row["Symptom_1"], row["Symptom_2"], row["Symptom_3"])
            label = row["Diagnosis"]
            rows.append((feat, label))
            if len(rows) >= limit:
                break
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


import math

if __name__ == "__main__":
    rows = load_rows()
    split = len(rows) // 2
    train_rows = rows[:split]
    test_rows = rows[split:]
    priors, cond, vocab = train(train_rows)
    hits = 0
    for feats, label in test_rows:
        if predict(feats, priors, cond, vocab) == label:
            hits += 1
    accuracy = hits / len(test_rows)
    print("Test accuracy:", round(accuracy, 3))
