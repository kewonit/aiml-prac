"""Linear SVM on a tiny email dataset with oversampling.
Features are made up counts (caps, links). I duplicate the minority spam rows before training.
After training I print accuracy, precision, recall.
"""
import random

raw = [
    ([1, 0], -1),
    ([2, 0], -1),
    ([0, 1], -1),
    ([3, 0], -1),
    ([4, 1], 1),
    ([5, 2], 1),
]

ham = [row for row in raw if row[1] == -1]
spam = [row for row in raw if row[1] == 1]

oversampled = ham + spam + spam  # clone spam samples to balance a bit
random.shuffle(oversampled)


def train_svm(rows, steps=600, lr=0.01, reg=0.01):
    weights = [0.0, 0.0]
    bias = 0.0
    for _ in range(steps):
        for feats, label in rows:
            margin = label * (weights[0] * feats[0] + weights[1] * feats[1] + bias)
            if margin < 1:
                weights[0] = weights[0] - lr * (reg * weights[0] - label * feats[0])
                weights[1] = weights[1] - lr * (reg * weights[1] - label * feats[1])
                bias = bias + lr * label
            else:
                weights[0] = weights[0] - lr * reg * weights[0]
                weights[1] = weights[1] - lr * reg * weights[1]
    return weights, bias


def predict(weights, bias, feats):
    score = weights[0] * feats[0] + weights[1] * feats[1] + bias
    return 1 if score >= 0 else -1


if __name__ == "__main__":
    weights, bias = train_svm(oversampled)
    TP = FP = TN = FN = 0
    for feats, label in raw:
        guess = predict(weights, bias, feats)
        if guess == 1 and label == 1:
            TP += 1
        elif guess == 1 and label == -1:
            FP += 1
        elif guess == -1 and label == -1:
            TN += 1
        else:
            FN += 1
    accuracy = (TP + TN) / len(raw)
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
