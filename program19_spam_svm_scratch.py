"""SVM from scratch without shortcuts for spam detection.
Dataset is tiny counts for keywords and links. Hinge loss gradient descent updates weights.
"""
import random

samples = [
    ([5, 2], 1),
    ([4, 1], 1),
    ([0, 1], -1),
    ([1, 0], -1),
    ([6, 3], 1),
    ([2, 0], -1),
    ([7, 2], 1),
    ([3, 0], -1),
    ([8, 3], 1),
    ([0, 0], -1)
]

random.shuffle(samples)

split = int(len(samples) * 0.8)
train = samples[:split]
test = samples[split:]


def train_svm(rows, steps=800, lr=0.01, reg=0.01):
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
    return 1 if weights[0] * feats[0] + weights[1] * feats[1] + bias >= 0 else -1


def evaluate(rows, weights, bias):
    TP = FP = TN = FN = 0
    for feats, label in rows:
        guess = predict(weights, bias, feats)
        if guess == 1 and label == 1:
            TP += 1
        elif guess == 1 and label == -1:
            FP += 1
        elif guess == -1 and label == -1:
            TN += 1
        else:
            FN += 1
    accuracy = (TP + TN) / len(rows)
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    return accuracy, precision, recall


if __name__ == "__main__":
    w, b = train_svm(train)
    accuracy, precision, recall = evaluate(test, w, b)
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
