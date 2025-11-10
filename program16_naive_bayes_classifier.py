"""Multinomial naive bayes for a toy spam dataset.
Messages are tiny strings; training counts words with laplace smoothing.
I print confusion matrix along with accuracy, precision, recall, F1.
"""

messages = [
    ("cheap meds available now", "spam"),
    ("team meeting tomorrow morning", "ham"),
    ("limited offer buy now", "spam"),
    ("project deadline moved", "ham"),
    ("earn cash fast", "spam"),
    ("lunch plans with boss", "ham"),
    ("claim your prize now", "spam"),
    ("schedule weekly sync", "ham"),
    ("new investment opportunity", "spam"),
    ("budget review notes", "ham")
]

split = int(len(messages) * 0.7)
train = messages[:split]
test = messages[split:]


def tokenize(text):
    return text.lower().split()


vocab = set()
class_counts = {}
word_counts = {}

for text, label in train:
    class_counts[label] = class_counts.get(label, 0) + 1
    words = tokenize(text)
    vocab.update(words)
    wc = word_counts.setdefault(label, {})
    for word in words:
        wc[word] = wc.get(word, 0) + 1


def predict(text):
    words = tokenize(text)
    total_docs = sum(class_counts.values())
    best_label = None
    best_score = None
    for label in class_counts:
        prior = class_counts[label] / total_docs
        score = 0.0
        for word in words:
            word_total = word_counts[label].get(word, 0)
            prob = (word_total + 1) / (sum(word_counts[label].values()) + len(vocab))
            score += math.log(prob)
        score += math.log(prior)
        if best_score is None or score > best_score:
            best_score = score
            best_label = label
    return best_label


import math

TP = FP = TN = FN = 0

for text, label in test:
    guess = predict(text)
    if guess == "spam" and label == "spam":
        TP += 1
    elif guess == "spam" and label == "ham":
        FP += 1
    elif guess == "ham" and label == "ham":
        TN += 1
    else:
        FN += 1

accuracy = (TP + TN) / len(test)
precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0


if __name__ == "__main__":
    print("Confusion matrix (TP, FP, TN, FN):", TP, FP, TN, FN)
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
