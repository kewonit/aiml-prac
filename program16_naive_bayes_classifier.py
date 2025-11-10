"""
email spam detection using multinomial naive bayes.
trains on word frequencies, predicts spam vs legit, then shows metrics.
"""

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATASET = "datasets/emails_16_17_18_19.csv"
# email words that hint if something's spam
FEATURES = ["free", "money", "credit", "cash", "prize"]


def load_email_data(max_rows=500):
    # grab email features and labels from csv
    X, y = [], []
    with open(DATASET) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("Prediction")
            if label not in {"0", "1"}:
                continue
            try:
                # extract word frequencies for each feature
                features_vec = [float(row.get(word, 0)) for word in FEATURES]
                X.append(features_vec)
                y.append(int(label))
                if len(X) >= max_rows:
                    break
            except ValueError:
                continue
    return np.array(X), np.array(y)


def split_data(X, y, train_ratio=0.7):
    # split into train/test sets
    idx = int(len(X) * train_ratio)
    return (X[:idx], y[:idx]), (X[idx:], y[idx:])


def train_naive_bayes(X_train, y_train):
    # calculate class priors: P(spam) and P(not spam)
    class_priors = {}
    word_counts = {}
    word_totals = {}
    
    # initialize laplace smoothing: add 1 to avoid log(0)
    for cls in np.unique(y_train):
        class_priors[cls] = np.sum(y_train == cls) / len(y_train)
        word_counts[cls] = np.ones(len(FEATURES))
        word_totals[cls] = float(len(FEATURES))
    
    # count word occurrences per class
    for feature_vec, label in zip(X_train, y_train):
        for i, freq in enumerate(feature_vec):
            word_counts[label][i] += freq
            word_totals[label] += freq
    
    return class_priors, word_counts, word_totals


def predict_spam(feature_vec, class_priors, word_counts, word_totals):
    # pick class with highest log probability
    best_class = None
    best_score = float('-inf')
    
    for cls in class_priors:
        # start with log of class prior probability
        score = math.log(class_priors[cls])
        # add up log probabilities for each word
        for i, freq in enumerate(feature_vec):
            word_prob = word_counts[cls][i] / word_totals[cls]
            if freq > 0:
                score += freq * math.log(word_prob)
        if score > best_score:
            best_score = score
            best_class = cls
    
    return best_class


def evaluate_model(X_test, y_test, class_priors, word_counts, word_totals):
    # get predictions and calculate metrics
    y_pred = [predict_spam(vec, class_priors, word_counts, word_totals) for vec in X_test]
    
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    # standard metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return y_pred, (tp, fp, tn, fn), accuracy, precision, recall, f1


def plot_confusion_matrix(y_true, y_pred):
    # seaborn heatmap for confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Naive Bayes Email Spam Detection')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load data
    X, y = load_email_data()
    (X_train, y_train), (X_test, y_test) = split_data(X, y)
    
    # train and predict
    priors, counts, totals = train_naive_bayes(X_train, y_train)
    y_pred, (tp, fp, tn, fn), acc, prec, rec, f1 = evaluate_model(
        X_test, y_test, priors, counts, totals
    )
    
    # show results
    print("\n=== Email Spam Detection Results ===")
    print(f"accuracy:  {acc:.3f}")
    print(f"precision: {prec:.3f}")
    print(f"recall:    {rec:.3f}")
    print(f"f1-score:  {f1:.3f}")
    print(f"\nconfusion matrix -> tp:{tp} fp:{fp} tn:{tn} fn:{fn}")
    
    plot_confusion_matrix(y_test, y_pred)
