from __future__ import annotations

"""Multinomial Naive Bayes email classifier built with scikit-learn utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

FILE_PATH = Path("datasets/emails_16_17_18_19.csv")
WORDS = [
    "free",
    "money",
    "enron",
    "credit",
    "project",
    "meeting",
    "cash",
    "offer",
    "prize",
]


def load_dataset(limit: int | None = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix and labels, optionally truncating for quick runs."""
    df = pd.read_csv(FILE_PATH, usecols=WORDS + ["Prediction"])
    df = df.dropna()
    df = df[df["Prediction"].isin([0, 1])]
    if limit:
        df = df.head(limit)
    X = df[WORDS].astype(float)
    y = df["Prediction"].astype(int)
    return X, y


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.3,
    seed: int = 10,
) -> tuple[float, str, pd.Series, pd.Series]:
    """Split the data, fit a Naive Bayes model, and return accuracy metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(
        y_test, preds, target_names=["Normal", "Spam"], digits=3
    )
    acc = accuracy_score(y_test, preds)
    return acc, report, y_test, preds


def plot_confusion(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Display the confusion matrix for the predicted labels."""
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Normal", "Spam"], cmap="Blues"
    )
    plt.title("Naive Bayes confusion matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    features, labels = load_dataset()
    accuracy, report, actual, predicted = train_and_evaluate(features, labels)
    print("Accuracy:", round(accuracy, 3))
    print(report)
    plot_confusion(actual, predicted)
