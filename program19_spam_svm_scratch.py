"""
binary classification of emails using svm from scratch.
shows hinge-loss optimization and evaluates with proper metrics.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = Path("datasets/emails_16_17_18_19.csv")


def load_email_dataset(dataset_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    # just grab the word-count features and spam labels from csv
    df = pd.read_csv(dataset_file)
    email_word_features = df.drop(columns=["Email No.", "Prediction"]).to_numpy(dtype=float)
    spam_labels = df["Prediction"].map({0: -1, 1: 1}).to_numpy(dtype=int)  # 0=normal, 1=spam
    return email_word_features, spam_labels


def split_train_test(features: np.ndarray, labels: np.ndarray, 
                     test_ratio: float = 0.2, seed: int = 11
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # shuffle and split into train and test sets
    rng = np.random.default_rng(seed)
    shuffled_indices = np.arange(len(features))
    rng.shuffle(shuffled_indices)
    split_point = int(len(shuffled_indices) * (1 - test_ratio))
    train_idx, test_idx = shuffled_indices[:split_point], shuffled_indices[split_point:]
    return (features[train_idx], labels[train_idx]), (features[test_idx], labels[test_idx])


def standardize_features(train_features: np.ndarray, test_features: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    # normalize by subtracting mean and dividing by std dev
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    train_normalized = (train_features - feature_mean) / feature_std
    test_normalized = (test_features - feature_mean) / feature_std
    return train_normalized, test_normalized


def train_svm_classifier(training_data: list, epochs: int = 35, 
                         learning_rate: float = 0.0008, 
                         lambda_reg: float = 0.01) -> Tuple[np.ndarray, float]:
    # initialize weight vector and bias to zero, then update using hinge loss
    num_features = training_data[0][0].shape[0]
    svm_weights = np.zeros(num_features)
    svm_bias = 0.0
    
    for _ in range(epochs):
        np.random.shuffle(training_data)
        for email_features, true_label in training_data:
            # margin tells us if we're right or how wrong we are
            margin = true_label * (np.dot(svm_weights, email_features) + svm_bias)
            # if margin < 1, we misclassify or barely get it right, so update
            if margin < 1:
                svm_weights = (1 - learning_rate * lambda_reg) * svm_weights + learning_rate * true_label * email_features
                svm_bias += learning_rate * true_label
            else:
                svm_weights = (1 - learning_rate * lambda_reg) * svm_weights
    return svm_weights, svm_bias


def predict_email_class(svm_weights: np.ndarray, svm_bias: float, 
                        email_features: np.ndarray) -> int:
    # if decision function >= 0, predict spam (1), else normal (-1)
    return 1 if np.dot(svm_weights, email_features) + svm_bias >= 0 else -1


def compute_metrics(test_data: list, svm_weights: np.ndarray, 
                    svm_bias: float) -> Tuple[float, float, float, float]:
    # count true positives, false positives, true negatives, false negatives
    tp = fp = tn = fn = 0
    for email_features, true_label in test_data:
        pred = predict_email_class(svm_weights, svm_bias, email_features)
        if pred == 1 and true_label == 1:
            tp += 1
        elif pred == 1 and true_label == -1:
            fp += 1
        elif pred == -1 and true_label == -1:
            tn += 1
        else:
            fn += 1
    
    # calculate accuracy, precision, recall, f1-score
    total_samples = tp + fp + tn + fn
    accuracy = (tp + tn) / total_samples if total_samples else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, f1_score


def visualize_performance(metrics: dict):
    # plot confusion matrix and metrics bar chart
    conf_matrix = np.array([[metrics['tn'], metrics['fp']], 
                            [metrics['fn'], metrics['tp']]])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Normal', 'Spam'], yticklabels=['Normal', 'Spam'])
    ax1.set_title('confusion matrix')
    ax1.set_ylabel('actual')
    ax1.set_xlabel('predicted')
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1-score']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    ax2.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylim([0, 1])
    ax2.set_title('svm performance metrics')
    ax2.set_ylabel('score')
    
    plt.tight_layout()
    plt.savefig('spam_detection_results.png', dpi=100, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # load and preprocess data
    email_features, spam_labels = load_email_dataset(DATASET_PATH)
    (train_features, train_labels), (test_features, test_labels) = split_train_test(email_features, spam_labels)
    train_features, test_features = standardize_features(train_features, test_features)
    
    # prepare data format and train svm
    train_data = [(feat, lbl) for feat, lbl in zip(train_features, train_labels)]
    test_data = [(feat, lbl) for feat, lbl in zip(test_features, test_labels)]
    
    svm_weights, svm_bias = train_svm_classifier(train_data)
    accuracy, precision, recall, f1 = compute_metrics(test_data, svm_weights, svm_bias)
    
    # count true/false positives/negatives for visualization
    tp = fp = tn = fn = 0
    for email_features, true_label in test_data:
        pred = predict_email_class(svm_weights, svm_bias, email_features)
        if pred == 1 and true_label == 1:
            tp += 1
        elif pred == 1 and true_label == -1:
            fp += 1
        elif pred == -1 and true_label == -1:
            tn += 1
        else:
            fn += 1
    
    # display results
    print("=" * 50)
    print("email spam detection - svm from scratch")
    print("=" * 50)
    print(f"training set size: {len(train_data)}")
    print(f"test set size: {len(test_data)}")
    print("-" * 50)
    print(f"accuracy:  {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall:    {recall:.3f}")
    print(f"f1-score:  {f1:.3f}")
    print("=" * 50)
    
    # visualize results
    metrics = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 
               'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    visualize_performance(metrics)
