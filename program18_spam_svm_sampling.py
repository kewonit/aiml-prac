"""
email spam detector using svm with oversampling to handle class imbalance.
maps emails as normal (not spam) or abnormal (spam) and shows performance metrics.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter


DATASET_PATH = Path("datasets/emails_16_17_18_19.csv")


# load the email dataset and convert labels to -1 (spam) and 1 (normal)
def load_emails(path):
    df = pd.read_csv(path)
    features = df.drop(columns=["Email No.", "Prediction"]).to_numpy(dtype=float)
    labels = df["Prediction"].map({0: -1, 1: 1}).to_numpy(dtype=int)
    return features, labels


# basically just copy-pasting the minority class until it matches the majority
def oversample_minority_class(email_features, email_labels, seed=7):
    label_counts = Counter(email_labels)
    if len(label_counts) < 2:
        return email_features, email_labels
    
    (majority_label, majority_count), (minority_label, minority_count) = sorted(
        label_counts.items(), key=lambda x: x[1], reverse=True
    )
    if majority_count == minority_count:
        return email_features, email_labels
    
    # grab all the minority class examples and duplicate them randomly
    minority_mask = np.where(email_labels == minority_label)[0]
    np.random.seed(seed)
    duplicated_indices = np.random.choice(minority_mask, size=majority_count - minority_count, replace=True)
    
    extra_features = email_features[duplicated_indices]
    extra_labels = email_labels[duplicated_indices]
    
    balanced_features = np.vstack((email_features, extra_features))
    balanced_labels = np.concatenate((email_labels, extra_labels))
    
    shuffle_order = np.random.permutation(len(balanced_features))
    return balanced_features[shuffle_order], balanced_labels[shuffle_order]


# the actual svm training using hinge loss. weights learn by seeing if margin < 1
def train_svm_classifier(training_data, epochs=35, learning_rate=0.0008, regularization=0.01):
    # initialize weights and bias to zero
    num_features = training_data[0][0].shape[0]
    svm_weights = np.zeros(num_features)
    svm_bias = 0.0
    
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        for email_vector, true_label in training_data:
            # check how far we are from the decision boundary
            decision_score = true_label * (np.dot(svm_weights, email_vector) + svm_bias)
            
            # if we're not confident enough, update weights using hinge loss
            if decision_score < 1:
                svm_weights = (1 - learning_rate * regularization) * svm_weights + learning_rate * true_label * email_vector
                svm_bias += learning_rate * true_label
            else:
                # just apply regularization to reduce magnitude
                svm_weights = (1 - learning_rate * regularization) * svm_weights
    
    return svm_weights, svm_bias


# predict if email is spam (-1) or normal (1)
def predict_spam(svm_weights, svm_bias, email_vector):
    return 1 if np.dot(svm_weights, email_vector) + svm_bias >= 0 else -1


# calculate accuracy, precision, recall, and f1 score
def evaluate_model(test_data, svm_weights, svm_bias):
    tp = fp = tn = fn = 0
    for email_vector, true_label in test_data:
        prediction = predict_spam(svm_weights, svm_bias, email_vector)
        if prediction == 1 and true_label == 1:
            tp += 1
        elif prediction == 1 and true_label == -1:
            fp += 1
        elif prediction == -1 and true_label == -1:
            tn += 1
        else:
            fn += 1
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    # load dataset
    X_all, y_all = load_emails(DATASET_PATH)
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=7)
    
    # normalize the data so all features have same scale
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    # handle class imbalance by oversampling spam emails
    X_balanced, y_balanced = oversample_minority_class(X_train_normalized, y_train)
    
    # convert to list of tuples for training
    training_set = [(feat, lbl) for feat, lbl in zip(X_balanced, y_balanced)]
    test_set = [(feat, lbl) for feat, lbl in zip(X_test_normalized, y_test)]
    
    # train the svm
    learned_weights, learned_bias = train_svm_classifier(training_set)
    
    # evaluate on test set
    acc, prec, rec, f1_score = evaluate_model(test_set, learned_weights, learned_bias)
    
    # show results
    print("\n--- spam detection results ---")
    print(f"original train distribution: {dict(Counter(y_train))}")
    print(f"after oversampling: {dict(Counter(y_balanced))}")
    print(f"\ntest distribution: {dict(Counter(y_test))}")
    print(f"\nmodel performance:")
    print(f"  accuracy:  {acc:.3f}")
    print(f"  precision: {prec:.3f}")
    print(f"  recall:    {rec:.3f}")
    print(f"  f1 score:  {f1_score:.3f}")
