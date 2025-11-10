"""
svm classifier from scratch with polynomial kernel to tell if tumors are bad or not.
uses kernel perceptron instead of fancy optimizers but gets the job done
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# where the cancer data lives
data_file = "datasets/Breast Cancer Wisconsin (Diagnostic)_21.csv"


def load_cancer_data():
    """
    grabs the data and picks a few features to keep things simple.
    malignant = 1, benign = -1 cuz that's how svm likes it
    """
    df = pd.read_csv(data_file)
    # just using 3 features so this doesn't take forever
    feature_columns = ["radius_mean", "texture_mean", "perimeter_mean"]
    features = df[feature_columns].values
    # turn M into 1 and B into -1
    labels = np.where(df["diagnosis"] == "M", 1, -1)
    return features, labels


def polynomial_kernel(point_a, point_b, degree=3):
    """
    polynomial kernel trick so we don't actually transform the data.
    basically makes it so svm can handle curvy decision boundaries
    """
    return (np.dot(point_a, point_b) + 1) ** degree


def train_svm_scratch(train_features, train_labels, epochs=5):
    """
    kernel perceptron training which is basically budget svm.
    keeps track of which training points matter (alphas > 0)
    """
    num_samples = len(train_features)
    support_weights = np.zeros(num_samples)
    
    # go through the data a few times fixing mistakes
    for _ in range(epochs):
        for i in range(num_samples):
            # calculate decision score using kernel trick
            decision_score = 0
            for j in range(num_samples):
                if support_weights[j] > 0:
                    kernel_value = polynomial_kernel(train_features[j], train_features[i])
                    decision_score += support_weights[j] * train_labels[j] * kernel_value
            
            # if we got it wrong, bump up the weight
            if train_labels[i] * decision_score <= 0:
                support_weights[i] += 1
    
    return support_weights


def predict_sample(train_features, train_labels, support_weights, test_point):
    """
    makes a prediction for one sample by checking against all support vectors.
    positive score = malignant, negative = benign
    """
    score = 0
    for i in range(len(train_features)):
        if support_weights[i] > 0:
            kernel_value = polynomial_kernel(train_features[i], test_point)
            score += support_weights[i] * train_labels[i] * kernel_value
    return score


def get_confusion_matrix(test_features, test_labels, train_features, train_labels, support_weights):
    """
    makes predictions on test set and counts up tp, fp, tn, fn.
    also saves scores for roc curve later
    """
    true_positive = false_positive = true_negative = false_negative = 0
    prediction_scores = []
    
    for i in range(len(test_features)):
        score = predict_sample(train_features, train_labels, support_weights, test_features[i])
        predicted_label = 1 if score >= 0 else -1
        actual_label = test_labels[i]
        
        prediction_scores.append(score)
        
        # counting the outcomes
        if predicted_label == 1 and actual_label == 1:
            true_positive += 1
        elif predicted_label == 1 and actual_label == -1:
            false_positive += 1
        elif predicted_label == -1 and actual_label == -1:
            true_negative += 1
        else:
            false_negative += 1
    
    confusion = np.array([[true_positive, false_positive], 
                         [false_negative, true_negative]])
    return confusion, prediction_scores


def calculate_roc_curve(test_labels, prediction_scores):
    """
    tries different thresholds to see how tpr and fpr change.
    this is what makes the roc curve
    """
    # sort scores to get good threshold values
    sorted_scores = sorted(prediction_scores)
    thresholds = sorted_scores[::len(sorted_scores)//20] + [sorted_scores[-1] + 1]
    
    tpr_values = []
    fpr_values = []
    
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for score, actual in zip(prediction_scores, test_labels):
            predicted = 1 if score >= threshold else -1
            if predicted == 1 and actual == 1:
                tp += 1
            elif predicted == 1 and actual == -1:
                fp += 1
            elif predicted == -1 and actual == -1:
                tn += 1
            else:
                fn += 1
        
        # avoid division by zero
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    return fpr_values, tpr_values


def plot_results(confusion, fpr_list, tpr_list):
    """
    shows the confusion matrix and roc curve side by side
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # confusion matrix heatmap
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['malignant', 'benign'], 
                yticklabels=['malignant', 'benign'])
    ax1.set_xlabel('predicted')
    ax1.set_ylabel('actual')
    ax1.set_title('confusion matrix')
    
    # roc curve
    ax2.plot(fpr_list, tpr_list, 'b-', linewidth=2, label='svm classifier')
    ax2.plot([0, 1], [0, 1], 'r--', label='random guess')
    ax2.set_xlabel('false positive rate')
    ax2.set_ylabel('true positive rate')
    ax2.set_title('roc curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load up the data
    all_features, all_labels = load_cancer_data()
    
    # split into train and test
    split_point = int(len(all_features) * 0.7)
    train_features = all_features[:split_point]
    train_labels = all_labels[:split_point]
    test_features = all_features[split_point:]
    test_labels = all_labels[split_point:]
    
    # train the svm from scratch
    support_weights = train_svm_scratch(train_features, train_labels)
    
    # evaluate and get metrics
    confusion, scores = get_confusion_matrix(test_features, test_labels, 
                                            train_features, train_labels, support_weights)
    fpr_list, tpr_list = calculate_roc_curve(test_labels, scores)
    
    # show results
    print("\nconfusion matrix:")
    print(confusion)
    print(f"\naccuracy: {(confusion[0,0] + confusion[1,1]) / confusion.sum():.3f}")
    
    # plot everything
    plot_results(confusion, fpr_list, tpr_list)
