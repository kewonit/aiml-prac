"""gaussian naive bayes from scratch for email spam detection.
uses laplace smoothing on class priors to handle unseen classes.
works with continuous numerical features and calculates gaussian pdf.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_FILE = "datasets/emails_16_17_18_19.csv"


class GaussianNaiveBayesScratch:
    # gaussian naive bayes from scratch with laplace smoothing
    def __init__(self, laplace_alpha=1.0):
        self.laplace_alpha = laplace_alpha

    def fit(self, feature_matrix, labels):
        # extract unique classes and their probabilities
        self.classes = np.unique(labels)
        self.class_means = {}
        self.class_variances = {}
        self.class_priors = {}
        
        # calculate mean, variance, and prior for each class
        for class_label in self.classes:
            features_in_class = feature_matrix[labels == class_label]
            self.class_means[class_label] = features_in_class.mean(axis=0)
            # add small epsilon to avoid zero variance issues
            self.class_variances[class_label] = features_in_class.var(axis=0) + 1e-9
            # laplace smoothing: add alpha to numerator and alpha*num_classes to denominator
            self.class_priors[class_label] = (features_in_class.shape[0] + self.laplace_alpha) / (
                feature_matrix.shape[0] + len(self.classes) * self.laplace_alpha
            )

    def _gaussian_probability_density(self, class_label, feature_vector):
        # calculate gaussian pdf for each feature given the class
        means = self.class_means[class_label]
        variances = self.class_variances[class_label]
        variances = variances + 1e-9  # prevent division by zero
        
        # gaussian formula: (1 / sqrt(2*pi*var)) * exp(-(x - mean)^2 / (2*var))
        numerator = np.exp(-((feature_vector - means) ** 2) / (2 * variances))
        denominator = np.sqrt(2 * np.pi * variances)
        # add small epsilon to prevent division by zero
        pdf_result = numerator / (denominator + 1e-9)
        return pdf_result

    def _predict_single_sample(self, feature_vector):
        # calculate posterior probability for each class and pick highest
        posteriors = []
        for class_label in self.classes:
            # start with log of prior probability
            prior = np.log(self.class_priors[class_label])
            # calculate log of conditional probability using gaussian pdf
            pdf_vals = self._gaussian_probability_density(class_label, feature_vector)
            # handle zero values to avoid log(0)
            pdf_vals[pdf_vals == 0] = 1e-9
            conditional = np.sum(np.log(pdf_vals))
            # posterior = log(prior) + log(likelihood)
            posterior = prior + conditional
            posteriors.append(posterior)
        # return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, feature_matrix):
        # predict for entire matrix by calling predict on each sample
        return np.array([self._predict_single_sample(sample) for sample in feature_matrix])


if __name__ == "__main__":
    # load email spam dataset
    df = pd.read_csv(DATA_FILE)
    # extract features (all columns except first id column and last prediction column)
    features = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    
    # split into 80% train, 20% test
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # train the model
    nb_model = GaussianNaiveBayesScratch(laplace_alpha=1.0)
    nb_model.fit(features_train, labels_train)
    
    # predict on test set
    predictions = nb_model.predict(features_test)
    
    # calculate metrics
    accuracy = accuracy_score(labels_test, predictions)
    precision = precision_score(labels_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels_test, predictions, average='weighted', zero_division=0)
    
    # print results
    print("✅ model evaluation results - email spam detection")
    conf_matrix = confusion_matrix(labels_test, predictions)
    print("confusion matrix:")
    print(conf_matrix)
    print(f"\naccuracy: {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"f1-score: {f1:.4f}")
    
    print("\ndetailed classification report:")
    print(classification_report(labels_test, predictions, target_names=['not spam', 'spam']))
    
    # visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['not spam', 'spam'], yticklabels=['not spam', 'spam'])
    plt.title('confusion matrix - gaussian naive bayes spam detection')
    plt.xlabel('predicted label')
    plt.ylabel('actual label')
    plt.tight_layout()
    plt.show()
