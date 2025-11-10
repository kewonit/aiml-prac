"""
svm with polynomial kernel to predict if students pass or fail based on study time,
attendance, and internal scores. builds a simple kernel perceptron from scratch.
"""
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FILE_PATH = "datasets/student_performance_dataset_20.csv"


def load_dataset(limit=160):
    # pull study hours, attendance, internal scores. label as pass (1) or fail (-1)
    dataset = []
    with open(FILE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [
                float(row["Study_Hours_per_Week"]),
                float(row["Attendance_Rate"]),
                float(row["Internal_Scores"])
            ]
            label = 1 if row["Pass_Fail"].strip().lower() == "pass" else -1
            dataset.append((features, label))
            if len(dataset) >= limit:
                break
    return dataset


def load_raw_data():
    # load the full dataset for visualization (study hours and exam scores)
    data = pd.read_csv(FILE_PATH)
    return data


def dot_product(vector_a, vector_b):
    # basic dot product between two vectors
    return sum(vector_a[i] * vector_b[i] for i in range(len(vector_a)))


def polynomial_kernel(vector_x, vector_y, degree=2):
    # polynomial kernel: (x·y + 1)^degree. transforms features into higher dimension
    return (dot_product(vector_x, vector_y) + 1) ** degree


def train_svm_kernel_perceptron(training_data, num_epochs=3):
    # weights (alphas) for each training sample. start at 0 and update if misclassified
    support_weights = [0 for _ in training_data]
    
    for _ in range(num_epochs):
        for sample_idx, (sample_features, sample_label) in enumerate(training_data):
            # decision function: sum of weighted kernel values
            decision_value = 0
            for weight_idx, (train_features, train_label) in enumerate(training_data):
                if support_weights[weight_idx]:
                    decision_value += support_weights[weight_idx] * train_label * \
                                      polynomial_kernel(train_features, sample_features)
            
            # if prediction is wrong, bump up this sample's weight
            if sample_label * decision_value <= 0:
                support_weights[sample_idx] += 1
    
    return support_weights


def make_prediction(training_data, support_weights, test_features):
    # compute decision function on new sample using weighted kernel
    decision_value = 0
    for weight, (train_features, train_label) in zip(support_weights, training_data):
        if weight:
            decision_value += weight * train_label * polynomial_kernel(train_features, test_features)
    return 1 if decision_value >= 0 else -1


if __name__ == "__main__":
    # load and split dataset
    all_data = load_dataset()
    split_idx = int(len(all_data) * 0.7)
    training_set = all_data[:split_idx]
    testing_set = all_data[split_idx:split_idx + 40]
    
    # train the model
    support_weights = train_svm_kernel_perceptron(training_set)
    
    # evaluate on test set
    true_positives = false_positives = true_negatives = false_negatives = 0
    for test_features, true_label in testing_set:
        predicted_label = make_prediction(training_set, support_weights, test_features)
        
        if predicted_label == 1 and true_label == 1:
            true_positives += 1
        elif predicted_label == 1 and true_label == -1:
            false_positives += 1
        elif predicted_label == -1 and true_label == -1:
            true_negatives += 1
        else:
            false_negatives += 1
    
    # calculate metrics: precision, recall, f1-score, and accuracy
    accuracy = (true_positives + true_negatives) / len(testing_set) if testing_set else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    print("accuracy:", round(accuracy, 3))
    print("precision:", round(precision, 3))
    print("recall:", round(recall, 3))
    print("f1-score:", round(f1_score, 3))
    
    # plot study hours vs final exam score
    raw_data = load_raw_data()
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=raw_data, x="Study_Hours_per_Week", y="Final_Exam_Score", 
                    hue="Pass_Fail", palette={"Pass": "green", "Fail": "red"}, s=100)
    plt.xlabel("Study Hours per Week", fontsize=12)
    plt.ylabel("Final Exam Score", fontsize=12)
    plt.title("Study Hours vs Final Exam Score (Pass/Fail)", fontsize=14, fontweight="bold")
    plt.legend(title="Result", fontsize=10)
    plt.tight_layout()
    plt.show()
