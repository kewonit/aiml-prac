"""
linear regression to predict it salary based on experience, education, and skills.
uses gradient descent from scratch and evaluates with 5-fold cross-validation.
"""

import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

DATA_FILE = "datasets/salary_data_14_converted.csv"


def string_to_number(mapping_dict, text):
    """convert categorical text to a number. keeps same mapping throughout."""
    text = text.strip().lower()
    if text not in mapping_dict:
        mapping_dict[text] = float(len(mapping_dict))
    return mapping_dict[text]


def load_data(limit=180):
    """read the csv and turn it into features (experience, education, etc) and salary targets."""
    data_rows = []
    education_map = {}
    job_map = {}
    
    with open(DATA_FILE, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                experience = float(row.get("Years of Experience", 0)) if row.get("Years of Experience") else 0
                education = string_to_number(education_map, row.get("Education Level", "unknown"))
                job = string_to_number(job_map, row.get("Job Title", "other"))
                salary = float(row.get("Salary", 0))
            except (ValueError, KeyError):
                continue
            
            if salary <= 0:
                continue
            
            # features: experience and education and job (these matter for salary prediction)
            features = [experience, education, job]
            data_rows.append((features, salary))
            
            if len(data_rows) >= limit:
                break
    
    return data_rows


def train_model(feature_matrix, salary_targets, iterations=500, learning_rate=0.00001):
    """gradient descent: tweak the weights to minimize error on the training data."""
    num_features = len(feature_matrix[0])
    weights = [0.0] * (num_features + 1)  # +1 for bias term
    
    for iteration in range(iterations):
        gradients = [0.0] * len(weights)
        
        for features, actual_salary in zip(feature_matrix, salary_targets):
            # calculate prediction: bias + sum of (weight * feature)
            predicted_salary = weights[0]
            for i, feature_value in enumerate(features):
                predicted_salary += weights[i + 1] * feature_value
            
            # how far off are we
            error = predicted_salary - actual_salary
            
            # update gradients to push weights in the right direction
            gradients[0] += error
            for i, feature_value in enumerate(features):
                gradients[i + 1] += error * feature_value
        
        # move weights down the gradient slope
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradients[i] / len(feature_matrix)
    
    return weights


def predict_salary(weights, features):
    """use the trained weights to guess a salary."""
    predicted = weights[0]
    for i, feature_value in enumerate(features):
        predicted += weights[i + 1] * feature_value
    return predicted


def calculate_mse(weights, feature_matrix, actual_salaries):
    """mean squared error: how wrong are we on average."""
    total_error = sum((predict_salary(weights, f) - actual) ** 2 
                      for f, actual in zip(feature_matrix, actual_salaries))
    return total_error / len(feature_matrix)


def evaluate_with_5fold():
    """split data into 5 chunks. train on 4, test on 1. repeat 5 times. average the errors."""
    all_data = load_data()
    fold_size = len(all_data) // 5
    mse_scores = []
    best_model = None
    
    for fold_num in range(5):
        # create train/test split for this fold
        test_start = fold_num * fold_size
        test_end = test_start + fold_size
        test_data = all_data[test_start:test_end]
        train_data = all_data[:test_start] + all_data[test_end:]
        
        # separate features and targets
        train_features = [item[0] for item in train_data]
        train_salaries = [item[1] for item in train_data]
        test_features = [item[0] for item in test_data]
        test_salaries = [item[1] for item in test_data]
        
        # train the model and measure performance
        model_weights = train_model(train_features, train_salaries)
        fold_mse = calculate_mse(model_weights, test_features, test_salaries)
        mse_scores.append(fold_mse)
        
        # keep the last fold for visualization
        best_model = (model_weights, test_features, test_salaries)
    
    avg_mse = sum(mse_scores) / len(mse_scores)
    return avg_mse, best_model, mse_scores


def visualize_results(weights, test_features, test_salaries):
    """show how predictions compare to actual salaries."""
    predictions = [predict_salary(weights, f) for f in test_features]
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(test_salaries, predictions, alpha=0.6, s=50)
    min_val = min(test_salaries + predictions)
    max_val = max(test_salaries + predictions)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Predictions vs Actual")
    
    # residuals (errors)
    plt.subplot(1, 2, 2)
    residuals = [predictions[i] - test_salaries[i] for i in range(len(test_salaries))]
    plt.scatter(predictions, residuals, alpha=0.6, s=50, color="green")
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residual (Error)")
    plt.title("Residual Plot")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    avg_mse_score, final_model, all_mse_scores = evaluate_with_5fold()
    print(f"5-fold cross-validation mse: {avg_mse_score:,.2f}")
    print(f"individual fold scores: {[f'{m:,.2f}' for m in all_mse_scores]}")
    
    if final_model:
        visualize_results(final_model[0], final_model[1], final_model[2])
