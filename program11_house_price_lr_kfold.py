"""
linear regression model to predict house prices from area, bedrooms, and location.
k-fold cross validation checks if model generalizes or just memorizes stuff.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = "datasets/House_Price_Dataset 11.csv"


def load_house_data(num_samples=80):
    # grab csv and encode location as numbers (dumb but works)
    data_df = pd.read_csv(FILE_PATH)
    data_df = data_df.head(num_samples)
    location_encoding = {"Rural": 0.0, "Suburban": 1.0, "Urban": 2.0}
    data_df["Location"] = data_df["Location"].map(location_encoding)
    feature_columns = ["Area", "Bedrooms", "Location"]
    X_features = data_df[feature_columns].values.tolist()
    y_prices = data_df["Price"].values.tolist()
    return X_features, y_prices


def compute_prediction(model_weights, feature_row):
    # just matrix multiplication basically: y = w0 + w1*x1 + w2*x2 + w3*x3
    result = model_weights[0]
    for i, feature_val in enumerate(feature_row):
        result += model_weights[i + 1] * feature_val
    return result


def fit_linear_regression(X_train, y_train, num_iterations=400, learning_rate=0.00000001):
    # start with random weights near zero
    model_weights = [0.0] * (len(X_train[0]) + 1)
    
    for _ in range(num_iterations):
        # compute gradients for all samples
        weight_gradients = [0.0] * len(model_weights)
        for feature_row, actual_price in zip(X_train, y_train):
            predicted_price = compute_prediction(model_weights, feature_row)
            prediction_error = predicted_price - actual_price
            weight_gradients[0] += prediction_error
            for j, feature_val in enumerate(feature_row):
                weight_gradients[j + 1] += prediction_error * feature_val
        
        # update weights by stepping in opposite direction of gradient
        num_samples = len(X_train)
        for i in range(len(model_weights)):
            model_weights[i] -= learning_rate * weight_gradients[i] / num_samples
    
    return model_weights


def calculate_mse(model_weights, X_data, y_data):
    # mean squared error: average of all squared mistakes
    squared_errors = [(compute_prediction(model_weights, x) - y) ** 2 for x, y in zip(X_data, y_data)]
    return sum(squared_errors) / len(squared_errors)


def calculate_r_squared(model_weights, X_data, y_data):
    # r² tells you how much of the variance your model explains (0-1 is good, 1 is perfect)
    y_mean = sum(y_data) / len(y_data)
    total_sum_squares = sum((y - y_mean) ** 2 for y in y_data)
    residual_sum_squares = sum((compute_prediction(model_weights, x) - y) ** 2 for x, y in zip(X_data, y_data))
    if total_sum_squares == 0:
        return 0.0
    return 1 - (residual_sum_squares / total_sum_squares)


def k_fold_cross_validation(X_features, y_prices, num_folds=4):
    # split data into chunks, train on most of it, test on one chunk
    fold_size = len(X_features) // num_folds
    fold_mse_scores = []
    last_fold_results = None
    
    for fold_idx in range(num_folds):
        # carve out this fold's test set
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size
        X_test = X_features[test_start:test_end]
        y_test = y_prices[test_start:test_end]
        
        # everything else is training data
        X_train = X_features[:test_start] + X_features[test_end:]
        y_train = y_prices[:test_start] + y_prices[test_end:]
        
        # train model and evaluate
        trained_weights = fit_linear_regression(X_train, y_train)
        fold_score = calculate_mse(trained_weights, X_test, y_test)
        fold_mse_scores.append(fold_score)
        last_fold_results = (trained_weights, X_test, y_test)
    
    avg_mse = sum(fold_mse_scores) / len(fold_mse_scores)
    return avg_mse, last_fold_results


def plot_results(model_weights, X_test, y_test):
    # predict on test set and plot actual vs predicted
    y_predicted = [compute_prediction(model_weights, x) for x in X_test]
    
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_predicted, alpha=0.6, s=100, color="steelblue")
    
    # perfect line would mean predictions = actual
    min_val = min(y_test + y_predicted)
    max_val = max(y_test + y_predicted)
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect")
    
    plt.xlabel("Actual Price", fontsize=11)
    plt.ylabel("Predicted Price", fontsize=11)
    plt.title("House Price Predictions vs Actual", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = load_house_data()
    avg_mse, (final_weights, X_final, y_final) = k_fold_cross_validation(X, y, num_folds=4)
    r_squared = calculate_r_squared(final_weights, X_final, y_final)
    print(f"avg mse across all folds: {avg_mse:.2f}")
    print(f"r² score on last fold: {r_squared:.4f}")
    plot_results(final_weights, X_final, y_final)
