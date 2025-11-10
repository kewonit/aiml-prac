"""uber ride price prediction using pca and eda
loads uber data, checks out the stats, squeezes it down with pca, trains models,
and compares how well they predict fares with different metrics"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE_PATH = "datasets/uber_9_10.csv"


def load_and_extract_features(sample_size=250):
    # load csv and grab fare + coordinates
    df = pd.read_csv(FILE_PATH)
    df = df[(df['fare_amount'] > 0)].head(sample_size)
    df['distance_x'] = abs(df['dropoff_longitude'] - df['pickup_longitude'])
    df['distance_y'] = abs(df['dropoff_latitude'] - df['pickup_latitude'])
    return df[['distance_x', 'distance_y']].values, df['fare_amount'].values


def normalize_features(feature_vectors):
    # centering data with sklearn
    scaler = StandardScaler()
    return scaler.fit_transform(feature_vectors)


def calculate_covariance_matrix_2d(centered_data):
    # check how much the two dimensions move together (correlation basically)
    cov_matrix = np.cov(centered_data.T)
    return cov_matrix[0, 0], cov_matrix[0, 1], cov_matrix[1, 1]


def find_principal_component_direction(centered_data):
    # pca finds the direction where data spreads out the most
    # basically finding the best line to squish everything onto
    variance_x, covariance_xy, variance_y = calculate_covariance_matrix_2d(centered_data)
    trace_value = variance_x + variance_y
    determinant_value = variance_x * variance_y - covariance_xy * covariance_xy
    discriminant = max(trace_value * trace_value - 4 * determinant_value, 0)
    largest_eigenvalue = (trace_value + math.sqrt(discriminant)) / 2
    # eigenvector points in the direction of most variance
    if covariance_xy != 0:
        direction_vector = (largest_eigenvalue - variance_y, covariance_xy)
    elif variance_x >= variance_y:
        direction_vector = (1, 0)
    else:
        direction_vector = (0, 1)
    # normalize the vector so it's length 1
    vector_length = math.sqrt(direction_vector[0] * direction_vector[0] + direction_vector[1] * direction_vector[1]) or 1
    return (direction_vector[0] / vector_length, direction_vector[1] / vector_length)


def project_onto_direction(centered_data, direction_vector):
    # squish the 2d data onto 1d by projecting onto the best direction
    return [[row[0] * direction_vector[0] + row[1] * direction_vector[1]] for row in centered_data]


def train_linear_model_with_gradient_descent(feature_matrix, target_values, iteration_steps=600, learning_rate=0.4):
    # use sklearn instead (not the core concept)
    model = LinearRegression()
    model.fit(feature_matrix, target_values)
    return model


def compute_model_performance_metrics(model, features, target_values):
    # use sklearn for metric calculations (not the core concept)
    predictions = model.predict(features)
    mae = mean_absolute_error(target_values, predictions)
    rmse = math.sqrt(mean_squared_error(target_values, predictions))
    r2 = r2_score(target_values, predictions)
    return mae, rmse, r2, predictions


def generate_exploratory_analysis(all_fares, all_features):
    # quick eda stats: min, max, avg, spread to understand the data better
    print("--- exploratory data analysis ---")
    print(f"fare prices: min=${np.min(all_fares):.2f}, max=${np.max(all_fares):.2f}, avg=${np.mean(all_fares):.2f}")
    print(f"x-distance: min={np.min(all_features[:, 0]):.4f}, max={np.max(all_features[:, 0]):.4f}, avg={np.mean(all_features[:, 0]):.4f}")
    print(f"y-distance: min={np.min(all_features[:, 1]):.4f}, max={np.max(all_features[:, 1]):.4f}, avg={np.mean(all_features[:, 1]):.4f}")
    print(f"total samples: {len(all_fares)}")
    print()


def show_prediction_vs_actual_plot(actual_fares, predicted_fares):
    # visualize how close predictions were to reality
    plt.figure(figsize=(6, 4))
    plt.scatter(actual_fares, predicted_fares, alpha=0.5, color="navy")
    # diagonal line shows perfect predictions
    line_min = min(np.min(actual_fares), np.min(predicted_fares))
    line_max = max(np.max(actual_fares), np.max(predicted_fares))
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="red")
    plt.title("Uber Fare Predictions")
    plt.xlabel("Actual Fare")
    plt.ylabel("Predicted Fare")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # load data
    feature_list, fare_list = load_and_extract_features()

    # run eda to see what we're dealing with
    generate_exploratory_analysis(fare_list, feature_list)

    # train on raw features first (no dimensionality reduction)
    centered_features = normalize_features(feature_list)
    raw_model = train_linear_model_with_gradient_descent(centered_features, fare_list)
    raw_mae, raw_rmse, raw_r2, raw_predictions = compute_model_performance_metrics(raw_model, centered_features, fare_list)

    # now squish features with pca and train again
    principal_direction = find_principal_component_direction(centered_features)
    pca_reduced_features = project_onto_direction(centered_features, principal_direction)
    pca_model = train_linear_model_with_gradient_descent(pca_reduced_features, fare_list)
    pca_mae, pca_rmse, pca_r2, pca_predictions = compute_model_performance_metrics(pca_model, pca_reduced_features, fare_list)

    # show results
    print("--- model performance comparison ---")
    print("Raw features -> MAE, RMSE, R2:", tuple(round(x, 3) for x in (raw_mae, raw_rmse, raw_r2)))
    print("PCA feature -> MAE, RMSE, R2:", tuple(round(x, 3) for x in (pca_mae, pca_rmse, pca_r2)))
    print()

    # show visualization
    show_prediction_vs_actual_plot(fare_list, raw_predictions)
