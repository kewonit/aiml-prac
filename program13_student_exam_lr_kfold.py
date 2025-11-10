"""
predicts exam scores using study hours, attendance, and internal marks.
validates the model with k-fold cross-validation. pretty straightforward stuff.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

data_path = "datasets/student_exam_scores_12_13.csv"


# grab the data columns we need
study_data = pd.read_csv(data_path)[["hours_studied", "attendance_percent", "Internal_marks", "exam_score"]]
study_features = study_data[["hours_studied", "attendance_percent", "Internal_marks"]].values
study_targets = study_data["exam_score"].values


def build_linear_model(train_feats, train_labels, iterations=500, learn_rate=0.0001):
    # weights start at zero: one for bias, rest for each feature
    model_weights = [0.0] * (train_feats[0].shape[0] + 1)
    
    for _ in range(iterations):
        grad_sum = [0.0] * len(model_weights)
        
        # loop through each training sample
        for sample, actual_score in zip(train_feats, train_labels):
            # calculate prediction: bias + (weight * feature for each feature)
            predicted_score = model_weights[0]
            for idx, feature_val in enumerate(sample):
                predicted_score += model_weights[idx + 1] * feature_val
            
            # error is how far off we are
            prediction_error = predicted_score - actual_score
            grad_sum[0] += prediction_error
            
            for idx, feature_val in enumerate(sample):
                grad_sum[idx + 1] += prediction_error * feature_val
        
        # update weights: move them towards lower error
        for idx in range(len(model_weights)):
            model_weights[idx] -= learn_rate * grad_sum[idx] / len(train_feats)
    
    return model_weights


def make_prediction(weights, feats):
    # prediction = bias + sum of (weight * feature)
    result = weights[0]
    for idx, val in enumerate(feats):
        result += weights[idx + 1] * val
    return result


def validate_with_kfold(features, targets, num_folds=5):
    kfold_splitter = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_mses = []
    last_fold_data = None
    
    for train_idx, test_idx in kfold_splitter.split(features):
        # split into train and test for this fold
        x_train_fold = features[train_idx]
        y_train_fold = targets[train_idx]
        x_test_fold = features[test_idx]
        y_test_fold = targets[test_idx]
        
        # train model on this fold
        weights = build_linear_model(x_train_fold, y_train_fold)
        
        # check how good it is on test data
        test_preds = [make_prediction(weights, sample) for sample in x_test_fold]
        fold_mse = mean_squared_error(y_test_fold, test_preds)
        fold_mses.append(fold_mse)
        
        # keep the last fold for visualization
        last_fold_data = (weights, x_test_fold, y_test_fold, test_preds)
    
    avg_mse = sum(fold_mses) / len(fold_mses)
    return avg_mse, last_fold_data


def plot_results(weights, test_feats, actual_scores, predictions):
    # plot: predicted vs actual to see how close we got
    plt.figure(figsize=(8, 5))
    plt.scatter(actual_scores, predictions, alpha=0.6, color="steelblue", s=60)
    
    # diagonal line shows perfect predictions
    min_val = min(min(actual_scores), min(predictions))
    max_val = max(max(actual_scores), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)
    
    plt.title("Student Exam Score Predictions (K-Fold Validation)")
    plt.xlabel("Actual Exam Score")
    plt.ylabel("Predicted Exam Score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # run k-fold validation
    avg_fold_mse, final_fold = validate_with_kfold(study_features, study_targets)
    
    print(f"average mse across 5 folds: {avg_fold_mse:.3f}")
    print(f"r2 score on last fold: {r2_score(final_fold[2], final_fold[3]):.3f}")
    
    if final_fold:
        plot_results(final_fold[0], final_fold[1], final_fold[2], final_fold[3])
