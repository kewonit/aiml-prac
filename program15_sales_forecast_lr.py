"""linear regression to forecast sales using ad spend, discounts, and customer footfall
5-fold cross validation to check how good the predictions are"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

FILE_PATH = "datasets/15 ad spends.csv"


def load_data(limit=150):
    # just grab ad spend, discounts, and clicks as features. revenue is what we're predicting
    df = pd.read_csv(FILE_PATH, nrows=limit)
    x = df[["Ad_Spend", "Discount_Applied", "Clicks"]].values
    y = df["Revenue"].values
    return x, y


def gradient_descent(x, y, steps=400, lr=0.00001):
    # linear regression from scratch. basically just tweaking weights to minimize error
    weights = np.zeros(x.shape[1] + 1)
    
    for _ in range(steps):
        # predict = bias + (weight1 * feature1) + (weight2 * feature2) + ...
        preds = weights[0] + np.dot(x, weights[1:])
        error = preds - y
        
        # update weights based on how wrong we were
        weights[0] -= lr * np.mean(error)
        weights[1:] -= lr * np.dot(x.T, error) / len(y)
    
    return weights


def predict(w, x):
    # same formula: bias + dot product of weights and features
    return w[0] + np.dot(x, w[1:])


def evaluate_kfold(x, y, k=5):
    # split data into k chunks, train on k-1 and test on 1, repeat k times
    kf = KFold(n_splits=k, shuffle=False)
    scores = []
    last_preds = None
    last_truth = None
    
    for train_idx, test_idx in kf.split(x):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
        
        # train the model
        w = gradient_descent(x_train, y_train)
        preds = predict(w, x_test)
        
        # check how bad we did with mse
        mse = mean_squared_error(y_test, preds)
        scores.append(mse)
        
        last_preds = preds
        last_truth = y_test
    
    return np.mean(scores), last_preds, last_truth


def plot_results(actual, predicted):
    # scatter plot to see how close predictions are to reality
    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 5))
    plt.scatter(actual, predicted, alpha=0.6, s=80, color="steelblue")
    
    # line of perfect predictions for reference
    line_min, line_max = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    plt.plot([line_min, line_max], [line_min, line_max], '--', color="red", linewidth=2, label="perfect prediction")
    
    plt.xlabel("Actual Revenue", fontsize=11)
    plt.ylabel("Predicted Revenue", fontsize=11)
    plt.title("Sales Forecast: Actual vs Predicted Revenue", fontsize=12, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x, y = load_data()
    avg_mse, final_preds, final_truth = evaluate_kfold(x, y, k=5)
    print(f"5-fold cross validation MSE: {avg_mse:.2f}")
    plot_results(final_truth, final_preds)
