"""linear regression from scratch - predicts exam score from study hours"""
import csv
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
FILE_PATH = "datasets/student_exam_scores_12_13.csv"


def load_data(limit=80):
    # reads study hours and exam scores from csv
    study_hours = []
    exam_scores = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            study_hours.append(float(row["hours_studied"]))
            exam_scores.append(float(row["exam_score"]))
            if len(study_hours) >= limit:
                break
    return study_hours, exam_scores


def calculate_mean(values):
    # just the sum divided by how many numbers there are
    return sum(values) / len(values)


def estimate_parameters(study_hours, exam_scores):
    # calculates slope and intercept for best-fit line using least squares
    # slope = how much score goes up for each hour studied
    # intercept = predicted score when study hours = 0
    mean_hours = calculate_mean(study_hours)
    mean_scores = calculate_mean(exam_scores)
    
    numerator = sum((study_hours[i] - mean_hours) * (exam_scores[i] - mean_scores) 
                    for i in range(len(study_hours)))
    denominator = sum((h - mean_hours) ** 2 for h in study_hours)
    
    slope = numerator / denominator if denominator else 0
    intercept = mean_scores - slope * mean_hours
    return slope, intercept


def make_predictions(slope, intercept, study_hours):
    # predict score using: y = intercept + slope * x
    return [intercept + slope * h for h in study_hours]


def mean_squared_error(predictions, actual_scores):
    # average of all the squared differences between predicted and actual
    return sum((predictions[i] - actual_scores[i]) ** 2 
               for i in range(len(predictions))) / len(predictions)


def r_squared_score(predictions, actual_scores):
    # how much of the score variation is explained by our model (0 to 1 scale)
    mean_score = calculate_mean(actual_scores)
    total_variance = sum((y - mean_score) ** 2 for y in actual_scores)
    residual_variance = sum((predictions[i] - actual_scores[i]) ** 2 
                            for i in range(len(predictions)))
    return 1 - (residual_variance / total_variance) if total_variance else 0


if __name__ == "__main__":
    study_hours, exam_scores = load_data()
    
    # train the model (estimate slope and intercept)
    slope, intercept = estimate_parameters(study_hours, exam_scores)
    
    # make predictions on training data
    predicted_scores = make_predictions(slope, intercept, study_hours)
    
    # print metrics
    print(f"slope: {slope:.4f}")
    print(f"intercept: {intercept:.4f}")
    print(f"mse: {mean_squared_error(predicted_scores, exam_scores):.3f}")
    print(f"r2 score: {r_squared_score(predicted_scores, exam_scores):.3f}")
    
    # plot actual vs predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(study_hours, exam_scores, alpha=0.6, s=50, label="actual scores")
    
    sorted_hours = sorted(study_hours)
    fitted_line = [intercept + slope * h for h in sorted_hours]
    plt.plot(sorted_hours, fitted_line, color="red", linewidth=2, label="fitted line")
    
    plt.xlabel("study hours", fontsize=11)
    plt.ylabel("exam score", fontsize=11)
    plt.title("linear regression: predicting exam score from study hours")
    plt.legend()
    plt.tight_layout()
    plt.show()
