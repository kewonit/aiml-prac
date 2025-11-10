"""Straight line fit using only study hours from the student score csv.
Everything is manual: compute slope/intercept, then spit out MSE and R2.
"""
import csv

FILE_PATH = "datasets/student_exam_scores_12_13.csv"


def load_hours(limit=80):
    hours = []
    scores = []
    with open(FILE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hours.append(float(row["hours_studied"]))
            scores.append(float(row["exam_score"]))
            if len(hours) >= limit:
                break
    return hours, scores


def mean(vals):
    return sum(vals) / len(vals)


def fit_line(xs, ys):
    avg_x = mean(xs)
    avg_y = mean(ys)
    num = sum((xs[i] - avg_x) * (ys[i] - avg_y) for i in range(len(xs)))
    den = sum((x - avg_x) ** 2 for x in xs)
    slope = num / den if den else 0
    intercept = avg_y - slope * avg_x
    return slope, intercept


def predict_line(slope, intercept, xs):
    return [intercept + slope * x for x in xs]


def mse(preds, targets):
    return sum((preds[i] - targets[i]) ** 2 for i in range(len(preds))) / len(preds)


def r2_score(preds, targets):
    avg = mean(targets)
    ss_tot = sum((y - avg) ** 2 for y in targets)
    ss_res = sum((preds[i] - targets[i]) ** 2 for i in range(len(targets)))
    return 1 - ss_res / ss_tot if ss_tot else 0


if __name__ == "__main__":
    xs, ys = load_hours()
    slope, intercept = fit_line(xs, ys)
    preds = predict_line(slope, intercept, xs)
    print("Slope:", round(slope, 4))
    print("Intercept:", round(intercept, 4))
    print("MSE:", round(mse(preds, ys), 3))
    print("R2:", round(r2_score(preds, ys), 3))
