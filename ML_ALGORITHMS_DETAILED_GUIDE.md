# Machine Learning Algorithms - Detailed Implementation Guide

## Table of Contents

1. [Linear Regression from Scratch (Program 12)](#linear-regression-from-scratch)
2. [Naive Bayes from Scratch (Program 17)](#naive-bayes-from-scratch)
3. [SVM with Oversampling (Program 18)](#svm-with-oversampling)
4. [SVM from Scratch (Program 19)](#svm-from-scratch)
5. [Polynomial Kernel SVM (Program 20)](#polynomial-kernel-svm)
6. [Breast Cancer Classification with ROC (Program 21)](#breast-cancer-classification-with-roc)

---

## Linear Regression from Scratch (Program 12)

### Problem Statement

Build a Linear Regression model **from scratch** to predict students' final exam scores based on their study hours. Implement all computations manually (without using built-in regression libraries) — including parameter estimation, prediction, and model evaluation using Mean Squared Error (MSE) and R² Score.

### Theory Overview

**Linear Regression** is a supervised learning algorithm that models the relationship between a dependent variable (target) and one or more independent variables (features) using a linear equation:

$$y = mx + b$$

Where:

- $y$ = predicted value (exam score)
- $x$ = input feature (hours studied)
- $m$ = slope (how much y changes per unit of x)
- $b$ = intercept (y-value when x=0)

**Key Concepts:**

1. **Ordinary Least Squares (OLS)**: Minimizes the sum of squared differences between actual and predicted values

2. **Slope Calculation**:
   $$m = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

3. **Intercept Calculation**:
   $$b = \bar{y} - m\bar{x}$$

4. **Mean Squared Error (MSE)**: Average squared difference between predictions and actual values
   $$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

5. **R² Score**: Proportion of variance explained by the model (0 to 1, higher is better)
   $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### Code Implementation Walkthrough

```python
import matplotlib.pyplot as plt
import pandas as pd

FILE_PATH = "datasets/student_exam_scores_12_13.csv"
```

**Data Loading** (Using pandas for convenience):

```python
def load_hours(limit=80):
    df = pd.read_csv(FILE_PATH)
    df = df.head(limit)
    hours = df["hours_studied"].tolist()
    scores = df["exam_score"].tolist()
    return hours, scores
```

- Loads CSV using pandas (easier than manual CSV parsing)
- Extracts only "hours_studied" (feature) and "exam_score" (target)
- Limits to 80 samples for faster computation

**Manual Mean Calculation**:

```python
def mean(vals):
    return sum(vals) / len(vals)
```

- Simple arithmetic mean: sum all values and divide by count
- Used for calculating $\bar{x}$ and $\bar{y}$

**Manual Slope and Intercept Calculation** (Core Algorithm):

```python
def fit_line(xs, ys):
    avg_x = mean(xs)
    avg_y = mean(ys)
    num = sum((xs[i] - avg_x) * (ys[i] - avg_y) for i in range(len(xs)))
    den = sum((x - avg_x) ** 2 for x in xs)
    slope = num / den if den else 0
    intercept = avg_y - slope * avg_x
    return slope, intercept
```

- **Numerator**: Covariance between x and y
- **Denominator**: Variance of x
- **Slope**: Covariance/Variance ratio
- **Intercept**: Adjusted to ensure line passes through mean point $(\bar{x}, \bar{y})$

**Prediction**:

```python
def predict_line(slope, intercept, xs):
    return [intercept + slope * x for x in xs]
```

- Apply linear equation: $\hat{y} = b + mx$
- Returns predicted values for all input x values

**Manual MSE Calculation**:

```python
def mse(preds, targets):
    return sum((preds[i] - targets[i]) ** 2 for i in range(len(preds))) / len(preds)
```

- Squares each error (prediction - actual)
- Averages all squared errors
- Penalizes large errors more than small ones

**Manual R² Score Calculation**:

```python
def r2_score(preds, targets):
    avg = mean(targets)
    ss_tot = sum((y - avg) ** 2 for y in targets)  # Total variance
    ss_res = sum((preds[i] - targets[i]) ** 2 for i in range(len(targets)))  # Residual variance
    return 1 - ss_res / ss_tot if ss_tot else 0
```

- **SS_tot**: Total sum of squares (variance in original data)
- **SS_res**: Residual sum of squares (variance in errors)
- R² = 1 means perfect fit, 0 means no better than mean

**Main Execution**:

```python
if __name__ == "__main__":
    xs, ys = load_hours()
    slope, intercept = fit_line(xs, ys)
    preds = predict_line(slope, intercept, xs)
    print("Slope:", round(slope, 4))
    print("Intercept:", round(intercept, 4))
    print("MSE:", round(mse(preds, ys), 3))
    print("R2:", round(r2_score(preds, ys), 3))

    # Visualization
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, color="royalblue", alpha=0.6, label="Actual scores")
    xs_sorted = sorted(xs)
    line_vals = [intercept + slope * x for x in xs_sorted]
    plt.plot(xs_sorted, line_vals, color="red", label="Fitted line")
    plt.title("Study Hours vs Exam Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
```

### Test Results

```
Slope: 1.5969
Intercept: 23.803
MSE: 16.143
R2: 0.611
```

**Interpretation**:

- **Slope (1.5969)**: Each additional hour of study increases exam score by ~1.6 points
- **Intercept (23.803)**: Expected score with 0 hours studied is ~23.8
- **MSE (16.143)**: Average squared error is 16.14 (RMSE ≈ 4 points)
- **R² (0.611)**: Model explains 61.1% of variance in exam scores

### Why Manual Implementation?

1. **Understanding**: Know exactly what's happening under the hood
2. **Interview Prep**: Common whiteboard question
3. **Debugging**: Can trace through each calculation step
4. **Customization**: Easy to modify formula or add constraints

---

## Naive Bayes from Scratch (Program 17)

### Problem Statement

Implement the Naïve Bayes algorithm **from scratch** to solve a real-world classification problem (disease diagnosis based on symptoms).

### Theory Overview

**Naive Bayes** is a probabilistic classifier based on Bayes' Theorem with the "naive" assumption that features are independent given the class label.

**Bayes' Theorem**:
$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

Where:

- $P(C|X)$ = Posterior probability (probability of class C given features X)
- $P(X|C)$ = Likelihood (probability of features X given class C)
- $P(C)$ = Prior probability (overall frequency of class C)
- $P(X)$ = Evidence (probability of features X)

**For classification**:
$$\text{Class} = \arg\max_{c} P(C=c) \prod_{i=1}^{n} P(X_i=x_i|C=c)$$

**Laplace Smoothing** (Add-1 smoothing):
$$P(X_i=x_i|C=c) = \frac{\text{count}(x_i, c) + 1}{\text{count}(c) + |\text{vocab}|}$$

Prevents zero probabilities when a feature-class combination hasn't been seen in training.

### Code Implementation Walkthrough

**Data Loading**:

```python
import math
import pandas as pd
from sklearn.metrics import accuracy_score

FILE_PATH = "datasets/disease_diagnosis_16_17.csv"

def load_rows(limit=200):
    df = pd.read_csv(FILE_PATH)
    df = df.head(limit)
    rows = []
    for _, row in df.iterrows():
        feat = (row["Symptom_1"], row["Symptom_2"], row["Symptom_3"])
        label = row["Diagnosis"]
        rows.append((feat, label))
    return rows
```

- Uses pandas to read CSV efficiently
- Each row contains 3 symptoms (categorical features)
- Label is the disease diagnosis

**Training Phase - Building Probability Tables**:

```python
def train(nb_rows):
    priors = {}  # P(C)
    cond = {}    # P(X|C)
    vocab = set()  # All unique symptoms

    # Count occurrences
    for feats, label in nb_rows:
        priors[label] = priors.get(label, 0) + 1
        bucket = cond.setdefault(label, {})
        for feat in feats:
            vocab.add(feat)
        for idx, feat in enumerate(feats):
            key = (idx, feat)  # (symptom_position, symptom_value)
            bucket[key] = bucket.get(key, 0) + 1

    return priors, cond, vocab
```

**Key Data Structures**:

1. **priors**: Dictionary of {disease: count}

   - Example: `{'Flu': 50, 'Cold': 30, 'COVID': 20}`

2. **cond**: Nested dictionary of {disease: {(position, symptom): count}}

   - Example: `{'Flu': {(0, 'fever'): 45, (1, 'cough'): 40}}`

3. **vocab**: Set of all unique symptoms
   - Used for Laplace smoothing denominator

**Prediction Phase**:

```python
def predict(example, priors, cond, vocab):
    total = sum(priors.values())
    best = None
    best_score = None

    for label, count in priors.items():
        # Start with log prior: log(P(C))
        score = math.log(count / total)
        bucket = cond[label]

        # Add log likelihoods: log(P(X_i|C)) for each feature
        for idx, feat in enumerate(example):
            key = (idx, feat)
            hit = bucket.get(key, 0)
            # Laplace smoothing: (count + 1) / (total + vocab_size)
            score += math.log((hit + 1) / (priors[label] + len(vocab)))

        # Keep track of highest score
        if best_score is None or score > best_score:
            best_score = score
            best = label

    return best
```

**Why Log Probabilities?**

- Multiplying many small probabilities causes **underflow** (numbers too small to represent)
- $\log(a \times b) = \log(a) + \log(b)$ converts multiplication to addition
- Comparison still works: if $P(A) > P(B)$ then $\log P(A) > \log P(B)$

**Main Execution**:

```python
if __name__ == "__main__":
    rows = load_rows()
    split = len(rows) // 2
    train_rows = rows[:split]
    test_rows = rows[split:]

    # Train
    priors, cond, vocab = train(train_rows)

    # Test
    y_true = [label for _, label in test_rows]
    y_pred = [predict(feats, priors, cond, vocab) for feats, _ in test_rows]

    # Evaluate using sklearn
    accuracy = accuracy_score(y_true, y_pred)
    print("Test accuracy:", round(accuracy, 3))
```

### Test Results

```
Test accuracy: 0.66
```

**Interpretation**:

- 66% accuracy on disease diagnosis
- Model correctly predicts 2 out of 3 diagnoses
- Better than random guessing (assuming balanced classes)

### When to Use Naive Bayes?

✅ **Good for**:

- Text classification (spam detection, sentiment analysis)
- Medical diagnosis with categorical symptoms
- Fast training and prediction
- Works well with small datasets

❌ **Not ideal for**:

- Features that are highly correlated (violates independence assumption)
- Continuous numerical features (need discretization or Gaussian NB)

---

## SVM with Oversampling (Program 18)

### Problem Statement

Implementation of an Email Spam Detection model using a Support Vector Machine (SVM) for binary classification (Normal vs Spam). Apply **oversampling** techniques to handle class imbalance and analyze model performance.

### Theory Overview

**Support Vector Machine (SVM)** finds the optimal hyperplane that maximizes the margin between classes.

**Linear SVM Decision Function**:
$$f(x) = w^T x + b$$

**Classification**:

- Predict class +1 if $f(x) \geq 0$
- Predict class -1 if $f(x) < 0$

**Hinge Loss** (for training):
$$L(w, b) = \frac{\lambda}{2}||w||^2 + \frac{1}{n}\sum_{i=1}^{n}\max(0, 1 - y_i(w^T x_i + b))$$

Where:

- $\lambda$ = regularization parameter (prevents overfitting)
- First term: Regularization (keeps weights small)
- Second term: Hinge loss (penalizes misclassifications and small margins)

**Gradient Descent Update Rules**:

- If margin $< 1$: `w ← (1 - lr·λ)w + lr·y·x` and `b ← b + lr·y`
- If margin $\geq 1$: `w ← (1 - lr·λ)w` (only regularization)

**Class Imbalance Problem**:

- Real datasets often have unequal class distributions
- Example: 1000 normal emails, 100 spam emails
- Model might ignore minority class and still get 90% accuracy

**Oversampling Solution**:

- Duplicate minority class samples until classes are balanced
- Random oversampling: randomly copy minority examples
- Alternative: SMOTE (creates synthetic samples)

### Code Implementation Walkthrough

**Imports and Data Loading**:

```python
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_email_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    features = df.drop(columns=["Email No.", "Prediction"]).to_numpy(dtype=float)
    labels = df["Prediction"].map({0: -1, 1: 1}).to_numpy(dtype=int)
    return features, labels
```

- Features: Word count vectors (each column represents a word frequency)
- Labels: 0 (not spam) → -1, 1 (spam) → +1 (SVM convention)

**Manual Oversampling Implementation**:

```python
def oversample_minority(
    X: np.ndarray, y: np.ndarray, seed: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    counts = Counter(y)
    if len(counts) < 2:
        return X, y

    # Find majority and minority classes
    ((maj_label, maj_count), (min_label, min_count)) = sorted(
        counts.items(), key=lambda item: item[1], reverse=True
    )
    if maj_count == min_count:
        return X, y

    rng = np.random.default_rng(seed)
    minority_indices = np.where(y == min_label)[0]

    # Sample with replacement to match majority count
    extra_indices = rng.choice(minority_indices,
                              size=maj_count - min_count,
                              replace=True)

    X_extra = X[extra_indices]
    y_extra = y[extra_indices]

    # Combine and shuffle
    X_balanced = np.vstack((X, X_extra))
    y_balanced = np.concatenate((y, y_extra))
    shuffle_idx = rng.permutation(len(X_balanced))

    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]
```

**Step-by-step**:

1. Count occurrences of each class
2. Identify which class has fewer samples
3. Randomly duplicate minority samples
4. Add duplicates to dataset
5. Shuffle to mix old and new samples

**Manual SVM Training** (Core Algorithm):

```python
def train_svm(
    rows: Sequence[Tuple[np.ndarray, float]],
    steps: int = 35,
    lr: float = 0.0008,
    reg: float = 0.01,
) -> Tuple[np.ndarray, float]:
    feature_count = rows[0][0].shape[0]
    weights = np.zeros(feature_count)
    bias = 0.0

    rows_list: List[Tuple[np.ndarray, float]] = list(rows)
    for _ in range(steps):
        np.random.shuffle(rows_list)
        for feats, label in rows_list:
            margin = label * (np.dot(weights, feats) + bias)

            if margin < 1:  # Hinge loss active
                weights = (1 - lr * reg) * weights + lr * label * feats
                bias += lr * label
            else:  # Only regularization
                weights = (1 - lr * reg) * weights

    return weights, bias
```

**Key Points**:

- **Margin**: $y_i(w^T x_i + b)$ should be $\geq 1$ for correct classification
- If margin < 1: Update both weights and bias (misclassified or too close)
- If margin ≥ 1: Only apply regularization (correct classification)
- Shuffle data each epoch for better convergence

**Prediction and Evaluation**:

```python
def predict(weights: np.ndarray, bias: float, feats: np.ndarray) -> int:
    return 1 if np.dot(weights, feats) + bias >= 0 else -1

def evaluate(
    rows: Iterable[Tuple[np.ndarray, float]],
    weights: np.ndarray,
    bias: float
) -> Tuple[float, float, float, float]:
    tp = fp = tn = fn = 0
    for feats, label in rows:
        guess = predict(weights, feats, bias)
        if guess == 1 and label == 1:
            tp += 1
        elif guess == 1 and label == -1:
            fp += 1
        elif guess == -1 and label == -1:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return accuracy, precision, recall, f1
```

**Metrics Explained**:

- **Accuracy**: (TP + TN) / Total - Overall correctness
- **Precision**: TP / (TP + FP) - Of predicted spam, how many are actually spam?
- **Recall**: TP / (TP + FN) - Of actual spam, how many did we catch?
- **F1 Score**: Harmonic mean of precision and recall

**Main Pipeline**:

```python
if __name__ == "__main__":
    X, y = load_email_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Balance classes
    X_balanced, y_balanced = oversample_minority(X_train, y_train)

    train_rows = [(feat, lbl) for feat, lbl in zip(X_balanced, y_balanced)]
    test_rows = [(feat, lbl) for feat, lbl in zip(X_test, y_test)]

    # Train and evaluate
    weights, bias = train_svm(train_rows)
    acc, prec, rec, f1 = evaluate(test_rows, weights, bias)

    print("Train class counts:", Counter(int(lbl) for lbl in y_train))
    print("Balanced train counts:", Counter(int(lbl) for lbl in y_balanced))
    print("Test class counts:", Counter(int(lbl) for lbl in y_test))
    print("Accuracy:", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall:", round(rec, 3))
    print("F1:", round(f1, 3))
```

### Test Results

```
Train class counts: Counter({-1: 2945, 1: 1192})
Balanced train counts: Counter({-1: 2945, 1: 2945})
Test class counts: Counter({-1: 727, 1: 308})
Accuracy: 0.965
Precision: 0.907
Recall: 0.984
F1: 0.944
```

**Interpretation**:

- **Before balancing**: 2945 normal vs 1192 spam (2.5:1 ratio)
- **After balancing**: 2945 vs 2945 (1:1 ratio - perfectly balanced)
- **Accuracy (96.5%)**: Very high overall correctness
- **Precision (90.7%)**: 90.7% of flagged emails are actually spam
- **Recall (98.4%)**: Catches 98.4% of all spam emails
- **F1 (94.4%)**: Excellent balance between precision and recall

### Why Oversampling Matters?

Without balancing, model might achieve 70% accuracy by just predicting "not spam" for everything. Oversampling forces the model to learn patterns in spam emails.

---

## SVM from Scratch (Program 19)

### Problem Statement

Implement an Email Spam Detection model **from scratch** using the Support Vector Machine (SVM) algorithm for binary classification, where emails are labeled as Normal (Not Spam) or Abnormal (Spam).

### Theory Overview

Same theoretical foundation as Program 18, but focuses on clean implementation without oversampling. This demonstrates the pure SVM algorithm with hinge loss optimization.

**Key Difference from Program 18**:

- No oversampling (uses natural class distribution)
- Focus on core SVM training algorithm
- Uses sklearn only for preprocessing (train/test split, scaling)

### Code Implementation Walkthrough

**Simplified Data Pipeline**:

```python
from typing import Iterable, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATH = Path("datasets/emails_16_17_18_19.csv")

def load_email_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.drop(columns=["Email No.", "Prediction"]).to_numpy(dtype=float)
    y = df["Prediction"].map({0: -1, 1: 1}).to_numpy(dtype=int)
    return X, y
```

**Manual SVM Training** (Identical to Program 18):

```python
def train_svm(
    rows: Sequence[Tuple[np.ndarray, float]],
    steps: int = 35,
    lr: float = 0.0008,
    reg: float = 0.01,
) -> Tuple[np.ndarray, float]:
    feature_count = rows[0][0].shape[0]
    weights = np.zeros(feature_count)
    bias = 0.0
    rows_list = list(rows)

    for _ in range(steps):
        np.random.shuffle(rows_list)
        for feats, label in rows_list:
            margin = label * (np.dot(weights, feats) + bias)

            if margin < 1:
                weights = (1 - lr * reg) * weights + lr * label * feats
                bias += lr * label
            else:
                weights = (1 - lr * reg) * weights

    return weights, bias
```

**Training Algorithm Breakdown**:

1. **Initialization**:

   - Start with zero weights and bias
   - Random initialization could work too

2. **Epoch Loop** (35 iterations):

   - Shuffle training data each epoch
   - Stochastic approach: update after each sample

3. **For Each Sample**:

   - Calculate margin: $y_i(w^T x_i + b)$
   - If margin < 1: Misclassified or within margin
     - Update weights: Consider both loss and regularization
     - Update bias: Push decision boundary
   - If margin ≥ 1: Correctly classified with good margin
     - Only apply regularization (decay weights slightly)

4. **Hyperparameters**:
   - `steps=35`: Number of passes through data
   - `lr=0.0008`: Learning rate (small for stability)
   - `reg=0.01`: Regularization strength (prevents overfitting)

**Prediction** (Sign of decision function):

```python
def predict(weights: np.ndarray, bias: float, feats: np.ndarray) -> int:
    return 1 if np.dot(weights, feats) + bias >= 0 else -1
```

**Evaluation** (Confusion Matrix Metrics):

```python
def evaluate(
    rows: Iterable[Tuple[np.ndarray, float]],
    weights: np.ndarray,
    bias: float
) -> Tuple[float, float, float, float]:
    tp = fp = tn = fn = 0
    for feats, label in rows:
        guess = predict(weights, feats, bias)
        if guess == 1 and label == 1:
            tp += 1
        elif guess == 1 and label == -1:
            fp += 1
        elif guess == -1 and label == -1:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return accuracy, precision, recall, f1
```

**Main Execution**:

```python
if __name__ == "__main__":
    X, y = load_email_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_rows = [(feat, lbl) for feat, lbl in zip(X_train, y_train)]
    test_rows = [(feat, lbl) for feat, lbl in zip(X_test, y_test)]

    weights, bias = train_svm(train_rows)
    acc, prec, rec, f1 = evaluate(test_rows, weights, bias)

    print("Train size:", len(train_rows))
    print("Test size:", len(test_rows))
    print("Accuracy:", round(acc, 3))
    print("Precision:", round(prec, 3))
    print("Recall:", round(rec, 3))
    print("F1:", round(f1, 3))
```

### Test Results

```
Train size: 4137
Test size: 1035
Accuracy: 0.974
Precision: 0.941
Recall: 0.969
F1: 0.955
```

**Interpretation**:

- **97.4% Accuracy**: Extremely high classification accuracy
- **94.1% Precision**: 94.1% of spam predictions are correct
- **96.9% Recall**: Catches 96.9% of actual spam
- **95.5% F1**: Excellent balance

**Why Better Than Program 18?**

- Larger training set (no balancing needed)
- Natural class distribution works well
- Dataset might have enough spam examples

### SVM vs Other Classifiers

| Aspect            | SVM                       | Naive Bayes   | Logistic Regression   |
| ----------------- | ------------------------- | ------------- | --------------------- |
| Decision Boundary | Maximum margin hyperplane | Probabilistic | Linear threshold      |
| Training Speed    | Slower                    | Fastest       | Fast                  |
| Memory Usage      | Medium                    | Low           | Low                   |
| Non-linear Data   | Yes (kernels)             | No            | No (without features) |
| Interpretability  | Medium                    | High          | High                  |

---

## Polynomial Kernel SVM (Program 20)

### Problem Statement

Implement an SVM model **from scratch** with a **Polynomial Kernel** to predict student performance (Pass/Fail) using study time, absences, and internal scores. Assess model performance using precision, recall, and F1-score.

### Theory Overview

**Kernel Trick**: Transforms data into higher-dimensional space where classes become linearly separable.

**Linear vs Kernel SVM**:

- **Linear**: Decision boundary is a straight line/plane
  - $f(x) = w^T x + b$
- **Kernel**: Decision boundary can be curved
  - $f(x) = \sum_{i} \alpha_i y_i K(x_i, x) + b$

**Polynomial Kernel**:
$$K(x, z) = (x^T z + 1)^d$$

Where $d$ is the degree (usually 2 or 3).

**Example (2D → 5D transformation, degree=2)**:

- Input: $x = [x_1, x_2]$
- Polynomial features: $[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]$
- Kernel computes this implicitly without explicit transformation!

**Kernel Perceptron Algorithm**:
Instead of maintaining weight vector $w$, we maintain coefficients $\alpha_i$ for each training sample:

1. Initialize $\alpha_i = 0$ for all training samples
2. For each training sample $(x_i, y_i)$:
   - Compute score: $\text{score} = \sum_{j} \alpha_j y_j K(x_j, x_i)$
   - If $y_i \times \text{score} \leq 0$: Misclassified, so $\alpha_i \leftarrow \alpha_i + 1$
3. Repeat for multiple epochs

**Prediction**:
$$\text{class} = \text{sign}\left(\sum_{i} \alpha_i y_i K(x_i, x_{\text{new}})\right)$$

### Code Implementation Walkthrough

**Data Loading**:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FILE_PATH = "datasets/student_performance_dataset_20.csv"

def load_rows(limit=160):
    df = pd.read_csv(FILE_PATH, encoding="utf-8")
    df = df.head(limit)
    rows = []
    for _, row in df.iterrows():
        feats = [
            float(row["Study_Hours_per_Week"]),
            float(row["Attendance_Rate"]),
            float(row["Internal_Scores"])
        ]
        label = 1 if row["Pass_Fail"].strip().lower() == "pass" else -1
        rows.append((feats, label))
    return rows
```

- 3 features: study hours, attendance rate, internal scores
- Binary classification: Pass (+1) or Fail (-1)

**Manual Dot Product**:

```python
def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))
```

- Simple element-wise multiplication and sum
- Used inside kernel function

**Polynomial Kernel Implementation**:

```python
def poly_kernel(a, b, degree=2):
    return (dot(a, b) + 1) ** degree
```

**Why $(x^T z + 1)$ instead of just $(x^T z)$?**

- Adding 1 includes bias term automatically
- Allows kernel to capture constant offset
- Example: $(x \cdot z + 1)^2 = (x \cdot z)^2 + 2(x \cdot z) + 1$

**Degree Effects**:

- **Degree 1**: Linear kernel (no transformation)
- **Degree 2**: Quadratic decision boundary
- **Degree 3**: Cubic curves (more flexible)
- **Higher degrees**: Risk of overfitting

**Training - Kernel Perceptron**:

```python
def train_kernel_perceptron(rows, epochs=3):
    alphas = [0 for _ in rows]

    for _ in range(epochs):
        for idx, (x_i, y_i) in enumerate(rows):
            # Compute decision score
            score = 0
            for j, (x_j, y_j) in enumerate(rows):
                if alphas[j]:  # Skip if alpha is zero
                    score += alphas[j] * y_j * poly_kernel(x_j, x_i)

            # If misclassified, increment alpha
            if y_i * score <= 0:
                alphas[idx] += 1

    return alphas
```

**Key Insights**:

- Only misclassified points get non-zero alphas
- Points with higher alphas are "support vectors"
- $\alpha_i$ represents how many times sample $i$ was misclassified

**Prediction Function**:

```python
def predict(rows, alphas, x):
    score = 0
    for alpha, (x_i, y_i) in zip(alphas, rows):
        if alpha:
            score += alpha * y_i * poly_kernel(x_i, x)
    return 1 if score >= 0 else -1
```

- Aggregate votes from all support vectors
- Each support vector contributes weighted by alpha
- Sign of sum determines class

**Main Execution**:

```python
if __name__ == "__main__":
    data = load_rows()
    split = int(len(data) * 0.7)
    train = data[:split]
    test = data[split:split + 40]
    alphas = train_kernel_perceptron(train)

    y_true = [label for _, label in test]
    y_pred = [predict(train, alphas, feats) for feats, _ in test]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1:", round(f1, 3))
```

### Test Results

```
Accuracy: 0.475
Precision: 0.475
Recall: 1.0
F1: 0.644
```

**Interpretation**:

- **47.5% Accuracy**: Poor performance (worse than random!)
- **47.5% Precision**: Less than half of "Pass" predictions are correct
- **100% Recall**: Model predicts "Pass" for EVERYTHING
- **64.4% F1**: Imbalanced - high recall, low precision

**Why Poor Performance?**

1. **Class imbalance**: Dataset might have many more "Pass" than "Fail"
2. **Model defaulting**: Predicting majority class for all samples
3. **Kernel perceptron limitations**: Simple algorithm, prone to overfitting
4. **Small dataset**: Only 160 samples for training
5. **Need better algorithm**: Full SVM with SMO optimization would work better

**How to Improve**:

- Use proper SVM optimization (SMO algorithm)
- Balance classes with oversampling/undersampling
- Try different kernel parameters (degree, regularization)
- Add more features or feature engineering

### When to Use Polynomial Kernels?

✅ **Good for**:

- Non-linearly separable data
- Moderate-sized datasets (100s to 1000s)
- Problems where feature interactions matter
- When you suspect polynomial relationships

❌ **Avoid when**:

- Linear kernel already works well (Occam's razor)
- Very large datasets (kernel computation is O(n²))
- High-dimensional data (may overfit)
- Need real-time predictions (slower than linear)

---

## Breast Cancer Classification with ROC (Program 21)

### Problem Statement

Develop an SVM classifier **from scratch** using a **Polynomial Kernel** on the Breast Cancer Wisconsin Dataset to distinguish between benign and malignant tumors. Evaluate using **confusion matrix** and **ROC curve**.

### Theory Overview

**ROC (Receiver Operating Characteristic) Curve**:

- Plots True Positive Rate vs False Positive Rate at various decision thresholds
- Shows trade-off between sensitivity and specificity

**Key Metrics**:

$$\text{True Positive Rate (TPR/Recall/Sensitivity)} = \frac{TP}{TP + FN}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$

$$\text{True Negative Rate (TNR/Specificity)} = \frac{TN}{TN + FP}$$

**AUC (Area Under Curve)**:

- Perfect classifier: AUC = 1.0
- Random classifier: AUC = 0.5
- Measures overall discriminative ability

**Confusion Matrix**:

```
                 Predicted
              Positive  Negative
Actual Pos      TP        FN
       Neg      FP        TN
```

**Threshold Selection**:

- High threshold → Few positives → High precision, low recall
- Low threshold → Many positives → Low precision, high recall
- ROC curve helps visualize this trade-off

### Code Implementation Walkthrough

**Data Loading**:

```python
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "datasets/Breast Cancer Wisconsin (Diagnostic)_21.csv"

def load_rows(limit=220):
    df = pd.read_csv(FILE_PATH, encoding="utf-8")
    df = df.head(limit)
    rows = []
    for _, row in df.iterrows():
        feats = [
            float(row["radius_mean"]),
            float(row["texture_mean"]),
            float(row["perimeter_mean"])
        ]
        label = 1 if row["diagnosis"].strip().upper() == "M" else -1  # M=Malignant, B=Benign
        rows.append((feats, label))
    return rows
```

- Medical dataset: tumor measurements
- 3 features: radius, texture, perimeter (mean values)
- Binary: Malignant (+1) or Benign (-1)

**Polynomial Kernel (Degree 3)**:

```python
def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def poly_kernel(a, b, degree=3):
    return (dot(a, b) + 1) ** degree
```

- Degree 3 for more complex decision boundaries
- Medical data often has non-linear patterns

**Training** (Same Kernel Perceptron):

```python
def train_kernel_perceptron(rows, epochs=3):
    alphas = [0 for _ in rows]
    for _ in range(epochs):
        for idx, (x_i, y_i) in enumerate(rows):
            score = 0
            for j, (x_j, y_j) in enumerate(rows):
                if alphas[j]:
                    score += alphas[j] * y_j * poly_kernel(x_j, x_i)
            if y_i * score <= 0:
                alphas[idx] += 1
    return alphas
```

**Scoring Function** (Returns continuous value):

```python
def score_point(rows, alphas, x):
    total = 0
    for alpha, (x_i, y_i) in zip(alphas, rows):
        if alpha:
            total += alpha * y_i * poly_kernel(x_i, x)
    return total
```

- Returns raw decision score (not just -1/+1)
- Used for ROC curve generation

**Prediction**:

```python
def predict(rows, alphas, x):
    return 1 if score_point(rows, alphas, x) >= 0 else -1
```

**Confusion Matrix Calculation**:

```python
def confusion(test, rows, alphas):
    TP = FP = TN = FN = 0
    scores = []

    for feats, label in test:
        s = score_point(rows, alphas, feats)
        guess = 1 if s >= 0 else -1
        scores.append((s, label))

        if guess == 1 and label == 1:
            TP += 1
        elif guess == 1 and label == -1:
            FP += 1
        elif guess == -1 and label == -1:
            TN += 1
        else:
            FN += 1

    return (TP, FP, TN, FN), scores
```

- Tracks all four confusion matrix values
- Also saves scores for ROC curve

**ROC Points Generation** (Manual Threshold Sweep):

```python
def roc_points(scores, cuts=5):
    scores_sorted = sorted(scores, key=lambda x: x[0])
    thresholds = []

    if not scores_sorted:
        return []

    # Sample thresholds across score range
    step = max(1, len(scores_sorted) // cuts)
    for idx in range(0, len(scores_sorted), step):
        thresholds.append(scores_sorted[idx][0])
    thresholds.append(scores_sorted[-1][0] + 1)

    out = []
    for t in thresholds:
        TP = FP = TN = FN = 0
        for score, label in scores:
            guess = 1 if score >= t else -1
            if guess == 1 and label == 1:
                TP += 1
            elif guess == 1 and label == -1:
                FP += 1
            elif guess == -1 and label == -1:
                TN += 1
            else:
                FN += 1

        TPR = TP / (TP + FN) if TP + FN else 0
        FPR = FP / (FP + TN) if FP + TN else 0
        out.append((round(t, 3), round(TPR, 3), round(FPR, 3)))

    return out
```

**How ROC Points Work**:

1. Sort all test scores
2. Select several threshold values
3. For each threshold:
   - Reclassify all points
   - Calculate TP, FP, TN, FN
   - Compute TPR and FPR
4. Each (FPR, TPR) pair is one point on ROC curve

**Main Execution with Visualization**:

```python
if __name__ == "__main__":
    data = load_rows()
    split = int(len(data) * 0.7)
    train = data[:split]
    test = data[split:split + 100]
    alphas = train_kernel_perceptron(train)
    cm, scores = confusion(test, train, alphas)

    print("Confusion matrix (TP, FP, TN, FN):", cm)

    roc_data = roc_points(scores)
    print("\nROC Points:")
    for t, tpr, fpr in roc_data:
        print("Threshold", t, "TPR", tpr, "FPR", fpr)

    # Plot ROC Curve
    plt.figure(figsize=(6, 5))
    fprs = [fpr for _, _, fpr in roc_data]
    tprs = [tpr for _, tpr, _ in roc_data]
    plt.plot(fprs, tprs, marker='o', color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Breast Cancer SVM')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### Test Results

```
Confusion matrix (TP, FP, TN, FN): (0, 0, 33, 33)

ROC Points:
Threshold -3395306202225.877 TPR 1.0 FPR 1.0
Threshold -1452415797141.551 TPR 0.606 FPR 1.0
Threshold -1081227464990.882 TPR 0.242 FPR 0.97
Threshold -654608887441.988 TPR 0.03 FPR 0.788
Threshold -436783312590.801 TPR 0.0 FPR 0.424
Threshold -227570809684.798 TPR 0.0 FPR 0.03
Threshold -227570809683.798 TPR 0.0 FPR 0.0
```

**Interpretation**:

**Confusion Matrix (0, 0, 33, 33)**:

- **TP=0**: No malignant tumors correctly identified
- **FP=0**: No false alarms
- **TN=33**: All benign tumors correctly identified
- **FN=33**: All malignant tumors missed!
- **Problem**: Model classifies everything as benign!

**ROC Curve Analysis**:

- At lowest threshold: TPR=1.0, FPR=1.0 (predicts all positive)
- At highest threshold: TPR=0.0, FPR=0.0 (predicts all negative)
- Poor discrimination (curve close to diagonal)
- AUC would be ≈ 0.5 (no better than random)

**Why Poor Performance?**

1. **Kernel perceptron limitations**: Simplified algorithm
2. **Only 3 features**: Real diagnosis needs more features
3. **Small training set**: Medical data needs more samples
4. **Class imbalance**: Likely more benign than malignant
5. **Feature scaling**: Medical measurements have different scales

**Clinical Implications**:

- In medicine, **high recall** (sensitivity) is critical
- Missing cancer (false negative) is worse than false alarm (false positive)
- Current model has 0% sensitivity - unacceptable!
- Need to adjust decision threshold or improve model

### ROC Curve Interpretation Guide

| ROC Curve Shape      | Meaning                                  |
| -------------------- | ---------------------------------------- |
| Passes through (0,1) | Perfect classifier                       |
| High on left side    | Good sensitivity at low FPR              |
| Close to diagonal    | Random guessing                          |
| Below diagonal       | Worse than random (inverted predictions) |

**Ideal Operating Point**:

- Depends on application
- Medical screening: High TPR (catch all cases)
- Spam detection: High precision (avoid false positives)
- Use Youden's Index: $J = TPR - FPR$ (maximizes both)

---

## Summary Comparison

| Program | Algorithm         | Key Feature          | Performance | Use Case                  |
| ------- | ----------------- | -------------------- | ----------- | ------------------------- |
| 12      | Linear Regression | Manual OLS           | R²=0.611    | Score prediction          |
| 17      | Naive Bayes       | Laplace smoothing    | Acc=66%     | Disease diagnosis         |
| 18      | Linear SVM        | Oversampling         | F1=94.4%    | Imbalanced spam detection |
| 19      | Linear SVM        | Clean implementation | F1=95.5%    | Balanced spam detection   |
| 20      | Poly Kernel SVM   | Degree 2 kernel      | F1=64.4%    | Student pass/fail         |
| 21      | Poly Kernel SVM   | ROC analysis         | Poor        | Cancer detection          |

## Key Takeaways

### Implementation Tips

1. **Always normalize/scale features** - Especially for distance-based algorithms
2. **Handle class imbalance** - Oversampling, undersampling, or weighted loss
3. **Use appropriate metrics** - Accuracy alone is misleading for imbalanced data
4. **Validate properly** - Train/test split, cross-validation, holdout set
5. **Start simple** - Linear models before complex kernels

### Algorithm Selection Guide

**Use Linear Regression when**:

- Predicting continuous values
- Features have linear relationship with target
- Need interpretable coefficients

**Use Naive Bayes when**:

- Features are (roughly) independent
- Need fast training and prediction
- Working with text or categorical data
- Have small training set

**Use Linear SVM when**:

- Binary classification
- High-dimensional data (many features)
- Want maximum margin classifier
- Data is linearly separable (or close)

**Use Kernel SVM when**:

- Data is NOT linearly separable
- Willing to sacrifice training speed
- Need flexible decision boundaries
- Dataset is moderate size (100s-1000s)

### Common Pitfalls

❌ Not scaling features → Poor convergence  
❌ Ignoring class imbalance → Biased predictions  
❌ Using accuracy for imbalanced data → Misleading  
❌ Overfitting with high-degree kernels → Poor generalization  
❌ Not validating assumptions → Invalid models

### Best Practices

✅ Exploratory Data Analysis first  
✅ Feature engineering and selection  
✅ Proper train/validation/test split  
✅ Cross-validation for hyperparameters  
✅ Multiple metrics (accuracy, precision, recall, F1)  
✅ Visualize results (scatter plots, ROC curves)  
✅ Compare against baselines

## References and Further Reading

- **Linear Regression**: "An Introduction to Statistical Learning" - Chapter 3
- **Naive Bayes**: "Pattern Recognition and Machine Learning" - Bishop
- **SVM**: "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- **Kernel Methods**: "Learning with Kernels" - Schölkopf & Smola
- **ROC Analysis**: "ROC Graphs: Notes and Practical Considerations" - Fawcett

---

_Generated on November 10, 2025_  
_Repository: aiml-prac_
