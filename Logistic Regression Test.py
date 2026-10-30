
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# Select target horizon
horizon = 10
target_col = f"target_{horizon}"

# Load data in chronological order
data = pd.read_csv("Data/SQQQ/SQQQ_stats.csv", index_col=0)
data = data.sort_index().dropna()

# Exclude ALL target columns from features to prevent leakage
X = data.drop(
    columns=[c for c in data.columns if c.startswith("target_")]
)
y = data[target_col].astype(int)

# Chronological 80/20 split
split_id = int(len(data) * 0.8)

X_train = X.iloc[:split_id]
X_test = X.iloc[split_id:]
y_train = y.iloc[:split_id]
y_test = y.iloc[split_id:]

# Scaling + logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5, gap=horizon)

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_lr = grid.best_estimator_
y_pred = best_lr.predict(X_test)

# Majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)

print("Best parameters:", grid.best_params_)
print("Best CV balanced accuracy:", grid.best_score_)

print("\nBaseline accuracy:",
      accuracy_score(y_test, baseline_pred))
print("Baseline balanced accuracy:",
      balanced_accuracy_score(y_test, baseline_pred))

print("\nLogistic regression accuracy:",
      accuracy_score(y_test, y_pred))
print("Logistic regression balanced accuracy:",
      balanced_accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
