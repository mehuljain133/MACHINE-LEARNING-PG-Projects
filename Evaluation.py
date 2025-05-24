# Unit IV Evaluation: Performance evaluation metrics, ROC Curves, Validation methods, Biasvariance decomposition, Model complexity

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

# --- Performance Metrics ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

# --- ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# --- Cross Validation ---
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# --- Validation Curve (Model Complexity) ---
param_range = np.arange(1, 21)  # Max iterations as proxy for complexity
train_scores, test_scores = validation_curve(
    LogisticRegression(max_iter=10000),
    X, y,
    param_name="max_iter",
    param_range=param_range,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(param_range, train_mean, label="Training score", color="blue")
plt.plot(param_range, test_mean, label="Cross-validation score", color="green")
plt.title("Validation Curve (Model Complexity by max_iter)")
plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()

# --- Bias-Variance Conceptual Plot ---

# This is a conceptual illustration: plotting hypothetical bias-variance decomposition.
bias_squared = np.linspace(0.5, 0.05, 10)
variance = np.linspace(0.05, 0.5, 10)
total_error = bias_squared + variance

plt.figure()
plt.plot(bias_squared, label="Bias^2", marker='o')
plt.plot(variance, label="Variance", marker='o')
plt.plot(total_error, label="Total Error", marker='o')
plt.title("Bias-Variance Decomposition (Conceptual)")
plt.xlabel("Model Complexity (increasing)")
plt.ylabel("Error")
plt.legend()
plt.grid()
plt.show()
