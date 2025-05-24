# Unit-I Introduction: Learning theory, Hypothesis and target class, Inductive bias and bias-variance tradeoff, Occam's razor, Limitations of inference machines, Approximation and estimation errors, Curse of dimensionality, dimensionality reduction, feature scaling, feature selection methods. 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Show curse of dimensionality (original features)
print(f"Original feature space dimension: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction (PCA)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Reduced dimensions (PCA): {X_train_pca.shape[1]}")

# Feature Selection (Univariate)
selector = SelectKBest(score_func=f_classif, k=5)
X_train_fs = selector.fit_transform(X_train_scaled, y_train)
X_test_fs = selector.transform(X_test_scaled)

# Simple model (Occam's Razor) - Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_fs, y_train)
y_pred_lr = lr.predict(X_test_fs)
print("Logistic Regression Accuracy (simple model):", accuracy_score(y_test, y_pred_lr))

# Complex model - Decision Tree (higher variance)
tree = DecisionTreeClassifier(max_depth=None, random_state=42)
tree.fit(X_train_fs, y_train)
y_pred_tree = tree.predict(X_test_fs)
print("Decision Tree Accuracy (complex model):", accuracy_score(y_test, y_pred_tree))

# Bias-Variance Tradeoff via learning curves
def plot_learning_curve(estimator, title, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), random_state=42)
    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training size")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.show()

# Plotting learning curves
plot_learning_curve(LogisticRegression(), "Learning Curve (Logistic Regression)", X_train_fs, y_train)
plot_learning_curve(DecisionTreeClassifier(), "Learning Curve (Decision Tree)", X_train_fs, y_train)
