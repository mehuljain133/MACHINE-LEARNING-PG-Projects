# Unit-II Regression: Linear regression with one variable, Linear regression with multiple variable, Gradient Descent, Logistic Regression, Polynomial regression, over-fitting, regularization. performance evaluation metrics, validation methods.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Generate synthetic data for regression
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# ----------------- Linear Regression (Univariate) -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("Linear Regression (Univariate):")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.title("Linear Regression (Univariate)")
plt.xlabel("X")
plt.ylabel("y")
plt.grid()
plt.show()

# ----------------- Linear Regression (Multivariate) -----------------
X_multi = np.random.rand(100, 3)
y_multi = 1 + 2 * X_multi[:, 0] + 3 * X_multi[:, 1] - 4 * X_multi[:, 2] + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
y_pred_multi = lr_multi.predict(X_test)

print("\nLinear Regression (Multivariate):")
print("MSE:", mean_squared_error(y_test, y_pred_multi))
print("R² Score:", r2_score(y_test, y_pred_multi))

# ----------------- Polynomial Regression -----------------
poly_model = make_pipeline(PolynomialFeatures(degree=4), LinearRegression())
poly_model.fit(X_train[:, [0]], y_train)
y_poly_pred = poly_model.predict(X_test[:, [0]])

print("\nPolynomial Regression:")
print("MSE:", mean_squared_error(y_test, y_poly_pred))
print("R² Score:", r2_score(y_test, y_poly_pred))

# ----------------- Overfitting demonstration -----------------
# Degree 1 vs Degree 10 Polynomial
plt.figure()
for degree in [1, 10]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train[:, [0]], y_train)
    X_range = np.linspace(0, 1, 100).reshape(-1, 1)
    plt.plot(X_range, model.predict(X_range), label=f'Degree {degree}')
plt.scatter(X_train[:, 0], y_train, color='black')
plt.title("Overfitting Example")
plt.legend()
plt.grid()
plt.show()

# ----------------- Regularization (Ridge & Lasso) -----------------
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print("\nRidge Regression R²:", ridge.score(X_test, y_test))
print("Lasso Regression R²:", lasso.score(X_test, y_test))

# ----------------- Logistic Regression -----------------
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)  # Feature scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot()
plt.grid(False)
plt.show()

# ----------------- Validation Methods -----------------
scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
print("\nCross-Validation Accuracy Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
