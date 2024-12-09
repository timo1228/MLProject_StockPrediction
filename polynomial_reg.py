import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from data import YahooDataSet

# Load the dataset
dataset = YahooDataSet()
X_train, X_test, y_train, y_test = dataset.train_and_test()

# For polynomial regression, we will try several polynomial degrees
degrees = [1, 2, 3, 4, 5]  # You can add or reduce as necessary
results = {}

for d in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Fit the linear model on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[d] = (mse, rmse, r2)
    print(f"Degree {d}: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

# Choose best polynomial degree based on RMSE or MSE
best_degree = min(results, key=lambda x: results[x][0])
print(f"Best polynomial degree: {best_degree} with MSE={results[best_degree][0]:.4f}")

# # Plot Actual vs Predicted for the best polynomial degree
# poly = PolynomialFeatures(degree=best_degree)
# X_test_poly = poly.fit_transform(X_test)
# model = LinearRegression()
# model.fit(poly.fit_transform(X_train), y_train)
# y_pred = model.predict(X_test_poly)

# plt.figure(figsize=(10, 6))
# plt.plot(y_test, label='Actual', linewidth=2)
# plt.plot(y_pred, label='Predicted', linewidth=2)
# plt.title(f"Polynomial Regression (Degree {best_degree})")
# plt.xlabel('Test Sample Index')
# plt.ylabel('Close Price')
# plt.legend()
# plt.show()

# # save plot
# plt.savefig('polynomial_regression.png')

# plot Actual vs Predicted for all polynomial degrees, save each as a separate plot
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_test_poly = poly.fit_transform(X_test)
    model = LinearRegression()
    model.fit(poly.fit_transform(X_train), y_train)
    y_pred = model.predict(X_test_poly)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linewidth=2)
    plt.title(f"Polynomial Regression (Degree {d})")
    plt.xlabel('Test Sample Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # save plot
    plt.savefig(f'polynomial_regression_{d}.png')
