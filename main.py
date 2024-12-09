import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from data import YahooDataSet

##############################
# Load the dataset
##############################
dataset = YahooDataSet()
X_train, X_test, y_train, y_test = dataset.train_and_test()

##############################
# Polynomial Regression
##############################
def train_polynomial_regression(X_train, y_train, X_test, y_test, degree=2):
    # Transform features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Fit linear regression on polynomial features
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    
    # Predictions
    y_train_pred = poly_model.predict(X_train_poly)
    y_test_pred = poly_model.predict(X_test_poly)
    
    # Evaluation
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"Polynomial Regression (degree={degree}) Results:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    return poly_model, poly, y_test_pred

# Example: Try polynomial regression with degree 2 and 3
_, _, poly_y2_pred = train_polynomial_regression(X_train, y_train, X_test, y_test, degree=2)
_, _, poly_y3_pred = train_polynomial_regression(X_train, y_train, X_test, y_test, degree=3)

# Plotting polynomial regression predictions vs actual
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Prices")
plt.plot(poly_y2_pred, label="Poly Deg 2 Predictions")
plt.plot(poly_y3_pred, label="Poly Deg 3 Predictions")
plt.title("Polynomial Regression Predictions vs Actual")
plt.xlabel("Test Data Points")
plt.ylabel("Close Price")
plt.legend()
plt.show()


##############################
# LSTM Implementation
##############################
def create_lstm_data(X, y, timesteps=30):
    # Convert the dataset into a time-series format suitable for LSTM
    # Here timesteps=30 means we use the past 30 days of features to predict the next day.
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i-timesteps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# Assume we want to use a window size of 30 timesteps
timesteps = 30

# Re-split the data based on timesteps alignment
# The original dataset was split as 80% train, 20% test, so we must ensure consistent indexing
num_train = int(len(y_train))
num_test = int(len(y_test))

# Combine train & test back temporarily to create a continuous series
X_combined = np.concatenate([X_train, X_test], axis=0)
y_combined = np.concatenate([y_train, y_test], axis=0)

X_all, y_all = create_lstm_data(X_combined, y_combined, timesteps)

# After transformation, we need to re-split according to our original ratio
# Our initial train split was 80%, after building sequences the first `timesteps` samples can't be used
split_index = num_train - timesteps

X_train_lstm = X_all[:split_index]
y_train_lstm = y_all[:split_index]
X_test_lstm = X_all[split_index:]
y_test_lstm = y_all[split_index:]

print("LSTM Data Shapes:")
print("X_train_lstm:", X_train_lstm.shape)
print("y_train_lstm:", y_train_lstm.shape)
print("X_test_lstm:", X_test_lstm.shape)
print("y_test_lstm:", y_test_lstm.shape)

def build_lstm_model(input_shape, units=50, dropout=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units))
    model.add(Dropout(dropout))
    model.add(Dense(1))  # Predicting a single value: next close price
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))
    return model

# Build and train the LSTM model
model = build_lstm_model(input_shape=(timesteps, X_train_lstm.shape[2]), units=50, dropout=0.2)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_lstm, y_train_lstm,
                    validation_split=0.1,
                    epochs=30,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# Evaluate the LSTM model
y_train_pred_lstm = model.predict(X_train_lstm)
y_test_pred_lstm = model.predict(X_test_lstm)

lstm_train_mse = mean_squared_error(y_train_lstm, y_train_pred_lstm)
lstm_test_mse = mean_squared_error(y_test_lstm, y_test_pred_lstm)
lstm_test_r2 = r2_score(y_test_lstm, y_test_pred_lstm)

print("LSTM Results:")
print(f"Train MSE: {lstm_train_mse:.4f}")
print(f"Test MSE: {lstm_test_mse:.4f}")
print(f"Test R²: {lstm_test_r2:.4f}")

# Plot LSTM predictions vs actual values
plt.figure(figsize=(10,5))
plt.plot(y_test_lstm, label="Actual Prices")
plt.plot(y_test_pred_lstm, label="LSTM Predictions")
plt.title("LSTM Predictions vs Actual")
plt.xlabel("Test Data Points (after timesteps)")
plt.ylabel("Close Price")
plt.legend()
plt.show()

plt.savefig('lstm_prediction_two.png')

##############################
# Analysis and Further Steps
##############################
# - Consider tuning hyperparameters of the LSTM (units, learning rate, epochs, timesteps).
# - Try different polynomial degrees and compare the metrics.
# - Experiment with different feature subsets or scaling methods.
# - For the LSTM, consider adding more LSTM layers, adjusting dropout, or using GRU.
# - Evaluate over different segments of the test set or perform a walk-forward validation approach.
