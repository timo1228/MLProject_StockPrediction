import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def train_and_test(self):
        pass

class YahooDataSet(Dataset):
    def __init__(self):
        path = "./data/nasdq.csv"
        self.inited = False
        # Initialize the data
        data = pd.read_csv(path)

        # Date column: Convert to datetime format
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Rename columns to remove spaces
        data.columns = data.columns.str.replace(' ', '')

        # Feature Engineering
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Close'].rolling(window=30).std()
        data['Rolling_Mean_Close'] = data['Close'].rolling(window=30).mean()
        data.dropna(inplace=True)

        self.data = data

        # Define target variable and features
        X = data.drop(['Close'], axis=1)
        y = data['Close']

        # Handle infinite values and NaNs
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(method='ffill', inplace=True)

        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values

        self.inited = True
        print("Data initialized and scaled successfully!")

    def train_and_test(self):
        if self.inited:
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            raise RuntimeError('Dataset not yet initialized!')

def test():
    dataset = YahooDataSet()
    X_train, X_test, y_train, y_test = dataset.train_and_test()

    # Baseline Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))
    print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))

    # Polynomial Regression
    from sklearn.preprocessing import PolynomialFeatures

    degree = 2  # Adjust as needed
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_lr = LinearRegression()
    poly_lr.fit(X_train_poly, y_train)
    y_pred_poly = poly_lr.predict(X_test_poly)
    print(f"Polynomial Regression (degree={degree}) R2 Score:", r2_score(y_test, y_pred_poly))
    print(f"Polynomial Regression (degree={degree}) RMSE:", mean_squared_error(y_test, y_pred_poly, squared=False))

    # Plotting Polynomial Regression Results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred_poly, label='Predicted')
    plt.title('Polynomial Regression Predictions vs Actual')
    plt.legend()
    plt.show()

    # Prepare data for LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # Scaling target variable
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    # Create sequences
    def create_sequences(X, y, time_steps=10):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 10  # Adjust as needed
    X_train_lstm, y_train_lstm = create_sequences(X_train, y_train_scaled, time_steps)
    X_test_lstm, y_test_lstm = create_sequences(X_test, y_test_scaled, time_steps)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32,
              validation_data=(X_test_lstm, y_test_lstm), verbose=1, shuffle=False)

    # Make predictions
    y_pred_lstm = model.predict(X_test_lstm)

    # Inverse transform predictions
    y_pred_lstm_inv = scaler_y.inverse_transform(y_pred_lstm)
    y_test_lstm_inv = scaler_y.inverse_transform(y_test_lstm)

    # Evaluate the model
    print("LSTM R2 Score:", r2_score(y_test_lstm_inv, y_pred_lstm_inv))
    print("LSTM RMSE:", mean_squared_error(y_test_lstm_inv, y_pred_lstm_inv, squared=False))

    # Plotting LSTM Results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_lstm_inv, label='Actual')
    plt.plot(y_pred_lstm_inv, label='Predicted')
    plt.title('LSTM Predictions vs Actual')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()
