import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from LSTM.data import YahooDataSet

# Load dataset
dataset = YahooDataSet()
X_train, X_test, y_train, y_test = dataset.train_and_test()

# Scale the data
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_Y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_Y.transform(y_test.reshape(-1, 1))

look_back = 10
days_to_predict = 1

# Create sequences
def create_sequences(X, y, look_back, days_to_predict):
    Xs, ys = [], []
    for i in range(len(X) - look_back - days_to_predict):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[(i + look_back):(i + look_back + days_to_predict)])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, look_back, days_to_predict)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, look_back, days_to_predict)

print("X_train_seq shape:", X_train_seq.shape)
print("y_train_seq shape:", y_train_seq.shape)
print("X_test_seq shape:", X_test_seq.shape)
print("y_test_seq shape:", y_test_seq.shape)

# Build the model - try different loss (e.g., 'mae' or Huber)
lstm_units_1 = 64
lstm_units_2 = 32
dropout_rate = 0.2
learning_rate = 0.001
batch_size = 32
epochs = 60

model = Sequential()
model.add(Bidirectional(LSTM(lstm_units_1, return_sequences=True), input_shape=(look_back, X_train_seq.shape[2])))
model.add(Dropout(dropout_rate))
model.add(LSTM(lstm_units_2, return_sequences=False))
model.add(Dropout(dropout_rate))
model.add(Dense(days_to_predict))

optimizer = Adam(learning_rate=learning_rate)
# Use MAE as the loss function
model.compile(optimizer=optimizer, loss='mae')

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1
)

# Predictions
y_pred_seq = model.predict(X_test_seq)

# Reshape for inverse scaling
y_pred_flat = y_pred_seq.reshape(-1, 1)
y_test_flat = y_test_seq.reshape(-1, 1)

y_pred_rescaled = scaler_Y.inverse_transform(y_pred_flat).reshape(-1, days_to_predict)
y_test_rescaled = scaler_Y.inverse_transform(y_test_flat).reshape(-1, days_to_predict)

# Metrics: MSE, RMSE, MAE, MAPE
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
# MAPE: mean absolute percentage error
# Avoid division by zero by adding a small epsilon if needed
epsilon = 1e-7
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / (y_test_rescaled + epsilon))) * 100

print("MAE Loss Results:")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape, "%")

# Naive baseline (for comparison)
naive_avg_scaled = np.mean(X_test_seq[:,:,0], axis=1)  # Average over look_back days
naive_pred_scaled = np.tile(naive_avg_scaled.reshape(-1, 1), (1, days_to_predict))

naive_pred_flat = naive_pred_scaled.reshape(-1, 1)
naive_pred_rescaled = scaler_Y.inverse_transform(naive_pred_flat).reshape(-1, days_to_predict)

mse_naive = mean_squared_error(y_test_rescaled, naive_pred_rescaled)
rmse_naive = np.sqrt(mse_naive)
mae_naive = mean_absolute_error(y_test_rescaled, naive_pred_rescaled)
mape_naive = np.mean(np.abs((y_test_rescaled - naive_pred_rescaled) / (y_test_rescaled + epsilon))) * 100

print("Naive Baseline Results:")
print("MSE:", mse_naive)
print("RMSE:", rmse_naive)
print("MAE:", mae_naive)
print("MAPE:", mape_naive, "%")

# Plot predictions vs actual (for first predicted day)
plt.figure(figsize=(10,6))
plt.plot(y_test_rescaled[:, 0], label='Actual', linewidth=2)
plt.plot(y_pred_rescaled[:, 0], label='LSTM Predicted', linewidth=2)
plt.title("Prediction vs Actual - Next {} Days (First Day)".format(days_to_predict))
plt.xlabel('Test Sequence Index')
plt.ylabel('Close Price')
plt.legend()
plt.show()

plt.savefig('lstm_predictions.png')

# Plot training history
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training History (Loss: MAE)")
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend()
plt.show()

plt.savefig('lstm_training_history.png')
