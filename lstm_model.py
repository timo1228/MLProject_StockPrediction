import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from data import YahooDataSet

# Load the dataset
dataset = YahooDataSet()
X_train, X_test, y_train, y_test = dataset.train_and_test()

# Scale the data (MinMaxScaler to scale between 0 and 1)
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_Y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_Y.transform(y_test.reshape(-1, 1))

def create_sequences(X, y, look_back, days_to_predict):
    Xs, ys = [], []
    for i in range(len(X) - look_back - days_to_predict):
        Xs.append(X[i:(i + look_back)])
        ys.append(y[(i + look_back):(i + look_back + days_to_predict)])
    return np.array(Xs), np.array(ys)

# Hyperparameter ranges
look_back_list = [10, 20, 30]              # Different sequence lengths
days_to_predict_list = [1]         # Different forecasting horizons
lstm_units_list = [(64,32), (128,64)]  # Different LSTM layer configurations
dropout_rates = [0.2]             # Different dropout rates
learning_rates = [0.001]       # Different learning rates
batch_sizes = [256]                 # Different batch sizes
epochs = 200

# Results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Prepare a CSV file to log results
results_file = 'results/hyperparam_results.csv'
if not os.path.exists(results_file):
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'run_number',
            'look_back',
            'days_to_predict',
            'lstm_units_1',
            'lstm_units_2',
            'dropout_rate',
            'learning_rate',
            'batch_size',
            'LSTM_MSE',
            'LSTM_RMSE',
            'Naive_MSE',
            'Naive_RMSE'
        ])


run_number = 0

for look_back in look_back_list:
    for days_to_predict in days_to_predict_list:
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, look_back, days_to_predict)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, look_back, days_to_predict)

        for (lstm_1_units, lstm_2_units) in lstm_units_list:
            for dropout_rate in dropout_rates:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        run_number += 1
                        run_dir = f"results/run_{run_number}_lookback{look_back}_days{days_to_predict}_lstm{lstm_1_units}-{lstm_2_units}_dropout{dropout_rate}_lr{lr}_batch{batch_size}"
                        os.makedirs(run_dir, exist_ok=True)

                        # Build the model
                        model = Sequential()
                        model.add(Bidirectional(LSTM(lstm_1_units, return_sequences=True),
                                                input_shape=(look_back, X_train_seq.shape[2])))
                        model.add(Dropout(dropout_rate))
                        model.add(LSTM(lstm_2_units, return_sequences=False))
                        model.add(Dropout(dropout_rate))
                        model.add(Dense(days_to_predict))

                        optimizer = Adam(learning_rate=lr)
                        model.compile(optimizer=optimizer, loss='mse')

                        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

                        # Train the model
                        history = model.fit(X_train_seq, y_train_seq,
                                            validation_data=(X_test_seq, y_test_seq),
                                            epochs=epochs,
                                            batch_size=batch_size,
                                            callbacks=[early_stopping],
                                            verbose=1)

                        # ======================================
                        # Predict on Test Data
                        # ======================================
                        y_pred_seq = model.predict(X_test_seq)
                        y_pred_flat = y_pred_seq.reshape(-1, 1)
                        y_test_flat = y_test_seq.reshape(-1, 1)

                        y_pred_rescaled = scaler_Y.inverse_transform(y_pred_flat).reshape(-1, days_to_predict)
                        y_test_rescaled = scaler_Y.inverse_transform(y_test_flat).reshape(-1, days_to_predict)

                        # LSTM metrics (Test)
                        mse_lstm_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)
                        rmse_lstm_test = np.sqrt(mse_lstm_test)

                        # Naive baseline (Test)
                        naive_avg_scaled = np.mean(X_test_seq[:,:,0], axis=1)
                        naive_pred_scaled = np.tile(naive_avg_scaled.reshape(-1, 1), (1, days_to_predict))

                        naive_pred_flat = naive_pred_scaled.reshape(-1, 1)
                        naive_pred_rescaled = scaler_Y.inverse_transform(naive_pred_flat).reshape(-1, days_to_predict)

                        mse_naive_test = mean_squared_error(y_test_rescaled, naive_pred_rescaled)
                        rmse_naive_test = np.sqrt(mse_naive_test)

                        # ======================================
                        # Predict on ALL Training Data
                        # ======================================
                        y_train_pred = model.predict(X_train_seq)
                        y_train_pred_flat = y_train_pred.reshape(-1, 1)
                        y_train_actual_flat = y_train_seq.reshape(-1, 1)

                        y_train_pred_rescaled = scaler_Y.inverse_transform(y_train_pred_flat).reshape(-1, days_to_predict)
                        y_train_actual_rescaled = scaler_Y.inverse_transform(y_train_actual_flat).reshape(-1, days_to_predict)

                        # LSTM metrics (Training)
                        mse_lstm_train = mean_squared_error(y_train_actual_rescaled, y_train_pred_rescaled)
                        rmse_lstm_train = np.sqrt(mse_lstm_train)

                        # Print metrics
                        print(f"Run #{run_number}: LSTM_Test_MSE={mse_lstm_test}, LSTM_Test_RMSE={rmse_lstm_test}, Naive_Test_MSE={mse_naive_test}, Naive_Test_RMSE={rmse_naive_test}, LSTM_Train_MSE={mse_lstm_train}, LSTM_Train_RMSE={rmse_lstm_train}")

                        # Save metrics and configuration
                        with open(results_file, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                run_number,
                                look_back,
                                days_to_predict,
                                lstm_1_units,
                                lstm_2_units,
                                dropout_rate,
                                lr,
                                batch_size,
                                mse_lstm_test,
                                rmse_lstm_test,
                                mse_naive_test,
                                rmse_naive_test,
                                mse_lstm_train,
                                rmse_lstm_train
                            ])

                        # ======================================
                        # Plot training history
                        # ======================================
                        plt.figure(figsize=(10,6))
                        plt.plot(history.history['loss'], label='Train Loss')
                        plt.plot(history.history['val_loss'], label='Val Loss')
                        plt.title("Training History")
                        plt.xlabel('Epoch')
                        plt.ylabel('MSE Loss')
                        plt.legend()
                        plt.savefig(os.path.join(run_dir, 'training_history.png'))
                        plt.close()

                        # ======================================
                        # Plot LSTM predictions vs actual (first day) on Test Data
                        # ======================================
                        plt.figure(figsize=(10,6))
                        plt.plot(y_test_rescaled[:, 0], label='Actual')
                        plt.plot(y_pred_rescaled[:, 0], label='LSTM Predicted')
                        plt.title(f"LSTM vs Actual - Next {days_to_predict} Days (Test Data, First Day)")
                        plt.xlabel('Test Sequence Index')
                        plt.ylabel('Close Price')
                        plt.legend()
                        plt.savefig(os.path.join(run_dir, 'lstm_predictions_test.png'))
                        plt.close()

                        # ======================================
                        # Plot naive baseline predictions vs actual (first day) on Test Data
                        # ======================================
                        plt.figure(figsize=(10,6))
                        plt.plot(y_test_rescaled[:, 0], label='Actual')
                        plt.plot(naive_pred_rescaled[:, 0], label='Naive Predicted')
                        plt.title(f"Naive vs Actual - Next {days_to_predict} Days (Test Data, First Day)")
                        plt.xlabel('Test Sequence Index')
                        plt.ylabel('Close Price')
                        plt.legend()
                        plt.savefig(os.path.join(run_dir, 'naive_predictions_test.png'))
                        plt.close()

                        # ======================================
                        # Plot LSTM predictions vs actual (first day) on Training Data
                        # ======================================
                        plt.figure(figsize=(10,6))
                        plt.plot(y_train_actual_rescaled[:, 0], label='Actual (Train)')
                        plt.plot(y_train_pred_rescaled[:, 0], label='LSTM Predicted (Train)')
                        plt.title(f"LSTM vs Actual - Next {days_to_predict} Days (Train Data, First Day)")
                        plt.xlabel('Train Sequence Index')
                        plt.ylabel('Close Price')
                        plt.legend()
                        plt.savefig(os.path.join(run_dir, 'lstm_predictions_train.png'))
                        plt.close()

                        # ======================================
                        # Debugging: One Training Sequence
                        # ======================================
                        train_sample_idx = 0
                        X_debug = X_train_seq[train_sample_idx:train_sample_idx+1]
                        y_debug_actual = y_train_seq[train_sample_idx:train_sample_idx+1]
                        y_debug_pred = model.predict(X_debug)

                        y_debug_pred_flat = y_debug_pred.reshape(-1, 1)
                        y_debug_actual_flat = y_debug_actual.reshape(-1, 1)
                        y_debug_pred_rescaled = scaler_Y.inverse_transform(y_debug_pred_flat).reshape(-1, days_to_predict)
                        y_debug_actual_rescaled = scaler_Y.inverse_transform(y_debug_actual_flat).reshape(-1, days_to_predict)

                        plt.figure(figsize=(10,6))
                        plt.plot(y_debug_actual_rescaled[0], label='Actual (Train Seq)')
                        plt.plot(y_debug_pred_rescaled[0], label='Predicted (Train Seq)')
                        plt.title('Training Data Sample Prediction Debug')
                        plt.xlabel('Days into Future')
                        plt.ylabel('Close Price')
                        plt.legend()
                        plt.savefig(os.path.join(run_dir, 'training_seq_debug.png'))
                        plt.close()

                        # Save summary
                        summary_file = os.path.join(run_dir, 'summary.txt')
                        with open(summary_file, 'w') as f:
                            f.write(f"Run #{run_number}\n")
                            f.write(f"Look_back: {look_back}\n")
                            f.write(f"Days_to_predict: {days_to_predict}\n")
                            f.write(f"LSTM units: {lstm_1_units}, {lstm_2_units}\n")
                            f.write(f"Dropout: {dropout_rate}\n")
                            f.write(f"Learning rate: {lr}\n")
                            f.write(f"Batch size: {batch_size}\n")
                            f.write(f"LSTM Test MSE: {mse_lstm_test}\n")
                            f.write(f"LSTM Test RMSE: {rmse_lstm_test}\n")
                            f.write(f"Naive Test MSE: {mse_naive_test}\n")
                            f.write(f"Naive Test RMSE: {rmse_naive_test}\n")
                            f.write(f"LSTM Train MSE: {mse_lstm_train}\n")
                            f.write(f"LSTM Train RMSE: {rmse_lstm_train}\n")
                            f.write("Training sequence prediction debug completed.\n")

                        # Save model
                        model.save(os.path.join(run_dir, 'model.h5'))

                        # Clear model from memory
                        del model