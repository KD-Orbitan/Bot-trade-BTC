import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BTCPricePredictor:
    def __init__(self, lookback=24):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_columns = None
    
    def load_data(self, file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback, 3])  # Cột 'close' là cột thứ 4 (index 3)
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, test_size=0.2):
        self.feature_columns = df.columns.tolist()
        df_scaled = self.scaler.fit_transform(df)
        X, y = self.create_sequences(df_scaled)
        split_idx = int(len(X) * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    def build_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=input_shape)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        os.makedirs('models', exist_ok=True)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint('models/best_model.h5', monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test).flatten()
        y_test_inv = self.scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1)], axis=1))[:, 3]
        y_pred_inv = self.scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred), 3)), y_pred.reshape(-1, 1)], axis=1))[:, 3]
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")
        return y_pred_inv, mae, mse, r2
    
    def plot_predictions(self, y_test, y_pred):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test, label='Actual', alpha=0.8)
        plt.plot(y_pred, label='Predicted', alpha=0.8)
        plt.legend()
        plt.savefig(f'plots/predictions.png')
        plt.close()

def main():
    predictor = BTCPricePredictor(lookback=24)
    data = predictor.load_data('binance_history/BTCUSDT_1h_processed.csv')
    X_train, X_test, y_train, y_test = predictor.prepare_data(data, test_size=0.2)
    model = predictor.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = predictor.train_model(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
    y_pred, mae, mse, r2 = predictor.evaluate_model(X_test, y_test)
    predictor.plot_predictions(y_test, y_pred)
    print("Training completed.")

if __name__ == "__main__":
    main()
