import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BTCPricePredictor:
    def __init__(self, lookback=24):
        """
        Initialize the BTC price predictor
        lookback: Number of previous hours to use for prediction
        """
        self.lookback = lookback
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load and prepare the dataset"""
        # Load data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df
    
    def create_sequences(self, data):
        """Create sequences for LSTM input"""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data.iloc[i + self.lookback]['close'])
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for training"""
        # Store feature columns
        self.feature_columns = df.columns.tolist()
        
        # Create sequences
        X, y = self.create_sequences(df)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        # Create directory for model checkpoints
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print("\nModel Performance Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        return y_pred, mae, rmse, r2
    
    def plot_predictions(self, y_test, y_pred, title="BTC Price Predictions vs Actual"):
        """Plot predictions against actual values"""
        plt.figure(figsize=(15, 6))
        plt.plot(y_test, label='Actual', alpha=0.8)
        plt.plot(y_pred, label='Predicted', alpha=0.8)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/predictions_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        plt.close()
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(15, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'plots/training_history_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        plt.close()

def main():
    # Initialize predictor
    predictor = BTCPricePredictor(lookback=24)  # Use 24 hours of data to predict next hour
    
    # Load data
    data = predictor.load_data('binance_history/BTCUSDT_1h_processed.csv')
    print("Data loaded successfully")
    print(f"Dataset shape: {data.shape}")
    print("\nFeatures:", data.columns.tolist())
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data(data, test_size=0.2)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Build model
    model = predictor.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    print("\nModel Summary:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = predictor.train_model(
        X_train, y_train,
        X_test, y_test,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred, mae, rmse, r2 = predictor.evaluate_model(X_test, y_test)
    
    # Plot results
    predictor.plot_predictions(y_test, y_pred)
    predictor.plot_training_history(history)
    
    print("\nTraining completed. Check 'plots' directory for visualizations.")

if __name__ == "__main__":
    main() 