#runnare direttamente questo file per eseguire la predictions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Function to load data
def load_data(file_path):
    """
    Loads data from a CSV file and returns the closing prices and dates.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Ensure 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort by date and remove duplicates
    data = data.sort_values(by='Date').drop_duplicates().reset_index(drop=True)
    
    return data

# Function to apply moving average smoothing
def apply_smoothing(data, smoothing_window=5):
    """
    Applies moving average smoothing to the 'Close' column.
    """
    data['Close_Smoothed'] = data['Close'].rolling(window=smoothing_window).mean()
    # Remove rows with NaN values caused by moving average
    data = data.dropna().reset_index(drop=True)
    return data

# Function to create sequences
def create_sequences(data, look_back=45):
    """
    Creates input/output sequences for the LSTM model.
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Function to build and train the LSTM model
def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=16):
    """
    Defines and trains the LSTM model on the provided data using Early Stopping.
    """
    # Create the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        LSTM(25),
        Dropout(0.2),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    
    return model, history

# Function to make predictions and inverse transform them
def make_predictions(model, X, scaler):
    """
    Uses the trained model to make predictions and inversely transforms the results.
    """
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Function to plot training history
def plot_training_history(history, savepath=None):
    """
    Plots the loss trend during training.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Trend During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

# Function to plot results
def plot_price_trend(dates, y_real, y_pred, title="Bitcoin Price Prediction with LSTM", savepath=None):
    """
    Plots the actual and predicted price trends on a graph with dates on the x-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_real, color='blue', label='Actual Price')
    plt.plot(dates, y_pred, color='red', label='Predicted Price')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Save the plot if a save path is specified
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    
    plt.show()

# Main function to execute all operations
def main():
    # 1. Load the data
    file_path = 'data.csv/databtc.csv'  # Update this path to your actual file location
    look_back = 45  # Look back window
    smoothing_window = 5  # Smoothing window
    data = load_data(file_path)
    
    # Extract the 'Close' prices and dates
    close_prices = data['Close'].values.reshape(-1, 1)
    dates = data['Date'].values
    
    # 2. Split data into training and test sets (90% training, 10% testing)
    split_index = int(0.9 * len(close_prices))
    train_prices = close_prices[:split_index]
    test_prices = close_prices[split_index:]
    train_dates = dates[:split_index]
    test_dates = dates[split_index:]
    
    # 3. Apply moving average smoothing separately to training and test data
    train_data = pd.DataFrame({'Date': train_dates, 'Close': train_prices.flatten()})
    test_data = pd.DataFrame({'Date': test_dates, 'Close': test_prices.flatten()})
    
    train_data = apply_smoothing(train_data, smoothing_window)
    test_data = apply_smoothing(test_data, smoothing_window)
    
    # After smoothing, extract the smoothed close prices
    train_prices_smoothed = train_data['Close_Smoothed'].values.reshape(-1, 1)
    test_prices_smoothed = test_data['Close_Smoothed'].values.reshape(-1, 1)
    train_dates_smoothed = train_data['Date'].values
    test_dates_smoothed = test_data['Date'].values
    
    # 4. Scale the data using MinMaxScaler fitted on training data only
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_prices_smoothed)
    scaled_test = scaler.transform(test_prices_smoothed)
    
    # 5. Create sequences for training and testing
    X_train, y_train = create_sequences(scaled_train, look_back)
    X_test, y_test = create_sequences(scaled_test, look_back)
    
    # Adjust dates for sequences
    train_sequence_dates = train_dates_smoothed[look_back:]
    test_sequence_dates = test_dates_smoothed[look_back:]
    
    # 6. Reshape for LSTM input (3D)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # 7. Build and train the LSTM model
    model, history = build_and_train_lstm(X_train, y_train, epochs=50, batch_size=16)
    
    # 8. Visualize training history
    plot_training_history(history, savepath='training_history.png')
    
    # 9. Make predictions on the test data
    predictions = make_predictions(model, X_test, scaler)
    
    # 10. Inversely transform the actual test prices for comparison
    y_test_denormalized = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 11. Visualize the results
    plot_price_trend(
        test_sequence_dates,
        y_test_denormalized.flatten(),
        predictions.flatten(),
        title="Bitcoin Price Prediction with LSTM",
        savepath="prediction_plot.png"
    )
    
    # Optional: Evaluate model performance using metrics like RMSE
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_denormalized, predictions))
    print(f"Test RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()
