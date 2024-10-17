import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Funzione per caricare e preparare i dati
def load_and_prepare_data(file_path, look_back=45, smoothing_window=5):
    """
    Carica i dati da un file CSV, filtra per il periodo desiderato, applica una media mobile,
    normalizza e crea input/output per il modello LSTM.
    """
    # Carica il dataset
    data = pd.read_csv(file_path)
    
    # Assicurarsi che la colonna 'Date' sia in formato datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Filtra i dati per il periodo dal 2021-01-01 al 2022-12-31
    start_date = '2021-01-01'
    end_date = '2022-12-31'
    mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
    data = data.loc[mask]
    
    # Ordina per data e rimuove duplicati
    data = data.sort_values(by='Date').drop_duplicates().reset_index(drop=True)
    
    # Applica una media mobile per smorzare la volatilità
    data['Close_Smoothed'] = data['Close'].rolling(window=smoothing_window).mean()
    
    # Rimuovi le righe con valori NaN causati dalla media mobile
    data = data.dropna().reset_index(drop=True)
    
    # Normalizza i dati di chiusura smorzati
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close_Smoothed']])
    
    # Crea le serie di input/output
    X, y = [], []
    dates = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
        dates.append(data['Date'].iloc[i])  # Aggiungi la data corrispondente al target
    return np.array(X), np.array(y), scaler, dates

# Funzione per costruire e addestrare il modello LSTM con meno dropout e più neuroni
def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=16):
    """
    Definisce e addestra il modello LSTM sui dati forniti, utilizzando Early Stopping.
    """
    # Crea il modello LSTM
    model = Sequential([
        LSTM(40, return_sequences=True, input_shape=(X_train.shape[1], 1), kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        LSTM(20),
        Dropout(0.4),
        Dense(1)
    ])
    
    # Compila il modello
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Callback per l'early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Addestra il modello
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    
    return model, history

# Funzione per fare previsioni e denormalizzarle
def make_predictions(model, X, scaler):
    """
    Usa il modello addestrato per fare previsioni e denormalizza i risultati.
    """
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Funzione per visualizzare i risultati del training
def plot_training_history(history, savepath=None):
    """
    Visualizza l'andamento della loss durante l'addestramento.
    """
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Loss di Addestramento')
    plt.plot(history.history['val_loss'], label='Loss di Validazione')
    plt.title('Andamento della Loss durante l\'Addestramento')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

# Funzione per visualizzare i risultati
def plot_price_trend(y_real, y_pred, dates, title="Andamento del Prezzo Bitcoin con LSTM", savepath=None):
    """
    Visualizza il trend di prezzo reale e predetto su un grafico con le date sull'asse x.
    """
    plt.figure(figsize=(12,6))
    
    plt.plot(dates, y_real, color='blue', label='Prezzo Reale')
    plt.plot(dates, y_pred, color='red', label='Prezzo Predetto')
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Prezzo')
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=45)
    
    # Salva il grafico se viene specificato un percorso di salvataggio
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    
    plt.show()

# Main function per eseguire tutte le operazioni
def main():
    # 1. Carica e prepara i dati
    file_path = 'data.csv/databtc.csv'
    look_back = 45  # Aumento il look back a 45 giorni
    smoothing_window = 5  # Leggero aumento del smoothing
    X, y, scaler, dates = load_and_prepare_data(file_path, look_back, smoothing_window)
    
    # 2. Suddividi i dati in training e test
    split = int(0.8 * len(X))  # 80% per addestramento, 20% per test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_train, dates_test = dates[:split], dates[split:]
    
    # 3. Aggiusta le dimensioni per il modello LSTM (3D per input LSTM)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # 4. Costruisci e addestra il modello LSTM
    model, history = build_and_train_lstm(X_train, y_train, epochs=50, batch_size=16)
    
    # 5. Visualizza la storia dell'addestramento
    plot_training_history(history, savepath='training_history.png')
    
    # 6. Fai previsioni sui dati di test
    predictions = make_predictions(model, X_test, scaler)
    
    # 7. Denormalizza i valori reali per confronto
    y_test_denormalized = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # 8. Visualizza i risultati
    plot_price_trend(y_test_denormalized.flatten(), predictions.flatten(), dates_test, title="Previsione del Prezzo Bitcoin con LSTM", savepath="grafico.png")

if __name__ == "__main__":
    main()
