import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Funzione per caricare e preparare i dati
def load_and_prepare_data(file_paths):
    """
    Carica i dati da più file CSV, li unisce, normalizza e crea input/output per il modello LSTM.
    """
    # Carica tutti i dataset e li unisce
    dataframes = [pd.read_csv(fp) for fp in file_paths]
    data = pd.concat(dataframes, axis=0)
    
    # Ordina per data e rimuovi duplicati
    data = data.sort_values(by='Date').drop_duplicates()
    
    # Normalizza i dati di chiusura
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    # Crea le serie di input/output
    X, y = [], []
    look_back = 60  # Numero di giorni per predire il giorno successivo
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# Funzione per costruire e addestrare il modello LSTM
def build_and_train_lstm(X_train, y_train, epochs=50, batch_size=32):
    """
    Definisce e addestra il modello LSTM sui dati forniti.
    """
    # Crea il modello LSTM
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    
    # Compila il modello
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Addestra il modello
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    return model

# Funzione per fare previsioni e denormalizzarle
def make_predictions(model, X, scaler):
    """
    Usa il modello addestrato per fare previsioni e denormalizza i risultati.
    """
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Funzione per visualizzare i risultati
def plot_price_trend(y_real, y_pred, title="Andamento del Prezzo con LSTM", savepath=None):
    """
    Visualizza il trend di prezzo reale e predetto su un grafico, con opzione per aggiungere una media mobile.
    """
    plt.figure(figsize=(10,6))
    
    # Applicare una media mobile ai dati per renderli più chiari
    window_size = 7  # Media mobile settimanale
    y_real_smooth = pd.Series(y_real).rolling(window=window_size).mean()
    y_pred_smooth = pd.Series(y_pred).rolling(window=window_size).mean()
    
    plt.plot(y_real_smooth, color='blue', label='Prezzo Reale')
    plt.plot(y_pred_smooth, color='red', label='Prezzo Predetto')
    plt.title(title)
    plt.xlabel('Tempo')
    plt.ylabel('Prezzo Normalizzato')
    plt.legend()
    plt.tight_layout()
    
    # Salva il grafico se viene specificato un percorso di salvataggio
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    
    plt.show()

# Main function per eseguire tutte le operazioni
def main():
    # 1. Carica e prepara i dati
    file_paths = ['data.csv/dataada.csv', 'data.csv/databtc.csv', 'data.csv/dataeth.csv']
    X, y, scaler = load_and_prepare_data(file_paths)
    
    # 2. Suddividi i dati in training e test
    split = int(0.5* len(X))  #70% per addestramento, 30% per test
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 3. Aggiusta le dimensioni per il modello LSTM (3D per input LSTM)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # 4. Costruisci e addestra il modello LSTM
    model = build_and_train_lstm(X_train, y_train)
    
    # 5. Fai previsioni sui dati di test
    predictions = make_predictions(model, X_test, scaler)
    
    # 6. Denormalizza i valori reali per confronto
    y_test_denormalized = scaler.inverse_transform([y_test])
    
    # 7. Visualizza i risultati
    plot_price_trend(y_test_denormalized.flatten(), predictions.flatten(), title="Previsione del Prezzo con LSTM", savepath="grafico.png")

if __name__ == "__main__":
    main()
