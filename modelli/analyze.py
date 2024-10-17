import pandas as pd
from modelli.metrics import calculate_daily_returns, calculate_correlation

def filter_data_by_date(data, start_date, end_date):
    """
    Filtra i dati per ciascuna criptovaluta in base a un intervallo di date specifico.
    """
    filtered_data = {}
    for name, df in data.items():
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        filtered_data[name] = df.loc[mask].reset_index(drop=True)
    return filtered_data

def analyze_correlation_in_period(data, start_date, end_date):
    """
    Calcola e stampa la matrice di correlazione per un intervallo di tempo specifico.
    """
    # Filtra i dati per l'intervallo di date specifico
    filtered_data = filter_data_by_date(data, start_date, end_date)
    
    # Calcola i ritorni giornalieri per il periodo selezionato
    for name in filtered_data:
        filtered_data[name] = calculate_daily_returns(filtered_data[name])
    
    # Calcola la matrice di correlazione per l'intervallo selezionato
    correlation_df = calculate_correlation(filtered_data)
    print(f"Matrice di correlazione per il periodo {start_date} - {end_date}:")
    print(correlation_df)
    return correlation_df

