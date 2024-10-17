from modelli.format import load_all_data
from modelli.metrics import calculate_daily_returns, calculate_volatility
from modelli.graph import plot_individual_price_trends_row, plot_correlation_matrix
from modelli.analyze import filter_data_by_date
from modelli.montecarlo import monte_carlo_simulation

def main():
    # Carica i dati di tutte le criptovalute
    data = load_all_data()
    
    # Calcola i ritorni giornalieri per ciascuna criptovaluta e stampa la volatilità
    for name in data:
        data[name] = calculate_daily_returns(data[name])
        volatility = calculate_volatility(data[name])
        print(f"Volatilità per {name}: {volatility}")
    
    # 1. Matrice di correlazione e grafici su tutto il periodo
    # Calcola e mostra la matrice di correlazione generale
    correlation_df = calculate_correlation(data)
    print("Matrice di correlazione generale:")
    print(correlation_df)
    plot_correlation_matrix(correlation_df, title="Matrice di Correlazione Generale", save_path="correlation_matrix_general.png")
    
    # Plotta e salva le tendenze dei prezzi individuali su tutto il periodo
    plot_individual_price_trends_row(data, save_path="individual_price_trends_general.png")
    
    # 2. Matrice di correlazione e grafici sul periodo di crash
    # Definisci l'intervallo di tempo del periodo di crash
    start_date = '2021-11-01'
    end_date = '2022-06-30'

    # Filtra i dati per l'intervallo di crash
    filtered_data = filter_data_by_date(data, start_date, end_date)
    
    # Plotta le tendenze dei prezzi individuali durante il periodo di crash su una singola riga e salva
    plot_individual_price_trends_row(filtered_data, save_path="individual_price_trends_crash.png")

    # Calcola e mostra la matrice di correlazione per il periodo specifico
    period_correlation_df = calculate_correlation(filtered_data)
    print(f"Matrice di correlazione per il periodo {start_date} - {end_date}:")
    print(period_correlation_df)
    plot_correlation_matrix(period_correlation_df, title=f"Matrice di Correlazione {start_date} - {end_date}", save_path="correlation_matrix_period.png")
    
    # 3. Simulazione Monte Carlo solo per Bitcoin
    # Crea un dataset contenente solo i dati di Bitcoin
    btc_data = {'bitcoin': data['bitcoin']}
    monte_carlo_simulation(btc_data, num_simulations=100, num_days=252, save_path="monte_carlo_btc.png")

if __name__ == "__main__":
    main()
