# Import necessary libraries
from modelli.format import load_all_data
from modelli.metrics import calculate_daily_returns, calculate_volatility, calculate_correlation, calculate_sharpe_ratio
from modelli.graph import plot_individual_price_trends_row, plot_correlation_matrix, plot_sharpe_ratios
from modelli.analyze import filter_data_by_date
from modelli.montecarlo import monte_carlo_simulation
import matplotlib.pyplot as plt

def main():
    # Load cryptocurrency data
    data = load_all_data()
    
    # Dictionary to store Sharpe Ratios for the overall period
    sharpe_ratios = {}
    
    # Calculate daily returns, volatility, and Sharpe Ratio for the entire dataset
    for name in data:
        # Calculate daily returns
        data[name] = calculate_daily_returns(data[name])
        
        # Calculate volatility
        volatility = calculate_volatility(data[name])
        print(f"Volatilità per {name}: {volatility}")
        
        # Calculate Sharpe Ratio
        sharpe_ratio = calculate_sharpe_ratio(data[name]['Daily_Return'])
        sharpe_ratios[name] = sharpe_ratio
        print(f"Sharpe Ratio per {name}: {sharpe_ratio}")

    # 1. Correlation matrix and graphs for the entire period
    correlation_df = calculate_correlation(data)
    print("Matrice di correlazione generale:")
    print(correlation_df)
    plot_correlation_matrix(correlation_df, title="Matrice di Correlazione Generale", save_path="correlation_matrix_general.png")
    
    # Plot individual price trends
    plot_individual_price_trends_row(data, save_path="individual_price_trends_general.png")
    
    # 2. Correlation matrix and graphs for the crash period
    start_date = '2021-11-01'
    end_date = '2022-06-30'
    filtered_data = filter_data_by_date(data, start_date, end_date)
    plot_individual_price_trends_row(filtered_data, save_path="individual_price_trends_crash.png")
    
    period_correlation_df = calculate_correlation(filtered_data)
    print(f"Matrice di correlazione per il periodo {start_date} - {end_date}:")
    print(period_correlation_df)
    plot_correlation_matrix(period_correlation_df, title=f"Matrice di Correlazione {start_date} - {end_date}", save_path="correlation_matrix_period.png")

    # Calculate Sharpe Ratios for the crash period
    sharpe_ratios_crash = {}
    for name in filtered_data:
        # Calculate daily returns for the crash period
        filtered_data[name] = calculate_daily_returns(filtered_data[name])
        
        # Calculate Sharpe Ratio for the crash period
        sharpe_ratio_crash = calculate_sharpe_ratio(filtered_data[name]['Daily_Return'])
        sharpe_ratios_crash[name] = sharpe_ratio_crash
        print(f"Sharpe Ratio for {name} during the crash period: {sharpe_ratio_crash}")

    # 3. Monte Carlo simulation for Bitcoin
    btc_data = {'bitcoin': data['bitcoin']}
    monte_carlo_simulation(btc_data, num_simulations=100, num_days=252, save_path="monte_carlo_btc.png")

    # Plot Sharpe Ratios and save the graph
    plot_sharpe_ratios(sharpe_ratios, save_path="sharpe_ratios.png")  # Save the overall Sharpe Ratio graph
    plot_sharpe_ratios(sharpe_ratios_crash, save_path="sharpe_ratios_crash.png")  # Save the crash period Sharpe Ratio graph

if __name__ == "__main__":
    main()
