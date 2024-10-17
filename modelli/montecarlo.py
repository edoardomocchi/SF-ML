import numpy as np
import matplotlib.pyplot as plt  # Ensure this line is included

def monte_carlo_simulation(data, num_simulations=100, num_days=252, save_path=None):
    """
    Esegue una simulazione Monte Carlo sull'andamento dei prezzi per ciascuna criptovaluta
    basata su media e volatilità storica. Plotta e salva i risultati se specificato.
    """
    simulation_results = {}
    
    # Create subplots; if only one cryptocurrency, 'axes' will be a single object
    fig, axes = plt.subplots(len(data), 1, figsize=(10, len(data) * 5))
    
    # Ensure 'axes' is always iterable
    if len(data) == 1:
        axes = [axes]  # Make it a list for consistency
    
    for ax, (name, df) in zip(axes, data.items()):
        # Calcola i parametri storici
        daily_returns = df['Daily_Return'].dropna()
        mean_return = daily_returns.mean()
        volatility = daily_returns.std()
        
        # Esegue la simulazione Monte Carlo
        prices = np.zeros((num_simulations, num_days))
        prices[:, 0] = df['Close'].iloc[-1]  # Prezzo iniziale: ultimo prezzo disponibile
        
        for sim in range(num_simulations):
            for day in range(1, num_days):
                shock = np.random.normal(loc=mean_return, scale=volatility)
                prices[sim, day] = prices[sim, day - 1] * (1 + shock)
        
        # Aggiunge i risultati alla simulazione
        simulation_results[name] = prices
        
        # Grafico della simulazione Monte Carlo per la criptovaluta
        ax.plot(prices.T, alpha=0.1)
        ax.set_title(f'Simulazione Monte Carlo - {name}')
        ax.set_xlabel('Giorni')
        ax.set_ylabel('Prezzo')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()
    
    return simulation_results
