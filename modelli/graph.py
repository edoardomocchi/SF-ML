import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_individual_price_trends_row(data, save_path=None):
    """
    Plotta le tendenze dei prezzi di chiusura per ciascuna criptovaluta individualmente,
    posizionando tutti i grafici in una singola riga senza etichette delle date sull'asse x.
    Salva il grafico se viene specificato un percorso di salvataggio.
    """
    num_cryptos = len(data)
    fig, axes = plt.subplots(1, num_cryptos, figsize=(20, 5), sharey=False)
    
    for ax, (name, df) in zip(axes, data.items()):
        df = df.sort_values('Date')
        min_val = df['Close'].min() * 0.95
        max_val = df['Close'].max() * 1.05
        ax.set_ylim([min_val, max_val])
        
        ax.plot(df['Date'], df['Close'], label=name, color='blue')
        ax.set_title(f'{name}')
        
        ax.set_xticklabels([])
        ax.set_xticks([])
        
        ax.set_yticklabels([])
        ax.set_yticks([])
    
    fig.supylabel('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()


def plot_price_trends(data, save_path=None):
    """
    Plotta la tendenza dei prezzi di chiusura per tutte le criptovalute su un singolo grafico.
    Salva il grafico se viene specificato un percorso di salvataggio.
    """
    plt.figure(figsize=(10, 6))
    
    for name, df in data.items():
        df = df.sort_values('Date')
        plt.plot(df['Date'], df['Close'], label=name)
    
    plt.title('Tendenze dei Prezzi di Chiusura')
    plt.xlabel('Data')
    plt.ylabel('Prezzo di Chiusura')
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()


def plot_correlation_matrix(correlation_df, title="Matrice di Correlazione", save_path=None):
    """
    Plotta la matrice di correlazione come una heatmap.
    Salva il grafico se viene specificato un percorso di salvataggio.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    
    plt.show()



def plot_sharpe_ratios(sharpe_ratios, save_path=None):
    # Plotting Sharpe Ratios
    names = list(sharpe_ratios.keys())
    values = list(sharpe_ratios.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(names, values, color='skyblue')
    plt.xlabel('Sharpe Ratio')
    plt.title('Sharpe Ratios of Cryptocurrencies')
    plt.axvline(0, color='red', linestyle='--')  # Line at 0 for reference
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")  # Save the figure
    plt.show()

# Call the plotting function after calculating Sharpe Ratios


