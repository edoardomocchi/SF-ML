import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib
import seaborn as sns

def plot_individual_price_trends(data, save_path=None):
    """
    Plots the closing price trends for each cryptocurrency individually,
    taking the dates directly from the dataset. Adds legends, axis labels, and improves aesthetics.
    Saves the plot if a save path is specified.
    """

    num_cryptos = len(data)
    cols = 2  # Number of columns in the grid
    rows = (num_cryptos + cols - 1) // cols  # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    colors = plt.cm.tab10.colors  # Use a colormap for different colors

    for idx, (name, df) in enumerate(data.items()):
        ax = axes[idx]

        # Ensure 'Date' column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])

        df = df.sort_values('Date')

        ax.plot(df['Date'], df['Close'], label=name, color=colors[idx % len(colors)])
        ax.set_title(f'{name} Closing Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()

        # Improve date formatting on x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis='x', rotation=45)

    # Remove any empty subplots if the number of cryptocurrencies is odd
    for ax in axes[num_cryptos:]:
        fig.delaxes(ax)

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
    
   



def plot_sharpe_ratios(sharpe_ratios, save_path=None):
    # Plotting Sharpe Ratios
    names = list(sharpe_ratios.keys())
    values = list(sharpe_ratios.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(names, values, color='skyblue')
    plt.xlabel('Sharpe Ratio')
    plt.title('Sharpe Ratios delle Criptovalute')
    plt.axvline(0, color='red', linestyle='--')  # Linea a 0 per riferimento

    if save_path:
        plt.savefig(save_path)
    
    plt.show()

    

  
# Call the plotting function after calculating Sharpe Ratios


