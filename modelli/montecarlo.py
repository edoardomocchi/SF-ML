import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def monte_carlo_simulation_prediction(data, num_simulations=100):
    """
    Performs a Monte Carlo simulation on the price trends for each cryptocurrency
    based on historical mean and volatility. Returns the simulation results.
    """
    simulation_results = {}

    # Define the simulation period
    simulation_start_date = pd.to_datetime('2024-01-01')
    simulation_end_date = pd.to_datetime('2024-09-30')
    num_days = (simulation_end_date - simulation_start_date).days

    for name, df in data.items():
        # Ensure 'Date' is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Calculate historical parameters
        historical_data = df[df['Date'] < simulation_start_date]
        daily_returns = historical_data['Daily_Return'].dropna()
        mean_return = daily_returns.mean()
        volatility = daily_returns.std()

        # Generate future dates from simulation_start_date to simulation_end_date
        future_dates = pd.date_range(start=simulation_start_date, end=simulation_end_date, freq='B')
        num_simulation_days = len(future_dates)

        # Initialize price array
        prices = np.zeros((num_simulations, num_simulation_days))
        # Initial price: last available price before simulation_start_date
        if not historical_data.empty:
            initial_price = historical_data['Close'].iloc[-1]
        else:
            raise ValueError(f"No historical data available before {simulation_start_date}")
        prices[:, 0] = initial_price

        # Perform Monte Carlo simulation
        shocks = np.random.normal(loc=mean_return, scale=volatility, size=(num_simulations, num_simulation_days))
        prices[:, 1:] = prices[:, [0]] * np.cumprod(1 + shocks[:, 1:], axis=1)

        # Add simulation results
        simulation_results[name] = {
            'prices': prices,
            'future_dates': future_dates,
            'historical_data': historical_data
        }

    return simulation_results




def monte_carlo_simulation_plot(simulation_results, data, save_path=None):
    """
    Plots the results of the Monte Carlo simulation and includes the historical prices.
    Saves the plot if a save path is specified.
    """
    fig, axes = plt.subplots(len(data), 1, figsize=(14, len(data) * 7))

    # Ensure 'axes' is always iterable
    if len(data) == 1:
        axes = [axes]  # Make it a list for consistency

    for ax, (name, result) in zip(axes, simulation_results.items()):
        prices = result['prices']
        future_dates = result['future_dates']
        historical_data = result['historical_data']

        # Calculate percentiles for confidence interval
        percentile_5 = np.percentile(prices, 5, axis=0)
        percentile_95 = np.percentile(prices, 95, axis=0)

        # Plot historical prices up to simulation start date
        ax.plot(historical_data['Date'], historical_data['Close'], color='blue', label='Historical Price')

        # Plot actual prices over the simulation period for comparison (if available)
        actual_data = data[name][(data[name]['Date'] >= future_dates[0]) & (data[name]['Date'] <= future_dates[-1])]
        if not actual_data.empty:
            ax.plot(actual_data['Date'], actual_data['Close'], color='green', label='Actual Price')

        # Plot simulations
        for sim in range(prices.shape[0]):
            ax.plot(future_dates, prices[sim], color='grey', alpha=0.1)

        # Shade the 90% confidence interval
        ax.fill_between(future_dates, percentile_5, percentile_95, color='orange', alpha=0.3, label='90% Confidence Interval')

        # Plot median of simulations
        median_simulation = np.median(prices, axis=0)
        ax.plot(future_dates, median_simulation, color='red', linestyle='--', label='Median Simulation')

        # Formatting
        ax.set_title(f'Monte Carlo Simulation - {name}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # Create a combined legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        # Reduce the number of x-axis date labels
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        # Rotate date labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Set x-limits to focus on the simulation period
        start_xlim = historical_data['Date'].iloc[-30] if len(historical_data) >= 30 else historical_data['Date'].iloc[0]
        ax.set_xlim([start_xlim, future_dates[-1] + pd.Timedelta(days=10)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
