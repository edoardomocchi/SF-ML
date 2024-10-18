import pandas as pd

def calculate_daily_returns(df):
    """
    Calcola i ritorni giornalieri in modo esplicito.
    
    Per ogni giorno t, il ritorno giornaliero è calcolato come:
        ritorno_giornaliero(t) = (Prezzo_chiusura(t) - Prezzo_chiusura(t-1)) / Prezzo_chiusura(t-1)
    """
    daily_returns = [None]  # Il primo valore è None perché non c'è un giorno precedente
    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i - 1]
        curr_close = df['Close'].iloc[i]
        daily_return = (curr_close - prev_close) / prev_close
        daily_returns.append(daily_return)
    
    df = df.copy()
    df['Daily_Return'] = daily_returns
    return df.dropna(subset=['Daily_Return'])  # Rimuove i valori None iniziali

def calculate_volatility(df):
    """
    Calcola la volatilità (deviazione standard) dei ritorni giornalieri in modo esplicito.
    
    Formula della volatilità:
        σ = sqrt(Σ (ritorno_giornaliero(t) - μ)^2 / (N - 1))
    """
    daily_returns = df['Daily_Return'].tolist()
    mean_return = sum(daily_returns) / len(daily_returns)
    
    squared_diffs = [(r - mean_return) ** 2 for r in daily_returns]
    variance = sum(squared_diffs) / (len(squared_diffs) - 1)
    volatility = variance ** 0.5  # Radice quadrata della varianza
    return volatility

def calculate_correlation(dfs):
    # Extract 'Daily_Return' columns and align them
    returns_df = pd.concat(
        [df['Daily_Return'] for df in dfs.values()], axis=1, join='inner'
    )
    returns_df.columns = dfs.keys()

    # Truncate to the minimum length (if necessary)
    min_length = returns_df.shape[0]
    returns_df = returns_df.iloc[:min_length]

    # Compute the correlation matrix using pandas built-in method
    correlation_matrix = returns_df.corr()

    return correlation_matrix


def calculate_sharpe_ratio(daily_returns, risk_free_rate=0):
    """
    Calcola il Sharpe Ratio per una serie di ritorni giornalieri.
    
    Parameters:
    daily_returns (pd.Series): Ritorni giornalieri della criptovaluta.
    risk_free_rate (float): Tasso di interesse privo di rischio (default è 0).
    
    Returns:
    float: Sharpe Ratio.
    """
    excess_returns = daily_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio
