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
    """
    Calcola la matrice di correlazione tra le criptovalute in modo esplicito.
    
    Formula della correlazione tra due serie temporali X e Y:
        corr(X, Y) = cov(X, Y) / (σ_X * σ_Y)
    
    dove:
        - cov(X, Y) è la covarianza tra X e Y
        - σ_X e σ_Y sono le deviazioni standard di X e Y
    """
    # Calcola i ritorni giornalieri per tutte le criptovalute e trova la lunghezza minima
    returns_data = {name: df['Daily_Return'].tolist() for name, df in dfs.items()}
    min_length = min(len(returns) for returns in returns_data.values())
    returns_data = {name: returns[:min_length] for name, returns in returns_data.items()}
    
    # Calcola le medie e le deviazioni standard per ogni criptovaluta
    means = {name: sum(returns) / len(returns) for name, returns in returns_data.items()}
    std_devs = {name: (sum((r - means[name]) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
                for name, returns in returns_data.items()}
    
    # Calcola la matrice di correlazione
    correlation_matrix = {}
    crypto_names = list(returns_data.keys())
    
    for i, name1 in enumerate(crypto_names):
        correlation_matrix[name1] = {}
        for j, name2 in enumerate(crypto_names):
            if i == j:
                correlation_matrix[name1][name2] = 1.0
            else:
                # Calcola la covarianza
                covariance = sum((returns_data[name1][k] - means[name1]) * (returns_data[name2][k] - means[name2])
                                 for k in range(min_length)) / (min_length - 1)
                
                # Calcola la correlazione
                correlation = covariance / (std_devs[name1] * std_devs[name2])
                correlation_matrix[name1][name2] = correlation
    
    return pd.DataFrame(correlation_matrix)
