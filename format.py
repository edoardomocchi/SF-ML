import pandas as pd
import os

def load_all_data():
    # Specifica i nomi dei file CSV in modo esplicito
    files = {
        'bitcoin': './data.csv/databtc.csv',
        'ethereum': './data.csv/dataeth.csv',
        'litecoin': './data.csv/dataltc.csv',
        'ripple': './data.csv/dataxrp.csv',
        'cardano': './data.csv/dataada.csv'
    }
    
    data = {}
    for name, filepath in files.items():
        data[name] = pd.read_csv(filepath)
    
    return data


