import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)
def get_stock_data(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()

    # On télécharge les données depuis Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # On cherche la colonne 'Adj Close' (Prix ajusté des dividendes)
    if 'Adj Close' in data:
        df = data['Adj Close']
    else:
        # Si la structure est complexe (multi-index), on cherche dedans
        if isinstance(data.columns, pd.MultiIndex):
            try:
                df = data.xs('Adj Close', axis=1, level=0)
            except KeyError:
                df = data['Close']
        else:
             df = data['Close']
    
    # On supprime les jours fériés (lignes vides)
    df = df.dropna()
    return df