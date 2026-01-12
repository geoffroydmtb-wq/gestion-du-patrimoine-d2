import pandas as pd
import numpy as np

def calculate_key_metrics(df_prices, risk_free_rate=0.03):
    """
    Calcule : Rendement, Volatilité, Sharpe, et Max Drawdown.
    """
    # 1. Calcul des rendements quotidiens
    returns = df_prices.pct_change().dropna()
    
    # 2. Performance Totale
    total_return = (df_prices.iloc[-1] / df_prices.iloc[0]) - 1
    
    # 3. Volatilité Annualisée
    volatility = returns.std() * np.sqrt(252)
    
    # 4. Ratio de Sharpe Annualisé
    mean_annual_return = returns.mean() * 252
    sharpe_ratio = (mean_annual_return - risk_free_rate) / volatility
    
    # 5. Max Drawdown (NOUVEAU)
    # On reconstruit l'évolution d'un investissement de 1€
    cumulative_returns = (1 + returns).cumprod()
    # On garde en mémoire le plus haut historique atteint à chaque instant (Running Max)
    running_max = cumulative_returns.cummax()
    # On calcule l'écart actuel par rapport à ce sommet
    drawdown = (cumulative_returns - running_max) / running_max
    # Le Max Drawdown est le minimum (le plus bas) de ces écarts
    max_drawdown = drawdown.min()
    
    # On rassemble tout
    metrics_df = pd.DataFrame({
        'Perf. Totale': total_return,
        'Volatilité (an)': volatility,
        'Ratio de Sharpe': sharpe_ratio,
        'Max Drawdown': max_drawdown
    })
    
    return metrics_df

def get_correlation_matrix(df_prices):
    """
    Calcule la matrice de corrélation.
    """
    returns = df_prices.pct_change().dropna()
    corr_matrix = returns.corr()
    return corr_matrix