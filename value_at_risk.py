import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def calculate_var_cvar(ticker, confidence_level=0.95, start_date='2015-01-01', end_date='2023-01-01'):
    """
    Calculates Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
    We use both the Historical method and the Parametric (Variance-Covariance) method.
    """
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("No data found.")
        return
        
    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()
    
    # -----------------------------
    # 1. Historical VaR & CVaR
    # -----------------------------
    # The percentile corresponds to 1 - confidence_level
    alpha = 1 - confidence_level
    hist_var = np.percentile(returns, alpha * 100)
    
    # CVaR is the average of returns worse than the VaR
    hist_cvar = returns[returns <= hist_var].mean()
    
    # -----------------------------
    # 2. Parametric (Normal) VaR & CVaR
    # -----------------------------
    mean = np.mean(returns)
    std_dev = np.std(returns)
    
    # Z-score for the given confidence level (e.g., -1.645 for 95%)
    z_score = norm.ppf(alpha)
    
    param_var = mean + (z_score * std_dev)
    
    # Expected shortfall for a normal distribution
    param_cvar = mean - (std_dev * norm.pdf(z_score) / alpha)
    
    print("\nRisk Profiling Results (Daily Returns)")
    print("-" * 40)
    print(f"Historical {confidence_level*100}% VaR:  {hist_var*100:.2f}%")
    print(f"Historical {confidence_level*100}% CVaR: {hist_cvar*100:.2f}%")
    print(f"Parametric {confidence_level*100}% VaR:  {param_var*100:.2f}%")
    print(f"Parametric {confidence_level*100}% CVaR: {param_cvar*100:.2f}%")
    print("-" * 40)
    
    # Plotting the Distribution of Returns
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=100, alpha=0.7, density=True, color='blue', label='Daily Returns Density')
    
    # Plot normal distribution overlay
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
    
    # Add vertical lines for Historical VaR and CVaR
    plt.axvline(hist_var, color='red', linestyle='dashed', linewidth=2, label=f'Historical VaR ({hist_var*100:.2f}%)')
    plt.axvline(hist_cvar, color='orange', linestyle='dotted', linewidth=2, label=f'Historical CVaR')

    plt.title(f'{ticker} Returns Distribution with VaR (Confidence Level: {confidence_level*100}%)')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.savefig('value_at_risk_results.png')
    print("Plot saved to 'value_at_risk_results.png'.")
    
if __name__ == "__main__":
    calculate_var_cvar('SPY', confidence_level=0.99)
