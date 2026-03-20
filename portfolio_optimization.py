import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def download_data(tickers, start_date='2018-01-01', end_date='2023-01-01'):
    """Downloads adjusted close prices for the given tickers."""
    print(f"Downloading data for {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Calculates portfolio return, volatility, and Sharpe ratio."""
    returns = np.sum(mean_returns * weights) * 252 # Annualize
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252) # Annualize
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe_ratio

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Objective function to minimize (equivalent to maximizing Sharpe)."""
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def optimize_portfolio(tickers, data):
    """Finds the optimal weights for the maximum Sharpe ratio portfolio."""
    returns = data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(tickers)
    
    # Arguments for optimizer
    args = (mean_returns, cov_matrix)
    
    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: Weights between 0 and 1 (long only)
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    # Initial guess: Equal weighting
    initial_guess = num_assets * [1. / num_assets,]
    
    print("Running optimization...")
    result = minimize(negative_sharpe, initial_guess, args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
                      
    if not result.success:
        print("Optimization failed.")
        return None
        
    optimal_weights = result.x
    opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
    
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker:10}: {weight*100:.2f}%")
        
    print("-" * 30)
    print(f"Expected Annual Return:  {opt_return * 100:.2f}%")
    print(f"Annual Volatility:       {opt_volatility * 100:.2f}%")
    print(f"Sharpe Ratio:            {opt_sharpe:.4f}")
    print("-" * 30)
    
    # Visualize the results
    fig = plt.figure(figsize=(10, 6))
    plt.bar(tickers, optimal_weights * 100)
    plt.title('Optimal Portfolio Weights (Maximum Sharpe Ratio)')
    plt.ylabel('Weight (%)')
    plt.savefig('portfolio_optimization_results.png')
    print("Plot saved to 'portfolio_optimization_results.png'.")
    
if __name__ == "__main__":
    # Define a basket of tech/blue chip stocks
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'JPM']
    data = download_data(tickers)
    
    if not data.empty:
        optimize_portfolio(tickers, data)
    else:
        print("Data could not be downloaded.")
