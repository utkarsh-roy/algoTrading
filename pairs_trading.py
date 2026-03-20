import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_pairs_trading(ticker1, ticker2, start_date='2018-01-01', end_date='2022-01-01', entry_z=2.0, exit_z=0.0):
    """
    Backtests a basic Statistical Arbitrage (Pairs Trading) strategy.
    
    We look for the spread between two historically correlated assets.
    If the spread deviates significantly from the mean (Z-Score > entry_z), we short the outperforming
    asset and buy the underperforming asset, expecting mean reversion.
    """
    print(f"Downloading data for {ticker1} and {ticker2}...")
    tickers = [ticker1, ticker2]
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    if data.empty or data.shape[1] < 2:
        print("Data error.")
        return
        
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Create cumulative price series normalized to 1
    cum_returns = (1 + returns).cumprod()
    
    # Calculate the spread (difference in cumulative returns)
    spread = cum_returns[ticker1] - cum_returns[ticker2]
    
    # Calculate Z-score of the spread
    # We use a 60-day rolling window
    rolling_mean = spread.rolling(window=60).mean()
    rolling_std = spread.rolling(window=60).std()
    z_score = (spread - rolling_mean) / rolling_std
    
    # Trading logic based on Z-Score
    positions = pd.DataFrame(index=z_score.index, columns=[ticker1, ticker2]).fillna(0)
    
    # Logic:
    # If Z > entry_z: spread is too high (Ticker 1 > Ticker 2). Short T1, Long T2
    # If Z < -entry_z: spread is too low (Ticker 1 < Ticker 2). Long T1, Short T2
    # Exit when Z crosses exit_z
    
    current_position = 0 # 1 means Long T1/Short T2, -1 means Short T1/Long T2, 0 means Neutral
    
    t1_pos = []
    t2_pos = []
    
    for z in z_score:
        if pd.isna(z):
            t1_pos.append(0)
            t2_pos.append(0)
            continue
            
        if z > entry_z:
            current_position = -1 # Short T1, Long T2
        elif z < -entry_z:
            current_position = 1  # Long T1, Short T2
        elif (current_position == -1 and z < exit_z) or (current_position == 1 and z > -exit_z):
            current_position = 0  # Exit position
            
        t1_pos.append(current_position)
        t2_pos.append(-current_position)
        
    positions[ticker1] = t1_pos
    positions[ticker2] = t2_pos
    
    # Calculate Strategy Returns
    # Position shifted by 1 to avoid lookahead bias
    strategy_returns = (positions.shift(1) * returns).sum(axis=1)
    cum_strategy = (1 + strategy_returns).cumprod()
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(cum_returns[ticker1], label=ticker1)
    plt.plot(cum_returns[ticker2], label=ticker2)
    plt.title(f'Normalized Prices of {ticker1} and {ticker2}')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(z_score, label='Z-Score', color='purple')
    plt.axhline(entry_z, color='r', linestyle='--')
    plt.axhline(-entry_z, color='g', linestyle='--')
    plt.axhline(exit_z, color='k', linestyle='-')
    plt.title('Spread Z-Score')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(cum_strategy, label='Strategy Returns', color='orange')
    plt.title('Pairs Trading Strategy Cumulative Returns')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pairs_trading_results.png')
    print("Plot saved to 'pairs_trading_results.png'.")
    
    print("-" * 30)
    print(f"Total Strategy Return: {(cum_strategy.iloc[-1] - 1)*100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # KO (Coca-Cola) and PEP (PepsiCo) are classic examples of highly correlated pairs
    backtest_pairs_trading('KO', 'PEP')
