import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_moving_average_crossover(ticker, short_window=50, long_window=200, start_date='2020-01-01', end_date='2023-01-01'):
    """
    Backtests a basic Simple Moving Average (SMA) crossover strategy.
    
    A buy signal is generated when the short-term MA crosses above the long-term MA.
    A sell signal is generated when the short-term MA crosses below the long-term MA.
    """
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("No data found. Check ticker or dates.")
        return
        
    print("Calculating Moving Averages...")
    # Calculate moving averages
    data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate signals
    data['Signal'] = 0
    # 1 when short MA is above long MA
    data['Signal'][short_window:] = np.where(data[f'SMA_{short_window}'][short_window:] > data[f'SMA_{long_window}'][short_window:], 1, 0)
    
    # Calculate trading positions (1 for buy, -1 for sell) based on signal changes
    data['Position'] = data['Signal'].diff()
    
    # Calculate Returns
    data['Market_Returns'] = data['Close'].pct_change()
    # Strategy returns (shift position by 1 to avoid lookahead bias)
    data['Strategy_Returns'] = data['Market_Returns'] * data['Signal'].shift(1)
    
    # Cumulative returns
    data['Cumulative_Market_Returns'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot Price and MAs
    plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='Close Price', alpha=0.5)
    plt.plot(data[f'SMA_{short_window}'], label=f'{short_window}-Day SMA')
    plt.plot(data[f'SMA_{long_window}'], label=f'{long_window}-Day SMA')
    
    # Plot Buy Signals
    plt.plot(data[data['Position'] == 1].index, 
             data[f'SMA_{short_window}'][data['Position'] == 1], 
             '^', markersize=10, color='g', lw=0, label='Buy Signal')
    # Plot Sell Signals
    plt.plot(data[data['Position'] == -1].index, 
             data[f'SMA_{short_window}'][data['Position'] == -1], 
             'v', markersize=10, color='r', lw=0, label='Sell Signal')
             
    plt.title(f'{ticker} - Moving Average Crossover Strategy')
    plt.ylabel('Price')
    plt.legend()
    
    # Plot Cumulative Returns
    plt.subplot(2, 1, 2)
    plt.plot(data['Cumulative_Market_Returns'], label='Market Returns')
    plt.plot(data['Cumulative_Strategy_Returns'], label='Strategy Returns')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    
    plt.tight_layout()
    # Save the plot securely without displaying the GUI
    plt.savefig('moving_average_crossover_results.png')
    print("Plot saved as 'moving_average_crossover_results.png'.")
    
    # Print statistics
    total_market_return = (data['Cumulative_Market_Returns'].iloc[-1] - 1) * 100
    total_strategy_return = (data['Cumulative_Strategy_Returns'].iloc[-1] - 1) * 100
    
    print("-" * 30)
    print(f"Total Market Return:   {total_market_return:.2f}%")
    print(f"Total Strategy Return: {total_strategy_return:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    print("Running Moving Average Crossover Strategy on SPY...")
    backtest_moving_average_crossover('SPY', short_window=50, long_window=200)
