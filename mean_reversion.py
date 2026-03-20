import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_bollinger_bands(ticker, window=20, num_std=2, start_date='2020-01-01', end_date='2023-01-01'):
    """
    Backtests a Mean Reversion strategy using Bollinger Bands.
    
    Buy Signal: When the price touches or goes below the lower band (oversold).
    Sell Signal: When the price touches or goes above the upper band (overbought).
    """
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("No data found.")
        return
        
    # Calculate Bollinger Bands
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper_Band'] = data['SMA'] + (data['STD'] * num_std)
    data['Lower_Band'] = data['SMA'] - (data['STD'] * num_std)
    
    # Generate Signals
    data['Signal'] = 0
    # Long when close < lower band
    data.loc[data['Close'] < data['Lower_Band'], 'Signal'] = 1
    # Short when close > upper band
    data.loc[data['Close'] > data['Upper_Band'], 'Signal'] = -1
    
    # This is a naive continuous position model; in reality, we'd hold until mean reversion (touching SMA)
    # Here, we carry the position forward until a new signal flips it
    data['Position'] = data['Signal'].replace(0, method='ffill').fillna(0)
    
    # Returns
    data['Market_Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Market_Returns'] * data['Position'].shift(1)
    
    data['Cumulative_Market'] = (1 + data['Market_Returns']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
    
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Prices, Bands, Signals
    plt.subplot(2, 1, 1)
    plt.plot(data['Close'], label='Close Price', alpha=0.6)
    plt.plot(data['SMA'], label='SMA', alpha=0.8, linestyle='--')
    plt.fill_between(data.index, data['Lower_Band'], data['Upper_Band'], color='grey', alpha=0.1)
    
    # Buy markers
    buy_signals = data[(data['Position'] == 1) & (data['Position'].shift(1) != 1)]
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', zorder=5)
    
    # Sell markers
    sell_signals = data[(data['Position'] == -1) & (data['Position'].shift(1) != -1)]
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', zorder=5)
    
    plt.title(f'{ticker} - Mean Reversion (Bollinger Bands)')
    plt.legend()
    
    # Plot 2: Returns
    plt.subplot(2, 1, 2)
    plt.plot(data['Cumulative_Market'], label='Market Returns')
    plt.plot(data['Cumulative_Strategy'], label='Strategy Returns')
    plt.title('Cumulative Returns')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mean_reversion_results.png')
    print("Plot saved to 'mean_reversion_results.png'.")
    
    market_ret = (data['Cumulative_Market'].iloc[-1] - 1) * 100
    strat_ret = (data['Cumulative_Strategy'].iloc[-1] - 1) * 100
    
    print("-" * 30)
    print(f"Total Market Return:   {market_ret:.2f}%")
    print(f"Total Strategy Return: {strat_ret:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # Test with a ticker known for ranging or to show the concept
    backtest_bollinger_bands('GLD', window=20, num_std=2)
