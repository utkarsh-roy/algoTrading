# AlgoTrading & Quantitative Finance Implementations

Welcome to your Quantitative Finance repository! This directory contains a foundational suite of Python scripts that implement fundamental algorithms, trading strategies, risk management tools, and derivative pricing models.

## Overview of Algorithms

### 1. Momentum Trading
- **`moving_average_crossover.py`**: Implements a classic Simple Moving Average (SMA) crossover strategy. It generates buy signals when a short-term SMA crosses above a long-term SMA (golden cross), and sell signals for the inverse (death cross).

### 2. Statistical Arbitrage & Mean Reversion
- **`mean_reversion.py`**: Uses Bollinger Bands (Standard Deviations around a Moving Average) to identify overbought and oversold market conditions.
- **`pairs_trading.py`**: Exploits historical correlations between two assets (e.g., KO and PEP). It shorts the outperformer and goes long on the underperformer when the spread's Z-Score diverges significantly from the mean.

### 3. Derivatives Pricing
- **`black_scholes.py`**: The classic Black-Scholes-Merton mathematical model for pricing European Call and Put Options, and calculating Greeks (Delta, Gamma, Vega, Theta, Rho).
- **`monte_carlo_pricing.py`**: Uses Geometric Brownian Motion (GBM) to simulate thousands of potential future random paths for an asset's price to estimate derivative payoffs.

### 4. Portfolio & Risk Management
- **`portfolio_optimization.py`**: Implements Harry Markowitz's Modern Portfolio Theory (MPT). Uses SciPy numerical optimization to find the portfolio weights that maximize the Sharpe Ratio (Efficient Frontier).
- **`value_at_risk.py`**: Calculates Historical and Parametric Value at Risk (VaR) and Conditional VaR (Expected Shortfall) to measure the risk of loss in a portfolio.
- **`pca_factor_model.py`**: Applies Principal Component Analysis (PCA) to discover the underlying statistical factors driving stock universe returns, helping build market-neutral factor portfolios.

## How to Run

1. **Install Dependencies:**
   Ensure you have the required Python libraries installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute Scripts:**
   You can run any of the scripts directly from the terminal. 
   ```bash
   python moving_average_crossover.py
   python portfolio_optimization.py
   ```
   
   *Note: Most strategies utilizing live market data will automatically download datasets using `yfinance`.* The scripts will generate terminal output and save summary plots (`.png` charts) into the same folder for your review.

## Modifications
These scripts are currently hardcoded with educational examples (e.g., pricing SPY, pairs trading KO and PEP). Open any `.py` file and scroll to the `if __name__ == "__main__":` block at the bottom to swap in your own ticker symbols, dates, and parameters!
