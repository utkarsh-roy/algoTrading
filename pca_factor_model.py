import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_factor_model(tickers, start_date='2020-01-01', end_date='2023-01-01'):
    """
    Applies Principal Component Analysis (PCA) to a universe of stocks to
    find the statistical factors that drive the variance in their returns.
    """
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Drop missing values
    data = data.dropna(axis=1) # Drop stocks with too much missing data
    data = data.dropna(axis=0) # Drop rows with missing data
    
    valid_tickers = data.columns.tolist()
    print(f"Using {len(valid_tickers)} valid tickers for PCA.")
    
    # Calculate daily log returns
    returns = np.log(data / data.shift(1)).dropna()
    
    # Standardize returns (Important for PCA so highly volatile stocks don't dominate)
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    
    # Apply PCA
    n_components = min(10, len(valid_tickers)) # Check top 10 factors
    pca = PCA(n_components=n_components)
    pca.fit(scaled_returns)
    
    # The explained variance ratio shows how much variance each component explains
    explained_variance = pca.explained_variance_ratio_
    
    print("-" * 40)
    print("PCA Explained Variance by Component:")
    for i, var in enumerate(explained_variance):
        print(f"Factor {i+1}: {var*100:.2f}%")
        
    print(f"Total variance explained by {n_components} factors: {np.sum(explained_variance)*100:.2f}%")
    print("-" * 40)
    
    # Visualizing Explained Variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, n_components + 1), explained_variance * 100, alpha=0.7)
    plt.title('Variance Explained per PCA Component')
    plt.xlabel('Component (Factor)')
    plt.ylabel('Variance Explained (%)')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, n_components + 1), np.cumsum(explained_variance) * 100, marker='o')
    plt.title('Cumulative Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance (%)')
    plt.axhline(y=80, color='r', linestyle='--', label='80% Variance Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('pca_variance_results.png')
    print("Plot saved to 'pca_variance_results.png'.")
    
    # Analyzing the first Principal Component (often viewed as the Market Factor)
    pc1_weights = pd.Series(pca.components_[0], index=valid_tickers)
    pc1_weights_sorted = pc1_weights.sort_values(ascending=False)
    
    print("\nTop 5 absolute weights in Component 1 (Market Factor):")
    print(pc1_weights_sorted.abs().nlargest(5))

if __name__ == "__main__":
    # We use a mix of sectors: Tech, Finance, Healthcare, Consumer
    universe_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 
                        'JPM', 'BAC', 'GS', 'MS', 
                        'JNJ', 'UNH', 'PFE',
                        'PG', 'KO', 'PEP',
                        'XOM', 'CVX']
    run_pca_factor_model(universe_tickers)
