import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_option_pricing(S, K, T, r, sigma, option_type='call', num_simulations=10000, num_steps=252):
    """
    Prices an option using Monte Carlo Simulation of Geometric Brownian Motion (GBM).
    
    S: Spot Price
    K: Strike Price
    T: Time to maturity (years)
    r: Risk-free rate
    sigma: Volatility
    num_simulations: Number of price paths to generate
    num_steps: Time steps (e.g., 252 trading days per year)
    """
    print(f"Running Monte Carlo Simulation with {num_simulations} paths and {num_steps} steps...")
    
    dt = T / num_steps
    
    # Create an array to hold all simulated paths
    # Rows are steps, Columns are simulations
    price_paths = np.zeros((num_steps + 1, num_simulations))
    price_paths[0] = S
    
    # Generate random paths
    for t in range(1, num_steps + 1):
        # Generate random standard normal numbers
        z = np.random.standard_normal(num_simulations)
        
        # Calculate next step price using GBM formula
        price_paths[t] = price_paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
        
    # Get the terminal prices (at maturity T)
    terminal_prices = price_paths[-1]
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(terminal_prices - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - terminal_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    # Discount payoffs back to present value
    present_values = payoffs * np.exp(-r * T)
    
    # The option price is the average of all discounted payoffs
    option_price = np.mean(present_values)
    
    # Calculate standard error (confidence intervals)
    standard_error = np.std(present_values) / np.sqrt(num_simulations)
    
    print("-" * 40)
    print(f"Monte Carlo {option_type.capitalize()} Option Price: ${option_price:.4f}")
    print(f"Standard Error: ${standard_error:.4f}")
    print("-" * 40)
    
    # Visualizing a subset of the paths
    plt.figure(figsize=(12, 6))
    subset_size = min(100, num_simulations) # Plot only 100 paths for clarity
    plt.plot(price_paths[:, :subset_size], linewidth=0.5)
    plt.title(f'Monte Carlo Simulation - {subset_size} Price Paths (GBM)')
    plt.xlabel('Steps (Days)')
    plt.ylabel('Asset Price')
    plt.axhline(K, color='r', linestyle='--', label=f'Strike Price ({K})')
    plt.legend()
    plt.savefig('monte_carlo_paths.png')
    print("Plot saved to 'monte_carlo_paths.png'.")
    
    return option_price

if __name__ == "__main__":
    # Same inputs as Black-Scholes example for comparison
    S = 100.0   # Spot
    K = 100.0   # Strike
    T = 1.0     # 1 year
    r = 0.05    # 5% rate
    sigma = 0.20 # 20% vol
    
    monte_carlo_option_pricing(S, K, T, r, sigma, option_type='call')
