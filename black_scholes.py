import numpy as np
from scipy.stats import norm
import math

class BlackScholesModel:
    """
    Implementation of the Black-Scholes-Merton model for pricing European call and put options.
    It also calculates the associated 'Greeks' for risk management.
    """
    def __init__(self, S, K, T, r, sigma):
        """
        S: Current stock price (Spot Price)
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
        # Pre-compute d1 and d2
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        
    def call_price(self):
        """Calculate the Call option price."""
        return (self.S * norm.cdf(self.d1, 0.0, 1.0) - 
                self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2, 0.0, 1.0))
    
    def put_price(self):
        """Calculate the Put option price."""
        return (self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2, 0.0, 1.0) - 
                self.S * norm.cdf(-self.d1, 0.0, 1.0))
                
    def get_greeks(self, option_type='call'):
        """Calculate the Greeks (Delta, Gamma, Vega, Theta, Rho)."""
        greeks = {}
        
        # Delta
        if option_type == 'call':
            greeks['Delta'] = norm.cdf(self.d1)
        else:
            greeks['Delta'] = norm.cdf(self.d1) - 1
            
        # Gamma (Same for Call and Put)
        greeks['Gamma'] = norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        
        # Vega (Same for Call and Put, returned as value per 1% change)
        greeks['Vega'] = (self.S * norm.pdf(self.d1) * np.sqrt(self.T)) / 100
        
        # Theta (returned as value per 1 day change)
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == 'call':
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            greeks['Theta'] = (term1 - term2) / 365
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            greeks['Theta'] = (term1 + term2) / 365
            
        # Rho (returned as value per 1% change)
        if option_type == 'call':
            greeks['Rho'] = (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)) / 100
        else:
            greeks['Rho'] = (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)) / 100
            
        return greeks

if __name__ == "__main__":
    # Example pricing
    S = 100.0   # Spot price
    K = 100.0   # Strike price
    T = 1.0     # 1 Year to maturity
    r = 0.05    # 5% Risk-free rate
    sigma = 0.20 # 20% Volatility
    
    bsm = BlackScholesModel(S, K, T, r, sigma)
    
    print(f"Inputs: Spot={S}, Strike={K}, Time={T}y, Rate={r}, Volatility={sigma}\n")
    
    print(f"Call Option Price: ${bsm.call_price():.4f}")
    print("Call Greeks:", bsm.get_greeks('call'))
    
    print(f"\nPut Option Price: ${bsm.put_price():.4f}")
    print("Put Greeks:", bsm.get_greeks('put'))
