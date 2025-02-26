import numpy as np
from scipy.stats import norm



def Black_Scholes_call(S0, K, T, r, sigma, q):
    """Calculate the Black-Scholes price for a European call option."""
    # S = underlying price
    # K = strike price of option
    # T = Time to expiration
    # r = risk-free interest rate
    # sigma = implied volatility

    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q - 0.5*sigma**2)*T)/(sigma*np.sqrt(T))

    call_price = S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    return call_price

def Black_Scholes_put(S0, K, T, r, sigma, q):
    """Calculate the Black-Scholes price for a European put option."""
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q - 0.5*sigma**2)*T)/(sigma*np.sqrt(T))

    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-q*T)*norm.cdf(-d1)

    return put_price