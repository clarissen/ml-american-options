import numpy as np
import pandas as pd
from scipy.stats import norm

import blackscholes as bs

def get_data(type):
    synth_calls = BS_call_synthetic_data(100,int(1e5))

    synth_puts = BS_put_synthetic_data(100,int(1e5))

    # setting dataframe
    if type == "calls":
        df = synth_calls
    else:
        df = synth_puts

    # defining features and target variable
    features = df.columns[:-1].tolist()
    target = df.columns[-1]

    return df, features, target

def BS_call_synthetic_data(S0, N):
    """ generate a synthetic data set by specifcying initial stock price and synthetic data length N"""

    # all of this data is initialized and synthetic
    S = np.full(N, S0) #Stock Price S
    K = np.random.uniform(90, 110, N) #Strike price range
    q = np.random.uniform(0.02, 0.06, N) #dividend yield range (annual)
    r = np.random.uniform(0.01, 0.06, N) #risk free rate range (annual)
    sigma = np.random.uniform(0.15, 0.3, N) #implied volatility range
    T = np.random.uniform(0.25, 2, N) #time to maturity range[years]

    # we compute the call price from the synthetic data
    call_price = bs.Black_Scholes_call(S, K, T, r, sigma, q)

    df = pd.DataFrame({
        'Stock Price (S)': S,
        'Strike Price (K)': K,
        'Dividend Yield (q)': q,
        'Risk-Free Rate (r)': r,
        'Implied Volatility (v)': sigma,
        'Time to Expiry (T)': T,
        'Call Option Price': call_price
    })

    return df

def BS_put_synthetic_data(S0, N):
    """ generate a synthetic data set by specifcying initial stock price and synthetic number of data points N"""

    # all of this data is initialized and synthetic
    S = np.full(N, S0) #Stock Price S
    K = np.random.uniform(90, 110, N) #Strike price range
    q = np.random.uniform(0.02, 0.06, N) #dividend yield range (annual)
    r = np.random.uniform(0.01, 0.06, N) #risk free rate range (annual)
    sigma = np.random.uniform(0.15, 0.3, N) #implied volatility range
    T = np.random.uniform(0.25, 2, N) #time to maturity range[years]

    # we compute the put price from the synthetic data
    put_price = bs.Black_Scholes_put(S, K, T, r, sigma, q)

    df = pd.DataFrame({
        'Stock Price (S)': S,
        'Strike Price (K)': K,
        'Dividend Yield (q)': q,
        'Risk-Free Rate (r)': r,
        'Implied Volatility (v)': sigma,
        'Time to Expiry (T)': T,
        'Put Option Price': put_price
    })

    return df
