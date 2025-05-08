# ml-american-options

# libraries
make sure you have numpy, scipy, scikit-learn, pandas

# how to run
python<version> main.py 

# The workflow of this code is as follows: 

1. data_optiondx.py and data_synthetic.py (which calls blackscholes.py)
  - synthetic case since we just used the BSM formula and a set of randomly generated parameters with reasonable bounds on the values (e.g. T was limited between 0 and 2 years). For the historical case, we were able to obtain S, K, v, T, and the option price from optionDX. The dividend yield, q, for SPY in Jan 2023 was prorated, taking the Q1 value and dividing it by three (averaged) 0.1314%. The fed rate for January 2023 was 4.33%.
  - Synthetic data was already generated in the data structure form,
    
Stock price (S) | Strike Price (K) | Dividend Yield (q) | Risk Free Rate (r) | Implied Vol (v) | Time to Expiry (T) | Option Price

  - but we had to perform some basic cleaning and reformatting to get the optionDX data into the python data frame with the following column structure, using the data collection description above. Features included everything but the option price, where that was the target.


2.  model.py
  - Using python scikit-learn, we implemented a Random Forest Model. A random forest model is an ensemble of many decision trees. Each decisions tree sees different subsets of the data via bagging and grows uniquely based on random feature sub-setting. Each tree grows to near full depth, becoming a strong (many levels deep),low-bias learner. This entire process controls overfitting at the forest level even if individual trees overfit bootstrap samples. This particular RF model was implemented by training it on 70% of the data, using 100 random trees, and regression on the option price target. The predictive performance was assessed using mean squared error.

3.  plotter.py
  - The data was visualized using a scatter plot of various call option prices against their corresponding expiration times

The first iteration of this project is in a poster form under the name FEC_ML_american_options_poster.pdf
