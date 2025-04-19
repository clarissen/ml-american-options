import pandas as pd
import yfinance as yf
import numpy as np

# OPTION DATA FROM OPTIONDX
# ==========================
# just studying quarter 1 data
# optiondx_pwd = "/Users/nicolas/desktop/quantitative_finance/FEC/ml-american-options/ml-american-options/optiondx/"
# SPY_pwd_q1_eod = optiondx_pwd + "spy_eod_2023q1-zfoivd/"

path_SPY_q1_2023 = "/Users/nicolas/desktop/quantitative_finance/FEC/ml-american-options/optiondx/spy_eod_2023q1-zfoivd/"

SPY_jan23_eod_df_raw = pd.read_csv(path_SPY_q1_2023 + "spy_eod_202301.txt", delimiter=", ")

timeser_underlying_unique = SPY_jan23_eod_df_raw.groupby("[QUOTE_READTIME]")["[UNDERLYING_LAST]"].last().reset_index()

call_dropables = ["[QUOTE_UNIXTIME]", "[QUOTE_READTIME]", "[QUOTE_TIME_HOURS]", "[EXPIRE_UNIX]", "[P_BID]", "[P_ASK]", "[P_SIZE]", "[P_LAST]", "[P_DELTA]", "[P_GAMMA]", "[P_VEGA]", "[P_THETA]", "[P_RHO]", "[P_IV]", "[P_VOLUME]"]
put_dropables = ["[QUOTE_UNIXTIME]", "[QUOTE_READTIME]", "[QUOTE_TIME_HOURS]", "[EXPIRE_UNIX]", "[C_DELTA]", "[C_GAMMA]", "[C_VEGA]", "[C_THETA]", "[C_RHO]", "[C_IV]", "[C_VOLUME]", "[C_LAST]", "[C_SIZE]", "[C_BID]", "[C_ASK]" ]

# starting with january 2023 data
df_calls = SPY_jan23_eod_df_raw.drop(columns=call_dropables)
df_puts = SPY_jan23_eod_df_raw.drop(columns=put_dropables)

# must reorganize data into the form of synthetic df for ease of use in already made algorithm
# the convention is: 
# Stock Price (S)	Strike Price (K)	Dividend Yield (q)	Risk-Free Rate (r)	Implied Volatility (v)	Time to Expiry (T)	Call Option Price
naming_convention_calls = ["Stock Price (S)",	"Strike Price (K)",	"Dividend Yield (q)",	"Risk-Free Rate (r)",	"Implied Volatility (v)",	"Time to Expiry (T)",	"Call Option Price"]
naming_convention_puts = ["Stock Price (S)",	"Strike Price (K)",	"Dividend Yield (q)",	"Risk-Free Rate (r)",	"Implied Volatility (v)",	"Time to Expiry (T)",	"Put Option Price"]

df_calls_cleaned = df_calls[["[UNDERLYING_LAST]", "[STRIKE]", "[C_IV]", "[DTE]",  "[C_LAST]"]]
df_puts_cleaned = df_puts[["[UNDERLYING_LAST]", "[STRIKE]", "[P_IV]", "[DTE]",  "[P_LAST]"]]

# ==========================


# RISK FREE RATE FROM FED INTEREST RATE
# ==========================
path_fed_rate = "/Users/nicolas/desktop/quantitative_finance/FEC/ml-american-options/fed/"
df_r = pd.read_csv(path_fed_rate + 'fed_interest_rate_q1_2023.csv')

risk_free_rate = float(df_r.at[5, "Federal funds effective rate"])/100
print(f"Risk free interest rate for Q1 2023: {risk_free_rate:.3%}")

# # ==========================

# SPY DIVIDEND YIELD JAN 2023
# ==========================
# Load SPY data
spy = yf.Ticker("SPY")

# Get dividend data in January 2023
dividends = spy.dividends['2023-01-01':'2023-01-31']
total_jan_div = dividends.sum()

# Get price data for January
price_data = spy.history(start='2023-01-01', end='2023-01-31')
avg_price_jan = price_data['Close'].mean()

# Known Q1 2023 dividend
q1_div = 1.506204

# Estimate monthly portion (1/3 of Q1)
jan_estimated_div = q1_div / 3

# Estimate monthly yield
jan_estimated_yield = jan_estimated_div / avg_price_jan

print(f"Estimated Dividend Yield for January 2023 (pro-rata): {jan_estimated_yield:.4%}")
# ==========================

# Creating the right data frame for ML model
# ==========================
df_calls_cleaned.insert(2,"div yield",jan_estimated_yield)
df_puts_cleaned.insert(2,"div yield",jan_estimated_yield)

# risk free rate insertion
df_calls_cleaned.insert(3,"rate",risk_free_rate)
df_puts_cleaned.insert(3,"rate",risk_free_rate)

# converting DTE to Time to expiry in (years)
df_calls_cleaned["[DTE]"] = df_calls_cleaned["[DTE]"]/365
df_puts_cleaned["[DTE]"] = df_puts_cleaned["[DTE]"]/365

# renaming according to synthetic convention
df_puts_cleaned.columns = naming_convention_puts

df_calls_cleaned.columns = naming_convention_calls

# ==========================

# functions to use data

def get_data(type):

    # setting dataframe
    if type == "calls":
        df = df_calls_cleaned
    else:
        df = df_puts_cleaned

    # defining features and target variable
    features = df.columns[:-1].tolist()
    target = df.columns[-1]

    return df, features, target
