import pandas as pd

optiondx_pwd = "/Users/nicolas/desktop/quantitative_finance/FEC/ml-american-options/optiondx/"
SPY_pwd_q1_eod = optiondx_pwd + "spy_eod_2023q1-zfoivd/"

SPY_jan23_eod_df_raw = pd.read_csv(
    SPY_pwd_q1_eod + "spy_eod_202301.txt", delimiter=", ")

timeser_underlying_unique = SPY_jan23_eod_df_raw.groupby("[QUOTE_READTIME]")["[UNDERLYING_LAST]"].last().reset_index()

call_dropables = ["[QUOTE_UNIXTIME]", "[QUOTE_READTIME]", "[QUOTE_TIME_HOURS]", "[EXPIRE_UNIX]", "[P_BID]", "[P_ASK]", "[P_SIZE]", "[P_LAST]", "[P_DELTA]", "[P_GAMMA]", "[P_VEGA]", "[P_THETA]", "[P_RHO]", "[P_IV]", "[P_VOLUME]"]
put_dropables = ["[QUOTE_UNIXTIME]", "[QUOTE_READTIME]", "[QUOTE_TIME_HOURS]", "[EXPIRE_UNIX]", "[C_DELTA]", "[C_GAMMA]", "[C_VEGA]", "[C_THETA]", "[C_RHO]", "[C_IV]", "[C_VOLUME]", "[C_LAST]", "[C_SIZE]", "[C_BID]", "[C_ASK]" ]

df_calls = SPY_jan23_eod_df_raw.drop(columns=call_dropables)
df_puts = SPY_jan23_eod_df_raw.drop(columns=put_dropables)