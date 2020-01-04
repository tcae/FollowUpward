import pandas as pd
import numpy as np
import indicators as ind
import env_config as env

""" Alternative feature set. Approach: less features and closer to intuitive performance correlation
    (what would I look at?).

    - regression line percentage increase per hour calculated on last 10d basis
    - regression line percentage increase per hour calculated on last 12h basis
    - regression line percentage increase per hour calculated on last 4h basis
    - regression line percentage increase per hour calculated on last 0.5h basis
    - regression line percentage increase per hour calculated on last 5min basis for last 2x5min periods
    - percentage of last 5min mean volume compared to last 1h mean 5min volume
    - not directly used: SDU = absolute standard deviation of price points above regression line
    - SDU - (current price - regression price) (== up potential) for 12h, 4h, 0.5h, 5min regression
    - not directly used: SDD = absolute standard deviation of price points below regression line
    - SDD + (current price - regression price) (== down potential) for 12h, 4h, 0.5h, 5min regression
"""
__REGRESSION_MINUTES = [(0, 5), (5, 5), (0, 30), (0, 4*60), (0, 12*60), (0, 10*24*60)]
__MHE = max([offset+minutes for (offset, minutes) in __REGRESSION_MINUTES]) - 1
# MHE == max history elements last element is part of history


def __check_input_consistency(df):
    ok = True
    diff = pd.date_range(start=df.index[0], end=df.index[len(df)-1], freq="T").difference(df.index)
    if len(diff) > 0:
        print(f"{env.nowstr()}: Warning: unexpected index differences")
        print(diff)
        ok = False
    if "close" not in df:
        print(f"{env.nowstr()} ERROR: missing 'close' column")
        ok = False
    if len(df) <= __MHE:
        print(f"len(df) = {len(df)} <= len for required history data elements {__MHE}")
        ok = False
    return ok


def calc_features(minute_data):
    """ minute_data has to be in minute frequency and shall have 10*24*60 minutes more history elements than
        the oldest returned element with feature data.
        The calculation is performed on 'close' data.
    """

    if not __check_input_consistency(minute_data):
        return None
    vec = minute_data.iloc[__MHE:]
    for tix in range(__MHE, len(minute_data)):
        for (offset, minutes) in __REGRESSION_MINUTES:
            df = minute_data.iloc[(tix-(offset+minutes)):(tix+1-offset)]
            delta, pred = ind.time_linear_regression(df)
            vec.loc[minute_data.index[tix], f"{offset}+{minutes}"] = delta
    return vec
