import pandas as pd
import numpy as np
import math
import indicators as ind
import env_config as env
from sklearn.linear_model import LinearRegression


""" Alternative feature set. Approach: less features and closer to intuitive performance correlation
    (what would I look at?).

    - regression line percentage gain per hour calculated on last 10d basis
    - regression line percentage gain per hour calculated on last 12h basis
    - regression line percentage gain per hour calculated on last 4h basis
    - regression line percentage gain per hour calculated on last 0.5h basis
    - regression line percentage gain per hour calculated on last 5min basis for last 2x5min periods
    - percentage of last 5min mean volume compared to last 1h and to 12h mean volume
    - not directly used: SDA = absolute standard deviation of price points above regression line
    - chance: SDA - (current price - regression price) (== up potential) for 12h, 4h, 0.5h, 5min regression
    - not directly used: SDB = absolute standard deviation of price points below regression line
    - risk: SDB + (current price - regression price) (== down potential) for 12h, 4h, 0.5h, 5min regression
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


def __linregr_gain_price(linregr, x_vec, y_vec, regr_period, offset=0):
    """ Receives a linear regression object, x and y float arrays, the regression period, offset into the past.
        The array must have at least the size of the regression period.

        Returns the regression gain of 60 elements (== 1h if frequency is minutes) and
        the regression price of the most recent (== last) element.
    """
    linregr.fit(x_vec[-regr_period-offset:-1-offset], y_vec[-regr_period-offset:-1-offset])  # perform linear regression
    y_pred = linregr.predict(x_vec[[0, -1]])[:, 0]  # sufficient to predict 1st and last x_vec points
    delta = y_pred[-1] - y_pred[0]
    timediff_factor = 60 / (regr_period - 1)  # constraint: y_vec values have consecutive 1 minute distance
    delta = delta * timediff_factor
    return (delta, y_pred[-1])


def __linregr_gain_price_chance_risk(linregr, x_vec, y_vec, regr_period, offset=0):
    """ Receives a linear regression object, x and y float arrays, the regression period, offset into the past.
        The array must have at least the size of the regression period.

        Returns the regression gain of 60 elements (== 1h if frequency is minutes),
        the regression price of the most recent (== last) element,
        chance of most recent price, risk of most recent price.
    """
    x_vec = x_vec[-regr_period-offset:-1-offset]
    y_vec = y_vec[-regr_period-offset:-1-offset]
    linregr.fit(x_vec, y_vec)  # perform linear regression
    y_pred = linregr.predict(x_vec[:])[:, 0]  # all x_vec points
    y_mean = y_vec.mean()

    sda = math.sqrt(np.mean(np.absolute(y_vec[y_vec >= y_pred] - y_mean)**2))
    chance = sda - (y_vec[-1] - y_pred[-1])

    sdb = math.sqrt(np.mean(np.absolute(y_vec[y_vec < y_pred] - y_mean)**2))
    risk = sdb - (y_vec[-1] - y_pred[-1])

    delta = y_pred[-1] - y_pred[0]
    timediff_factor = 60 / (regr_period - 1)  # constraint: y_vec values have consecutive 1 minute distance
    delta = delta * timediff_factor
    return (delta, y_pred[-1], chance, risk)


def __vol_rel(volumes, long_period, short_period=5):
    """ Receives a float array of volumes, a short and a long period.

        Returns the ratio of the short period volumes mean and the long period volumes mean.
    """
    return volumes[-short_period:].mean() / volumes[-long_period:].mean()


def __cal_features(prices):
    """ Receives a float numpy array of consecutive prices in fixed minute frequency starting with the oldest price.
        Returns a structured numpy array of features per price.
    """
    maxh = 10*24*60  # == maximum history
    if prices.ndim > 1:
        print("WARNING: unexpected close array dimension {close.ndim}")
        return None
    if prices.size <= maxh:
        print(f"WARNING: insufficient length of price vector tocalculate any features")
        return None
    resulttype = np.dtype(
        [("regr_prc10d", "f8"), ("regr_prc12h", "f8"), ("regr_prc4h", "f8"), ("regr_prc30m", "f8"),
         ("regr_prc5m_1", "f8"), ("regr_prc5m_0", "f8"),
         ("regr_gain10d", "f8"), ("regr_gain12h", "f8"), ("regr_gain4h", "f8"), ("regr_gain30m", "f8"),
         ("regr_gain5m_1", "f8"), ("regr_gain5m_0", "f8"),
         ("vol5m_rel1h", "f8"), ("vol5m_rel12h", "f8"),
         ("chance12h", "f8"), ("chance4h", "f8"), ("chance30m", "f8"), ("chance5m", "f8"),
         ("risk12h", "f8"), ("risk4h", "f8"), ("risk30m", "f8"), ("risk5m", "f8")])
    result = np.zeros(prices.size-maxh+1, resulttype)
    xfull = np.arange(maxh).reshape(-1, 1)
    prices = prices.reshape(-1, 1)
    linregr = LinearRegression()  # create object for the class
    for pix in range(maxh, prices.size):
        result["regr_prc10d"][pix-maxh], result["regr_prc10d"][pix-maxh] = \
            __linregr_gain_price(linregr, xfull, prices[pix:], 10*24*60, 0)
        result["regr_prc12h"][pix-maxh], result["regr_gain12h"][pix-maxh], \
            result["chance12h"][pix-maxh], result["risk12h"][pix-maxh] = \
            __linregr_gain_price_chance_risk(linregr, xfull, prices[pix:], 12*60, 0)
        result["regr_prc4h"][pix-maxh], result["regr_gain4h"][pix-maxh], \
            result["chance4h"][pix-maxh], result["risk4h"][pix-maxh] = \
            __linregr_gain_price_chance_risk(linregr, xfull, prices[pix:], 4*60, 0)
        result["regr_prc30m"][pix-maxh], result["regr_gain30m"][pix-maxh], \
            result["chance30m"][pix-maxh], result["risk30m"][pix-maxh] = \
            __linregr_gain_price_chance_risk(linregr, xfull, prices[pix:], 30, 0)
        result["regr_prc5m_1"][pix-maxh], result["regr_gain5m_1"][pix-maxh] = \
            __linregr_gain_price(linregr, xfull, prices[pix:], 5, 5)
        result["regr_gain5m_0"][pix-maxh], result["regr_gain5m_0"][pix-maxh], \
            result["chance5m"][pix-maxh], result["risk5m"][pix-maxh] = \
            __linregr_gain_price_chance_risk(linregr, xfull, prices[pix:], 5, 0)


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
