import pandas as pd
import numpy as np
import math
# import indicators as ind
import env_config as env
from env_config import Env
import cached_crypto_data as ccd
from crypto_features import TargetsFeatures
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
    - chance: 2*SDA - (current price - regression price) (== up potential) for 12h, 4h, 0.5h, 5min regression
    - not directly used: SDB = absolute standard deviation of price points below regression line
    - risk: 2*SDB + (current price - regression price) (== down potential) for 12h, 4h, 0.5h, 5min regression
"""
# REGRESSION_KPI = [(0, 0, 5, "5m_0"), (1, 5, 5, "5m_5")]
REGRESSION_KPI = [(1, 0, 5, "5m_0"), (1, 5, 5, "5m_5"),
                  (0, 0, 30, "30m"), (0, 0, 4*60, "4h"),
                  (0, 0, 12*60, "12h"), (1, 0, 10*24*60, "10d")]
MHE = max([offset+minutes for (regr_only, offset, minutes, ext) in REGRESSION_KPI]) - 1
# VOL_KPI = [(5, 60, "5m1h")]
# MHE == max history elements last element is part of history
VOL_KPI = [(5, 60, "5m1h"), (5, 12*60, "5m12h")]
COL_PREFIX = ["price", "gain", "chance", "risk", "vol"]


def check_input_consistency(df):
    ok = True
    rng = pd.date_range(start=df.index[0], end=df.index[len(df)-1], freq="T")
    rlen = len(rng)
    diff = rng.difference(df.index)
    if len(diff) > 0:
        print(f"{env.nowstr()}: Warning: unexpected index differences: len(df)={len(df)} vs range={rlen}")
        # print(diff)
        ok = False
    # if "open" not in df:
    #     print(f"{env.nowstr()} ERROR: missing 'open' column")
    #     ok = False
    # if "high" not in df:
    #     print(f"{env.nowstr()} ERROR: missing 'high' column")
    #     ok = False
    # if "low" not in df:
    #     print(f"{env.nowstr()} ERROR: missing 'low' column")
    #     ok = False
    if "close" not in df:
        print(f"{env.nowstr()} ERROR: missing 'close' column")
        ok = False
    if "volume" not in df:
        print(f"{env.nowstr()} ERROR: missing 'volume' column")
        ok = False
    if len(df) <= MHE:
        print(f"len(df) = {len(df)} <= len for required history data elements {MHE}")
        ok = False
    return ok


def __linregr_gain_price(linregr, x_vec, y_vec, regr_period, offset=0):
    """ Receives a linear regression object, x and y float arrays, the regression period, offset into the past.
        The array must have at least the size of the regression period.

        Returns the regression gain of 60 elements (== 1h if frequency is minutes) and
        the regression price of the most recent (== last) element.
    """
    if offset != 0:
        x_vec = x_vec[-regr_period-offset:-offset]
        y_vec = y_vec[-regr_period-offset:-offset]
    else:
        x_vec = x_vec[-regr_period:]
        y_vec = y_vec[-regr_period:]
    linregr.fit(x_vec, y_vec)  # adapt linear regression
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
    if offset != 0:
        x_vec = x_vec[-regr_period-offset:-offset]
        y_vec = y_vec[-regr_period-offset:-offset]
    else:
        x_vec = x_vec[-regr_period:]
        y_vec = y_vec[-regr_period:]
    linregr.fit(x_vec, y_vec)  # adapt linear regression
    y_pred = linregr.predict(x_vec[:])[:, 0]  # all x_vec points
    y_vec = y_vec.reshape(-1)
    y_mean = y_vec.mean()

    filter = y_vec > y_pred
    if filter.any():
        sda = math.sqrt(np.mean(np.absolute(y_vec[y_vec > y_pred] - y_mean)**2))
    else:
        sda = 0  # no vector elements above regression line in filled gaps, i.e. y_vec == y_mean
    chance = (2 * sda - (y_vec[-1] - y_pred[-1]))  # / sda

    filter = (y_vec < y_pred)
    if filter.any():
        sdb = math.sqrt(np.mean(np.absolute(y_vec[filter] - y_mean)**2))
    else:
        sdb = 0  # no vector elements below regression line in filled gaps, i.e. y_vec == y_mean

    risk = (2 * sdb + (y_vec[-1] - y_pred[-1]))  # / sdb

    delta = y_pred[-1] - y_pred[0]
    timediff_factor = 60 / (regr_period - 1)  # constraint: y_vec values have consecutive 1 minute distance
    delta = delta * timediff_factor
    return (delta, y_pred[-1], chance, risk)


def __vol_rel(volumes, long_period, short_period=5):
    """ Receives a float array of volumes, a short and a long period.

        Returns the ratio of the short period volumes mean and the long period volumes mean.
    """
    assert volumes.size > 0
    return volumes[-short_period:].mean() / volumes[-long_period:].mean()


def cal_features(ohlcv):
    """ Receives a float ohlcv DataFrame of consecutive prices in fixed minute frequency starting with the oldest price.
        Returns a DataFrame of features per price base on close prices and volumes
        that begins 'MHE' minutes later than 'ohlcv'.
    """
    cols = [
        COL_PREFIX[ix] + ext
        for regr_only, offset, minutes, ext in REGRESSION_KPI
        for ix in range(4) if ((ix < 2) or (not regr_only))]
    cols = cols + [COL_PREFIX[4] + ext for (svol, lvol, ext) in VOL_KPI]
    prices = ohlcv["close"]
    volumes = ohlcv["volume"].values
    df = pd.DataFrame(columns=cols, index=prices.index[MHE:], dtype=np.double)
    xfull = np.arange(MHE+1).reshape(-1, 1)
    prices = prices.values.reshape(-1, 1)
    linregr = LinearRegression()  # create object for the class
    for pix in range(MHE, prices.size):  # pix == prices index
        tic = df.index[pix - MHE]
        for (regr_only, offset, minutes, ext) in REGRESSION_KPI:
            if regr_only:
                df.loc[tic, ["gain"+ext, "price"+ext]] = \
                    __linregr_gain_price(linregr, xfull, prices[:pix+1], minutes, offset)
            else:
                df.loc[tic, ["gain"+ext, "price"+ext, "chance"+ext, "risk"+ext]] = \
                    __linregr_gain_price_chance_risk(linregr, xfull, prices[:pix+1], minutes, offset)
        for (svol, lvol, ext) in VOL_KPI:
            df["vol"+ext][tic] = __vol_rel(volumes, lvol, svol)
    return df


def calc_features_nocache(minute_data):
    """ minute_data has to be in minute frequency and shall have 'MHE' minutes more history elements than
        the oldest returned element with feature data.
        The calculation is performed on 'close' and 'volume' data.
    """
    if not check_input_consistency(minute_data):
        return None
    return cal_features(minute_data)


class CondensedFeatures(TargetsFeatures):
    """ Holds the source ohlcv data as pandas DataFrame 'minute_data',
        the feature vectors as DataFrame rows of 'vec' and
        the target trade signals in column 'target' in both DataFrames.

    """

    def __init__(self, base, minute_dataframe=None, path=None):
        self.feature_type = "Fcondensed1"
        super().__init__(base, minute_dataframe, path)

    def calc_features(self, minute_data):
        """ minute_data has to be in minute frequency and shall have 'MHE' minutes more history elements than
            the oldest returned element with feature data.
            The calculation is performed on 'close' and 'volume' data.
        """
        if not check_input_consistency(minute_data):
            return None
        return cal_features(minute_data)


if __name__ == "__main__":
    if True:
        cdf = ccd.load_asset_dataframe("btc", path=Env.data_path, limit=MHE+10)
        features = cal_features(cdf)
        print(cdf.head(5))
        print(cdf.tail(5))
        print(features.head(5))
        print(features.tail(5))
    else:
        cols = [
            COL_PREFIX[ix] + ext
            for regr_only, offset, minutes, ext in REGRESSION_KPI
            for ix in range(4) if ((ix < 2) or (not regr_only))]
        cols = cols + [COL_PREFIX[4] + ext for (svol, lvol, ext) in VOL_KPI]
        print(cols)
