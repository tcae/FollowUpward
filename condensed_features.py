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

    2020-05-17: the problem with this festure set is that it is not clipped
    to handle extreme values. I saw in Dec 2017 amplitude of 40% per minute and amplitudes of several hundred %
    which dominates training.
    New approach: clip at 5% / h regression gradient and at +-1% distance to 4h regression line.
    Also remove regression line of 10d, 5min-5min offset, gain and risk calculation.
    Introduce distance to 4h regression line and RSI for 4h and 2h, 1h, 15 min regression gradient.
"""
import pandas as pd
import numpy as np
import math
import logging
# import indicators as ind
# import env_config as env
# from env_config import Env
import cached_crypto_data as ccd
# import crypto_features as cf
# import crypto_targets as ct
from sklearn.linear_model import LinearRegression
import bottleneck as bn

logger = logging.getLogger(__name__)


# REGRESSION_KPI = [(0, 0, 5, "5m_0"), (1, 5, 5, "5m_5")]
# (regression_only, offset, regression_minutes, label)
# regression_only == TRUE: 3 features = slope, last point y
# regression_only == FALSE: 5 features = slope, last point y, risk, chance
REGRESSION_KPI = [(1, 0, 5, "5m_0"), (1, 5, 5, "5m_5"),
                  (0, 0, 30, "30m"), (0, 0, 4*60, "4h"),
                  (0, 0, 12*60, "12h"), (1, 0, 10*24*60, "10d")]
FEATURE_COUNT = 3 * 1 + 3 * 3 + 2  # 12 + 2 vol (see below) = 14 features per sample
HMWF = max([offset+minutes for (regr_only, offset, minutes, ext) in REGRESSION_KPI]) - 1
# HMWF == history minutes required without features
# VOL_KPI = [(5, 60, "5m1h")]
VOL_KPI = [(5, 60, "5m1h"), (5, 12*60, "5m12h")]
COL_PREFIX = ["gain", "chance", "risk", "vol"]


def check_input_consistency(df):
    ok = True
    rng = pd.date_range(start=df.index[0], end=df.index[len(df)-1], freq="T")
    rlen = len(rng)
    diff = rng.difference(df.index)
    if len(diff) > 0:
        logger.warning(f"unexpected index differences: len(df)={len(df)} vs range={rlen}")
        # logger.debug(f"{diff}")
        ok = False
    # if "open" not in df:
    #     logger.debug("missing 'open' column")
    #     ok = False
    # if "high" not in df:
    #     logger.debug("missing 'high' column")
    #     ok = False
    # if "low" not in df:
    #     logger.debug("missing 'low' column")
    #     ok = False
    if "close" not in df:
        logger.error("missing 'close' column")
        ok = False
    if "volume" not in df:
        logger.error("missing 'volume' column")
        ok = False
    if len(df) <= HMWF:
        logger.warning(f"len(df) = {len(df)} <= len for required history data elements {HMWF+1}")
        ok = False
    return ok


def _linregr_gain_price(linregr, x_vec, y_vec, regr_period, offset=0):
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
    return (delta)


def _linregr_gain_price_chance_risk(linregr, x_vec, y_vec, regr_period, offset=0):
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
    return (delta, chance, risk)


def _vol_rel(volumes, long_period, short_period=5):
    """ Receives a float array of volumes, a short and a long period.

        Returns the ratio of the short period volumes mean and the long period volumes mean.
    """
    assert volumes.size > 0
    if volumes[-long_period:].mean() == 0:
        return 0.0
    else:
        return volumes[-short_period:].mean() / volumes[-long_period:].mean()


def cal_features(ohlcv):
    """ Receives a float ohlcv DataFrame of consecutive prices in fixed minute frequency starting with the oldest price.
        Returns a DataFrame of features per price base on close prices and volumes
        that begins 'HMWF' minutes later than 'ohlcv.index[0]'.
    """
    cols = [
        COL_PREFIX[ix] + ext
        for regr_only, offset, minutes, ext in REGRESSION_KPI
        for ix in range(3) if ((ix < 1) or (not regr_only))]
    cols = cols + [COL_PREFIX[3] + ext for (svol, lvol, ext) in VOL_KPI]
    prices = ohlcv["close"]
    volumes = ohlcv["volume"].values.reshape(-1, 1)
    df = pd.DataFrame(columns=cols, index=prices.index[HMWF:], dtype=np.double)
    xfull = np.arange(HMWF+1).reshape(-1, 1)
    prices = prices.values.reshape(-1, 1)
    linregr = LinearRegression()  # create object for the class
    for pix in range(HMWF, prices.size):  # pix == prices index
        tic = df.index[pix - HMWF]
        norm_prices = prices[pix-HMWF:pix+1].copy() / prices[pix]
        for (regr_only, offset, minutes, ext) in REGRESSION_KPI:
            if regr_only:
                df.loc[tic, ["gain"+ext]] = \
                    _linregr_gain_price(linregr, xfull, norm_prices, minutes, offset)
            else:
                df.loc[tic, ["gain"+ext, "chance"+ext, "risk"+ext]] = \
                    _linregr_gain_price_chance_risk(linregr, xfull, norm_prices, minutes, offset)
        for (svol, lvol, ext) in VOL_KPI:
            df["vol"+ext][tic] = _vol_rel(volumes[:pix+1], lvol, svol)
    return df


class F3cond14(ccd.Features):

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        return HMWF

    def load_data(self, base: str):
        df = super().load_data(base)
        return df

    def keys(self):
        "returns the list of element keys"
        cols = [
            COL_PREFIX[ix] + ext
            for regr_only, offset, minutes, ext in REGRESSION_KPI
            for ix in range(3) if ((ix < 1) or (not regr_only))]
        cols = cols + [COL_PREFIX[3] + ext for (svol, lvol, ext) in VOL_KPI]
        return cols

    def mnemonic(self):
        "returns a string that represents this class as mnemonic, e.g. to use it in file names"
        return "F3cond{}".format(FEATURE_COUNT)

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Downloads or calculates new data from 'first' sample up to and including 'last'.
        """
        ohlc_first = first - pd.Timedelta(self.history(), unit="T")
        df = self.ohlcv.get_data(base, ohlc_first, last)
        return cal_features(df)


def regression_line(yarr, window=5):
    """
        This implementation ignores index and assumes an equidistant x values
        yarr is a one dimensional numpy array

        Regression Equation(y) = a + bx
        Slope(b) = (NΣXY - (ΣX)(ΣY)) / (NΣ(X^2) - (ΣX)^2)
        Intercept(a) = (ΣY - b(ΣX)) / N
        used from https://www.easycalculation.com/statistics/learn-regression.php

        returns 2 numpy arrays: slope and regr_end
        slope are the gradients
        regr_end are the regression line last points
    """
    # yarr = df.to_numpy()
    # xarr = df.index.values
    yarr = yarr.flatten()
    len = yarr.shape[0]
    if len == 0:
        return np.full((len), np.nan, dtype=np.float), np.full((len), np.nan, dtype=np.float)
    xarr = np.array([x for x in range(len)], dtype=float)
    xyarr = xarr * yarr
    xy_mov_sum = bn.move_sum(xyarr, window=window)
    x_mov_sum = bn.move_sum(xarr, window=window)
    x2arr = xarr * xarr
    x2_mov_sum = bn.move_sum(x2arr, window=window)
    x_mov_sum_2 = x_mov_sum * x_mov_sum
    y_mov_sum = bn.move_sum(yarr, window=window)
    slope = (window * xy_mov_sum - (x_mov_sum * y_mov_sum)) / (window * x2_mov_sum - x_mov_sum_2)
    intercept = (y_mov_sum - (slope * x_mov_sum)) / window
    regr_end = xarr * slope + intercept
    return slope, regr_end


def _regression_unit_test():
    df = pd.DataFrame(data={"y": [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]}, index=[59, 60, 61, 62, 63, 65, 65], dtype=float)
    slope, regr_end = regression_line(df["y"].to_numpy(), len(df))
    df = pd.DataFrame(index=df.index)
    df["slope"] = slope
    # df["intercept"] = intercept
    df["regr_end"] = regr_end
    # assert np.equal(df.loc[63:, "slope"].values.round(2), np.array([0.29, 0.24, 0.31])).all()
    # print(df)
    df = df.dropna()
    print(df)
    # print(df.loc[63:, "slope"])


def relative_volume(volumes, short_window=5, large_window=60):
    """
        This implementation ignores index and assumes an equidistant x values
        yarr is a one dimensional numpy array

        returns a numpy array with short mean/long mean relation values
    """
    if volumes.shape[0] < large_window:
        return np.full((volumes.shape[0]), np.nan, dtype=np.float)
    short_mean = bn.move_mean(volumes, short_window)
    large_mean = bn.move_mean(volumes, large_window)
    vol_rel = short_mean / large_mean
    return vol_rel


def _relative_volume_unit_test():
    df = pd.DataFrame(data={"y": [np.NaN, 2, 3, 4, 5, 0, 0]}, dtype=float)
    df["vol_rel"] = relative_volume(df["y"].to_numpy(), 2, 5)
    assert np.equal(df.loc[5:, "vol_rel"].values.round(2), np.array([0.89, 0.])).all()
    print(df)
    # df = df.dropna()
    # print(df)


def rolling_pivot_range(ohlcv_df, minutes_agg):
    """Time aggregation through rolling aggregation with the consequence that new data is
    generated every minute and even long time aggregations reflect all minute bumps in their
    features

    range: high-low
    pivot: average(open, high, low, close)

    in:
        dataframe of minute ohlcv data of a currency pair
        with the columns: open, high, low, close.
        minutes_agg is the number of minutes used to aggregate pivot and ranges
    out:
        dataframe of pivot and range aggregations
    """
    df = pd.DataFrame()
    df["range"] = ohlcv_df.high.rolling(minutes_agg).max() - ohlcv_df.low.rolling(minutes_agg).min()
    df["pivot"] = (
        ohlcv_df.open.rolling(minutes_agg).mean() + ohlcv_df.high.rolling(minutes_agg).mean() +
        ohlcv_df.low.rolling(minutes_agg).mean() + ohlcv_df.close.rolling(minutes_agg).mean()) / 4
    return df


def _rolling_pivot_range_unit_test():
    cols = ["open", "high", "low", "close", "volume"]
    dat = [
        [0, 0, 0, 0, 0],
        [1, 4, 1, 1, 1],
        [2, 2, 0, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]]
    df = pd.DataFrame(
        data=dat,
        columns=cols, index=pd.date_range("2012-10-08 18:15:05", periods=5, freq="T", tz="Europe/Amsterdam"))
    print(df)
    ndf = rolling_pivot_range(df, 1)
    print(ndf)
    ndf = rolling_pivot_range(df, 2)
    print(ndf)


def expand_feature_vectors(feature_df, minutes_agg, periods):
    """ Builds a feature vector of all columns found in feature_df
        that are assumed rolling aggregations of length minutes_agg.
        'periods' features with minutes_agg distance are copied into the expanded feature df columns.

    Result:
        A DataFrame with feature vectors as rows. The column name indicates
        the type of feature, e.g. 'pivot' or 'range' with aggregation+'T_'+period
        as column prefix.
    """
    df = pd.DataFrame(index=feature_df.index)
    for period in range(periods):
        ctitle = str(minutes_agg) + "T_" + str(period) + "_"
        offset = period*minutes_agg
        # now add feature columns according to aggregation
        for col in feature_df.columns:
            df[ctitle + col] = feature_df[col].shift(offset)
    df = df.dropna()
    if df.empty:
        logger.warning("empty dataframe from expand_feature_vectors")
    return df


def _expand_feature_vectors_unit_test():
    cols = ["open", "high", "low", "close", "volume"]
    dat = [
        [0, 0, 0, 0, 0],
        [1, 4, 1, 1, 1],
        [2, 2, 0, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5]]
    df = pd.DataFrame(
        data=dat,
        columns=cols, index=pd.date_range("2012-10-08 18:15:05", periods=6, freq="T", tz="Europe/Amsterdam"))
    print(df)
    ndf = rolling_pivot_range(df, 1)
    print(ndf)
    # ndf = rolling_pivot_range(df, 2)
    # ndf = ndf.dropna()
    # print(ndf)
    ndf = expand_feature_vectors(ndf, 2, 3)
    print(ndf)


def pivot_range(ohlcv_df: pd.DataFrame, aggregation_minutes: int = 1):
    """ Receives a float ohlcv_df DataFrame of consecutive prices in fixed minute frequency
        and a number aggregation_minutes that identifies the number of minutes bundled.
        Returns 2 numpy arrays pivots and ranges of length as the incoming ohlcv_df.
    """
    time_aggs = {5: 12}
    if (ohlcv_df is None) or ohlcv_df.empty:
        ranges = np.array([], dtype=np.float)
        pivots = np.array([], dtype=np.float)
    else:
        time_agg = f"{aggregation_minutes}T"
        ranges = np.array((
            ), dtype=np.float)
        pivots = np.array(((
            ohlcv_df["open"].to_numpy() +
            ohlcv_df["close"].to_numpy() +
            ohlcv_df["high"].to_numpy() +
            ohlcv_df["low"].to_numpy()) / 4), dtype=np.float)

        # ! under constructioin
        high = ohlcv_df["open"].resample(time_agg-1)
        high = ohlcv_df["high"].resample(time_agg).max()
        high = ohlcv_df["low"].resample(time_agg).min()
        high = ohlcv_df["close"].resample(time_agg).max()
        ranges = \
            high.to_numpy() - \
            ohlcv_df["low"].rolling(time_agg).min().to_numpy()
        pivots = (ohlcv_df["open"].to_numpy() +
                  ohlcv_df["close"].to_numpy() +
                  ohlcv_df["high"].to_numpy() +
                  ohlcv_df["low"].to_numpy()) / 4  # == pivot price

    return pivots, ranges


def __pivot_regression__relative_volume_unit_test():
    df = pivot_range(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))
    print(df)
    print(df.empty)


class F4cond14(ccd.Features):

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        return HMWF

    def load_data(self, base: str):
        df = super().load_data(base)
        return df

    def keys(self):
        "returns the list of element keys"
        cols = [
            COL_PREFIX[ix] + ext
            for regr_only, offset, minutes, ext in REGRESSION_KPI
            for ix in range(3) if ((ix < 1) or (not regr_only))]
        cols = cols + [COL_PREFIX[3] + ext for (svol, lvol, ext) in VOL_KPI]
        return cols

    def mnemonic(self):
        "returns a string that represents this class as mnemonic, e.g. to use it in file names"
        return "F3cond{}".format(FEATURE_COUNT)

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Downloads or calculates new data from 'first' sample up to and including 'last'.
        """
        ohlcv_first = first - pd.Timedelta(self.history(), unit="T")
        ohlcv_df = self.ohlcv.get_data(base, ohlcv_first, last)
        df = pd.DataFrame(index=ohlcv_df.index)
        df["pivot"], df["range"] = pivot_range(ohlcv_df)
        yarr = df["pivot"].to_numpy()

        df["grad_5m"], df["level_5m"] = regression_line(yarr, window=5)
        df["grad_15m"], df["level_15m"] = regression_line(yarr, window=15)
        df["grad_30m"], df["level_30m"] = regression_line(yarr, window=30)
        df["grad_1h"], df["level_1h"] = regression_line(yarr, window=60)
        df["grad_2h"], df["level_2h"] = regression_line(yarr, window=2*60)
        df["grad_4h"], df["level_4h"] = regression_line(yarr, window=4*60)
        df["grad_12h"], df["level_12h"] = regression_line(yarr, window=12*60)

        df["vol_5m12h"] = relative_volume(ohlcv_df["volume"].to_numpy(), short_window=5, large_window=12*60)
        df["vol_5m1h"] = relative_volume(ohlcv_df["volume"].to_numpy(), short_window=5, large_window=60)
        df = df.dropna()
        return


def feature_percentiles():
    bases = ["btc", "xrp", "eos", "bnb", "eth", "neo", "ltc", "trx"]
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    features = F3cond14(ccd.Ohlcv())
    # base_info = {base: features.load_data(base).describe(percentiles=percentiles, include='all') for base in bases}
    base_info = list()
    dfl = list()
    for base in bases:
        df = features.load_data(base)
        dfl.append(df)
        if df is not None:
            base_info.append(df.describe(percentiles=percentiles, include='all'))
        else:
            logger.error(f"unexpected df=None for base {base}")
    df_all = pd.concat(dfl, join="outer", ignore_index=True)
    base_info.append(df_all.describe(percentiles=percentiles, include='all'))
    bases.append("all")
    info_df = pd.concat(base_info, join="outer", keys=bases)
    info_df.index = info_df.index.swaplevel(0, 1)
    print(info_df)


if __name__ == "__main__":
    # feature_percentiles()
    # _regression_unit_test()
    # _relative_volume_unit_test()
    # _pivot_regression__relative_volume_unit_test()
    # _rolling_pivot_range_unit_test()
    _expand_feature_vectors_unit_test()
