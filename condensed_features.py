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
    xy_mov_sum = bn.move_sum(xyarr, window=window, min_count=window)
    x_mov_sum = bn.move_sum(xarr, window=window, min_count=window)
    x2arr = xarr * xarr
    x2_mov_sum = bn.move_sum(x2arr, window=window, min_count=window)
    x_mov_sum_2 = x_mov_sum * x_mov_sum
    y_mov_sum = bn.move_sum(yarr, window=window, min_count=window)
    slope = (window * xy_mov_sum - (x_mov_sum * y_mov_sum)) / (window * x2_mov_sum - x_mov_sum_2)
    intercept = (y_mov_sum - (slope * x_mov_sum)) / window
    regr_end = xarr * slope + intercept
    return slope, regr_end


def _regression_test():
    df = pd.DataFrame(data={"y": [2.9, 3.1, 3.6, 3.8, 4, 4.1, 5]}, index=[59, 60, 61, 62, 63, 65, 65], dtype=float)
    slope, regr_end = regression_line(df["y"].to_numpy(), len(df))
    df = pd.DataFrame(index=df.index)
    df["slope"] = slope
    # df["intercept"] = intercept
    df["regr_end"] = regr_end
    # assert np.equal(df.loc[63:, "slope"].values.round(2), np.array([0.29, 0.24, 0.31])).all()
    print(df)
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
    short_mean = bn.move_mean(volumes, short_window, min_count=1)
    large_mean = bn.move_mean(volumes, large_window, min_count=1)
    large_mean[large_mean == 0] = 1  # mitigate division with zero
    vol_rel = short_mean / large_mean
    return vol_rel


def _relative_volume_test():
    df = pd.DataFrame(data={"y": [np.NaN, 2, 3, 4, 5, 0, 0]}, dtype=float)
    df["vol_rel"] = relative_volume(df["y"].to_numpy(), 2, 5)
    assert np.equal(df.loc[5:, "vol_rel"].values.round(2), np.array([0.89, 0.])).all()
    print(df)
    df = df.dropna()
    print(df)


def rolling_range(ohlcv_pivot_df, minutes_agg):
    """Time aggregation through rolling aggregation with the consequence that new data is
    generated every minute and even long time aggregations reflect all minute bumps in their
    features

    range: high-low

    in:
        dataframe of minute ohlcv data of a currency pair
        with the columns: open, high, low, close.
        minutes_agg is the number of minutes used to aggregate ranges
    out:
        dataframe range aggregations
    """
    return ohlcv_pivot_df.high.rolling(minutes_agg, min_periods=1).max() - \
        ohlcv_pivot_df.low.rolling(minutes_agg, min_periods=1).min()


def rolling_pivot(ohlcv_pivot_df, minutes_agg):
    """Time aggregation through rolling aggregation with the consequence that new data is
    generated every minute and even long time aggregations reflect all minute bumps in their
    features

    pivot: is expected as input coulmn as average(open, high, low, close)

    in:
        dataframe of minute ohlcv data of a currency pair
        with the columns: pivot.
        minutes_agg is the number of minutes used to aggregate pivot
    out:
        numpy of pivot aggregations
    """
    return bn.move_mean(ohlcv_pivot_df["pivot"].to_numpy(), window=minutes_agg, min_count=1)


def _rolling_pivot_range_test():
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
    df["pivot"] = ccd.calc_pivot(df)
    print(df)
    ndf = pd.DataFrame(index=df.index)
    ndf["range"] = rolling_range(df, 1)
    ndf["pivot"] = rolling_pivot(df, 1)
    print(ndf)
    ndf["range"] = rolling_range(df, 2)
    ndf["pivot"] = rolling_pivot(df, 2)
    print(ndf)


def expand_feature_vectors(feature_df, minutes_agg, periods):
    """ Builds a feature vector of all columns found in feature_df
        that are assumed rolling aggregations of length minutes_agg.
        'periods' features with minutes_agg distance are copied into the expanded feature df columns.

    Result:
        A DataFrame with feature vectors as rows. The column name indicates
        the type of feature, e.g. 'pivot' or 'range' with aggregation+'T_'+period
        as column prefix.
        Consider NaN elements in the first minutes_agg*(periods-1) rows
    """
    df = pd.DataFrame(index=feature_df.index)
    for period in range(periods):
        ctitle = str(period+1)
        offset = period*minutes_agg
        # now add feature columns according to aggregation
        for col in feature_df.columns:
            df[col+ctitle] = feature_df[col].shift(offset)
    # df = df.dropna()
    if df.empty:
        logger.warning("empty dataframe from expand_feature_vectors")
    return df


def _expand_feature_vectors_test():
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
    df["pivot"] = ccd.calc_pivot(df)
    print(df)
    rdf = pd.DataFrame(index=df.index)
    rdf["range"] = rolling_range(df, 1)
    rdf["pivot"] = rolling_pivot(df, 1)
    print(rdf)
    # ndf = rolling_pivot_range(df, 2)
    ndf = expand_feature_vectors(rdf, 2, 3)
    print(ndf)
    ndf = ndf.dropna()
    print(ndf)


def delta_pivot_features(ohlcvp_df, suffix, minutes, periods):
    """ Returns the subsequent difference of 'minutes' rolling pivot values for the last 'periods'
        normalized with the last pivot price value.
        It is assumed (and not checked) that ovphlcvp contains values of minute frequency without gaps.
    """
    if "pivot" not in ohlcvp_df:
        ohlcvp_df["pivot"] = ccd.calc_pivot(ohlcvp_df)
    rdf = pd.DataFrame(index=ohlcvp_df.index)
    pcol = "dpiv"+suffix
    rdf[pcol] = rolling_pivot(ohlcvp_df, minutes)

    if periods <= 1:
        logger.error(f"no features due to unexpected periods {periods}")
    # now expand vector
    ndf = pd.DataFrame(index=rdf.index)
    for period in range(periods):
        ctitle = str(period)
        offset = period*minutes
        # now add percentage delta of pivot price
        ndf[pcol+ctitle] = (rdf[pcol].shift(offset) - rdf[pcol].shift(offset+minutes)) / ohlcvp_df["pivot"]
    return ndf


def range_features(ohlcvp_df, suffix, minutes, periods):
    """ Returns the rolling range (high - low) over 'minutes' values for the last 'periods'
        normalized with the last pivot price value.
        It is assumed (and not checked) that ovphlcvp contains values of minute frequency without gaps.
    """
    if "pivot" not in ohlcvp_df:
        ohlcvp_df["pivot"] = ccd.calc_pivot(ohlcvp_df)
    rdf = pd.DataFrame(index=ohlcvp_df.index)
    rcol = "range"+suffix
    rdf[rcol] = rolling_range(ohlcvp_df, minutes)

    if periods <= 1:
        logger.error(f"no features due to unexpected periods {periods}")
    # now expand vector
    ndf = pd.DataFrame(index=rdf.index)
    for period in range(periods):
        ctitle = str(period)
        offset = period*minutes
        # now normalize the price range
        ndf[rcol+ctitle] = rdf[rcol].shift(offset) / ohlcvp_df["pivot"]
    return ndf


def regression_features(ohlcvp_df, suffix, minutes):
    """ Returns the rolling gradient and last price of a 1D linear regression over 'minutes' values
        normalized with the last pivot price value.
        It is assumed (and not checked) that ovphlcvp contains values of minute frequency without gaps.
    """
    if "pivot" not in ohlcvp_df:
        ohlcvp_df["pivot"] = ccd.calc_pivot(ohlcvp_df)
    rdf = pd.DataFrame(index=ohlcvp_df.index)
    grad_col = "grad"+suffix
    level_col = "lvl"+suffix
    rdf[grad_col], rdf[level_col] = regression_line(ohlcvp_df["pivot"].to_numpy(), window=minutes)
    rdf[grad_col] = rdf[grad_col] / ohlcvp_df["pivot"]
    rdf[level_col] = rdf[level_col] / ohlcvp_df["pivot"]
    return rdf


def volume_features(ohlcvp_df, suffix, minutes, norm_minutes):
    """ Returns the rolling volume relation of 'minutes' / 'norm_minutes'.
        It is assumed (and not checked) that ohlcvp contains values of minute frequency without gaps.
    """
    rdf = pd.DataFrame(index=ohlcvp_df.index)
    vol_col = "vol"+suffix
    rdf[vol_col] = relative_volume(ohlcvp_df["volume"].to_numpy(), short_window=minutes, large_window=norm_minutes)
    return rdf


class F4CondAgg(ccd.Features):

    def __init__(self, ohlcv: ccd.Ohlcv):
        super().__init__(ohlcv)
        self.func = {
            "dpiv": delta_pivot_features,
            "range": range_features,
            "regr": regression_features,
            "vol": volume_features
        }
        self.config = {
            "dpiv": [  # dpiv == difference of subsequent pivot prices
                {"suffix": "5m", "minutes": 5, "periods": 12}],
            "range": [
                {"suffix": "5m", "minutes": 5, "periods": 12},
                {"suffix": "1h", "minutes": 60, "periods": 4}],
            "regr": [  # regr == regression
                {"suffix": "5m", "minutes": 5},
                {"suffix": "15m", "minutes": 15},
                {"suffix": "30m", "minutes": 30},
                {"suffix": "1h", "minutes": 60},
                {"suffix": "2h", "minutes": 2*60},
                {"suffix": "4h", "minutes": 4*60},
                {"suffix": "12h", "minutes": 12*60}],
            "vol": [
                {"suffix": "5m12h", "minutes": 5, "norm_minutes": 12*60},
                {"suffix": "5m1h", "minutes": 5, "norm_minutes": 60}]
        }

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        hl = list()
        for aspect in self.config:
            if (aspect == "dpiv") or (aspect == "range"):
                hl.extend([self.config[aspect][ix]["minutes"] *
                           self.config[aspect][ix]["periods"] for ix in range(len(self.config[aspect]))])
            elif aspect == "regr":
                hl.extend([self.config[aspect][ix]["minutes"] for ix in range(len(self.config[aspect]))])
            elif aspect == "vol":
                hl.extend([self.config[aspect][ix]["norm_minutes"] for ix in range(len(self.config[aspect]))])
            else:
                logger.error(f"unexpected aspect {aspect}")
        return max(hl)

    def load_data(self, base: str):
        df = super().load_data(base)
        return df

    def keys(self):
        "returns the list of element keys"
        kl = list()
        for aspect in self.config:
            for ix in range(len(self.config[aspect])):
                if (aspect == "dpiv") or (aspect == "range"):
                    kl.extend([
                        aspect + self.config[aspect][ix]["suffix"] + str(p+1)
                        for p in range(self.config[aspect][ix]["periods"])])
                elif aspect == "regr":
                    kl.append("grad" + self.config[aspect][ix]["suffix"])
                    kl.append("lvl" + self.config[aspect][ix]["suffix"])
                elif aspect == "vol":
                    kl.append(aspect + self.config[aspect][ix]["suffix"])
                else:
                    logger.error(f"unexpected aspect {aspect}")
        return kl

    def mnemonic(self):
        "returns a string that represents this class as mnemonic, e.g. to use it in file names"
        return "F4CondAgg"

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Downloads or calculates new data from 'first' sample up to and including 'last'.
        """
        ohlcv_first = first - pd.Timedelta(self.history(), unit="T")
        ohlcv_df = self.ohlcv.get_data(base, ohlcv_first, last)
        df = pd.DataFrame(index=ohlcv_df.index)
        for aspect in self.config:
            for ix in range(len(self.config[aspect])):
                args = [self.config[aspect][ix][e] for e in self.config[aspect][ix]]
                adf = self.func[aspect](ohlcv_df, *args)
                for col in adf:
                    # to_numpy() saves index check
                    df[col] = adf[col].to_numpy()

        df = df.dropna()
        return df

    def new_data_test(self):
        print(f"history: {self.history()}")
        kl = self.keys()
        print(f"len(keys): {len(kl)} keys: {kl}")
        self.config = {
            "dpiv": [
                {"suffix": "2m", "minutes": 2, "periods": 2}],
            "range": [
                {"suffix": "2m", "minutes": 2, "periods": 2}],
            "regr": [  # regr == regression
                {"suffix": "2m", "minutes": 2}],
            "vol": [
                {"suffix": "2m3m", "minutes": 2, "norm_minutes": 3}]
        }
        print(f"history: {self.history()}")
        print(self.keys())
        cols = ["open", "high", "low", "close", "volume"]
        dat = [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 2],
            [2, 2, 1, 1, 3],
            [3, 3, 2, 2, 4],
            [4, 4, 3, 3, 5],
            [6, 6, 5, 5, 6]]
        ohlcv_df = pd.DataFrame(
            data=dat,
            columns=cols, index=pd.date_range("2012-10-08 18:15:05", periods=6, freq="T", tz="Europe/Amsterdam"))
        ohlcv_df["pivot"] = ccd.calc_pivot(ohlcv_df)
        ohlcv_df["pivot2m"] = rolling_pivot(ohlcv_df, 2)
        print(ohlcv_df)
        df = pd.DataFrame(index=ohlcv_df.index)
        for aspect in self.config:
            for ix in range(len(self.config[aspect])):
                args = [self.config[aspect][ix][e] for e in self.config[aspect][ix]]
                print(f"aspect: {aspect} args: {args}")
                adf = self.func[aspect](ohlcv_df, *args)
                print(adf)
                for col in adf:
                    # to_numpy() saves index check
                    df[col] = adf[col].to_numpy()
        kl = self.keys()
        print(f"len(keys): {len(kl)} keys: {kl}")
        df["target"] = df["grad2m"].shift(-1)
        print(df)


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
    # _regression_test()
    # _relative_volume_test()
    # _pivot_regression__relative_volume_test()
    # _rolling_pivot_range_test()
    # _expand_feature_vectors_test()
    F4CondAgg(ccd.Ohlcv()).new_data_test()
