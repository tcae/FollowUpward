import pandas as pd
import numpy as np
import math
import logging
# import indicators as ind
# import env_config as env
from env_config import Env
import cached_crypto_data as ccd
import crypto_features as cf
from crypto_features import TargetsFeatures
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

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
# (regression_only, offset, regression_minutes, label)
# regression_only == TRUE: 3 features = slope, last point y
# regression_only == FALSE: 5 features = slope, last point y, risk, chance
REGRESSION_KPI = [(1, 0, 5, "5m_0"), (1, 5, 5, "5m_5"),
                  (0, 0, 30, "30m"), (0, 0, 4*60, "4h"),
                  (0, 0, 12*60, "12h"), (1, 0, 10*24*60, "10d")]
FEATURE_COUNT = 3 * 2 + 3 * 4 + 2  # 18 + 2 vol (see below) = 20 features per sample
HMWF = max([offset+minutes for (regr_only, offset, minutes, ext) in REGRESSION_KPI]) - 1
# HMWF == history minutes required without features
# VOL_KPI = [(5, 60, "5m1h")]
VOL_KPI = [(5, 60, "5m1h"), (5, 12*60, "5m12h")]
COL_PREFIX = ["price", "gain", "chance", "risk", "vol"]


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
        that begins 'HMWF' minutes later than 'ohlcv.index[0]'.
    """
    cols = [
        COL_PREFIX[ix] + ext
        for regr_only, offset, minutes, ext in REGRESSION_KPI
        for ix in range(4) if ((ix < 2) or (not regr_only))]
    cols = cols + [COL_PREFIX[4] + ext for (svol, lvol, ext) in VOL_KPI]
    prices = ohlcv["close"]
    volumes = ohlcv["volume"].values
    df = pd.DataFrame(columns=cols, index=prices.index[HMWF:], dtype=np.double)
    xfull = np.arange(HMWF+1).reshape(-1, 1)
    prices = prices.values.reshape(-1, 1)
    linregr = LinearRegression()  # create object for the class
    for pix in range(HMWF, prices.size):  # pix == prices index
        tic = df.index[pix - HMWF]
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
    """ minute_data has to be in minute frequency and shall have 'HMWF' minutes more history elements than
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
        super().__init__(base, minute_dataframe, path)

    @staticmethod
    def feature_str():
        "returns a string that represent the features class as mnemonic"
        return "F2cond24"

    @staticmethod
    def feature_count():
        """ returns the number of features for one sample

        - without close and target column
        - regression_only == TRUE: 3 features = slope, last point y, relative volume
        - regression_only == FALSE: 5 features = slope, last point y, risk, chance, relative volume
        - FEATURE_COUNT = 3 * 3 + 3 * 5  # 14 features per sample
        """
        return FEATURE_COUNT

    @staticmethod
    def history():
        "history_minutes_without_features"
        return HMWF

    def calc_features(self, minute_data):
        """ minute_data has to be in minute frequency and shall have 'HMWF' minutes more history elements than
            the oldest returned element with feature data.
            The calculation is performed on 'close' and 'volume' data.
        """
        if not check_input_consistency(minute_data):
            return None
        return cal_features(minute_data)


class F2cond20(ccd.Features):

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        return HMWF

    def keys(self):
        "returns the list of element keys"
        cols = [
            COL_PREFIX[ix] + ext
            for regr_only, offset, minutes, ext in REGRESSION_KPI
            for ix in range(4) if ((ix < 2) or (not regr_only))]
        cols = cols + [COL_PREFIX[4] + ext for (svol, lvol, ext) in VOL_KPI]
        return cols

    def mnemonic(self):
        "returns a string that represents this class as mnemonic, e.g. to use it in file names"
        return "F2cond{}".format(FEATURE_COUNT)

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Downloads or calculates new data from 'first' sample up to and including 'last'.
        """
        ohlc_first = first - pd.Timedelta(self.history(), unit="T")
        df = self.ohlcv.get_data(base, ohlc_first, last)
        return cal_features(df)


if __name__ == "__main__":
    for base in Env.usage.bases:
        tf = CondensedFeatures(base, path=Env.data_path)
        tfv = tf.calc_features_and_targets()
        cf.report_setsize(f"{base} tf.minute_data", tf.minute_data)
        cf.report_setsize(f"tf.vec {base} tf.vec", tf.vec)
        cf.report_setsize(f"tfv {base} tf.vec", tfv)
"""
    if True:
        cdf = ccd.load_asset_dataframe("btc", path=Env.data_path, limit=HMWF+10)
        features = cal_features(cdf)
        logger.debug(str(cdf.head(5)))
        logger.debug(str(cdf.tail(5)))
        logger.debug(str(features.head(5)))
        logger.debug(str(features.tail(5)))
    else:
        cols = [
            COL_PREFIX[ix] + ext
            for regr_only, offset, minutes, ext in REGRESSION_KPI
            for ix in range(4) if ((ix < 2) or (not regr_only))]
        cols = cols + [COL_PREFIX[4] + ext for (svol, lvol, ext) in VOL_KPI]
        logger.debug(str(cols))
"""
