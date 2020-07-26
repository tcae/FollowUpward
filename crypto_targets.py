import pandas as pd
import logging
import numpy as np
from sklearn.utils import Bunch
from env_config import Env
import env_config as env
import cached_crypto_data as ccd
import condensed_features as cof

logger = logging.getLogger(__name__)

FEE = 1/1000  # in per mille, transaction fee is 0.1%
TRADE_SLIP = 0  # 1/1000  # in per mille, 0.1% trade slip
HOLD = "hold"
BUY = "buy"
SELL = "sell"
UNDETERMINED = -2  # don't consider UNDETERMINED a valid target
TARGETS = {HOLD: 0, BUY: 1, SELL: 2}  # dict with int encoding of target labels
TARGET_CLASS_COUNT = len(TARGETS)
TARGET_NAMES = {0: HOLD, 1: BUY, 2: SELL}  # dict with int encoding of targets


def max_look_back_minutes():
    return 30  # max look back in minutes


def report_setsize(setname, df):
    logger.info(f"{str_setsize(df)} on {setname}")


def str_setsize(df):
    hc = len(df[df.target == TARGETS[HOLD]])
    sc = len(df[df.target == TARGETS[SELL]])
    bc = len(df[df.target == TARGETS[BUY]])
    sumc = hc + sc + bc
    return f"buy={bc} sell={sc} hold={hc} sum={sumc}; total={len(df)} diff={len(df)-sumc}"


def to_scikitlearn(df, np_data=None, descr=None):
    """Load and return the crypto dataset (classification).
    """

    fn_list = list(df.keys())
    if "target" in fn_list:
        fn_list.remove("target")
        target = df["target"].values
    else:
        target = None
    if "close" in fn_list:
        fn_list.remove("close")
    if np_data is None:
        # data = df[fn_list].to_numpy(dtype=float) # incompatible with pandas 0.19.2
        data = df[fn_list].values
    else:
        data = np_data
        # target = df["target"].to_numpy(dtype=float) # incompatible with pandas 0.19.2
        # tics = df.index.to_numpy(dtype=np.datetime64) # incompatible with pandas 0.19.2
    tics = df.index.values  # compatible with pandas 0.19.2
    feature_names = np.array(fn_list)
    target_names = np.array(TARGETS.keys())
    if descr is None:
        descr = "missing description"

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 tics=tics,
                 descr=descr,
                 feature_names=feature_names)


def trade_signals(close):
    """ Receives a numpy array of close prices starting with the oldest.
        Returns a numpy array of signals.

        algorithm:

        - SELL if fardelta < sell threshold within max distance reach and minute delta negative.
        - BUY if fardelta > buy threshold within max distance reach and minute delta positive.
        - HOLD if fardelta and minute delta have different leading sign.
        - else HOLD.
    """
    buy_threshold = float(10/1000)
    sell_threshold = float(-5/1000)
    if close.ndim > 1:
        logger.warning("unexpected close array dimension {close.ndim}")
    maxdistance = min(close.size, max_look_back_minutes())
    dt = np.dtype([("target", "i4"), ("nowdelta", "f8"), ("fardelta", "f8"),
                   ("flag", "bool")])
    notes = np.zeros(close.size, dt)
    notes["target"] = UNDETERMINED
    notes[0]["nowdelta"] = 0.
    subnotes = notes[1:]
    subnotes["nowdelta"] = (close[1:] - close[:-1]) / close[:-1]
    notes[-1]["fardelta"] = notes[-1]["nowdelta"]
    notes[0]["target"] = TARGETS[HOLD]

    for bix in range(1, maxdistance):
        eix = close.size - bix  # bix = begin index,  eix = end index
        # slicing requires eix+1 because end of slice is excluded
        subnotes = notes[1:eix+1]
        subnotes["fardelta"] = (close[bix:] - close[:eix]) / close[bix:]

        subnotes["flag"] = (
            (subnotes["fardelta"] < sell_threshold) & (subnotes["nowdelta"] < 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[SELL]

        subnotes["flag"] = (
            (subnotes["fardelta"] < sell_threshold) & (subnotes["nowdelta"] > 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[HOLD]

        subnotes["flag"] = (
            (subnotes["fardelta"] > buy_threshold) & (subnotes["nowdelta"] > 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[BUY]

        subnotes["flag"] = (
            (subnotes["fardelta"] > buy_threshold) & (subnotes["nowdelta"] < 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[HOLD]
        # print(f"subnotes {bix}: \n{subnotes}")
        # print(f"notes {bix}: \n{[(ix, notes[ix]) for ix, _ in enumerate(notes)]}")

    # assign all UNDETERMINED targets a HOLD signal
    notes["flag"] = (notes["target"] == UNDETERMINED)
    notes["target"][notes["flag"]] = TARGETS[HOLD]

    assert ~(notes["target"] == UNDETERMINED).any()
    return notes["target"]


def invalidate_all_later_entries(ix_list: list, ix: int):
    while (len(ix_list) > 0) and (ix_list[0] >= ix):
        del ix_list[0]
    return ix_list


def target_signals_gains(close):
    """ Receives a numpy array of close prices starting with the oldest.
        Returns a numpy array of signals.

        algorithm:

        - SELL if fardelta < sell threshold within max distance reach and minute delta negative.
        - BUY if fardelta > buy threshold within max distance reach and minute delta positive.
        - HOLD if fardelta and minute delta have different leading sign.
        - else HOLD.
    """
    buy_threshold = float(10/1000)
    sell_threshold = float(-5/1000)
    peak_list = list()
    dip_list = list()
    assert close.ndim == 1, f"unexpected close array dimension {close.ndim}"
    maxdistance = min(close.size, max_look_back_minutes())
    dt = np.dtype([("target", "i4"), ("gain", "f8")])
    notes = np.zeros(close.size, dt)
    notes[-1]["target"] = TARGETS[HOLD]
    notes[-1]["gain"] = 0.0
    notes[:-1]["gain"] = (close[1:] - close[:-1]) / close[:-1]  # gain of next minute in the future
    gain_ix = close.size - 1
    loss_ix = close.size - 1

    ix = close.size - 2
    while ix >= 0:
        if notes[ix]["gain"] <= 0:
            if gain_ix <= loss_ix:  # establish new loss sequence
                loss_ix = ix + 1
                dip_list.append(loss_ix)  # add local dip
            sig_loss_ix = dip_list[0]
            while (sig_loss_ix > loss_ix) and \
                  (((sig_loss_ix - ix) > maxdistance) or (close[loss_ix] < close[sig_loss_ix])):
                del dip_list[0]
                sig_loss_ix = dip_list[0]
            farloss = (close[sig_loss_ix] - close[ix]) / close[ix]
            notes[ix]["gain"] = farloss
            if farloss < sell_threshold:
                notes[ix]["target"] = TARGETS[SELL]
                peak_list = invalidate_all_later_entries(peak_list, sig_loss_ix)
            else:
                notes[ix]["target"] = TARGETS[HOLD]
        else:  # notes[ix]["gain"] > 0
            if gain_ix >= loss_ix:  # establish new gain sequence
                gain_ix = ix + 1
                peak_list.append(gain_ix)  # add local dip
            sig_gain_ix = peak_list[0]
            while (sig_gain_ix > gain_ix) and \
                  (((sig_gain_ix - ix) > maxdistance) or (close[gain_ix] > close[sig_gain_ix])):
                del peak_list[0]
                sig_gain_ix = peak_list[0]
            fargain = (close[sig_gain_ix] - close[ix]) / close[ix]
            notes[ix]["gain"] = fargain
            if fargain > buy_threshold:
                notes[ix]["target"] = TARGETS[BUY]
                dip_list = invalidate_all_later_entries(dip_list, sig_gain_ix)
            else:
                notes[ix]["target"] = TARGETS[HOLD]
        ix -= 1
    return notes


def crypto_trade_targets(df):
    """ df is a pandas DataFrame with at least a 'close' column that
        represents the close price.

        The trade_targets numpy array is returned.
    """
    trade_targets = trade_signals(df.close.values)
    # df.loc[:, "target"] = trade_targets
    return trade_targets


def trade_target_performance(target_df):
    """ Calculates the performance of 'target' trade signals.

        'target_df' shall be a pandas DataFrame with at least
        the columns 'close' and 'target'.
    """
    # logger.debug(f"calculate target_performance")
    perf = 0.
    ta_holding = False
    lix = target_df.columns.get_loc("target")  # lix == label index
    assert lix > 0, f"cannot find column {lix}"
    cix = target_df.columns.get_loc("close")  # cix == close index
    assert cix > 0, f"cannot find column {cix}"

    assert target_df.index.is_unique, "unexpected not unique index"
    last = target_df.iat[0, cix]
    for tix in range(len(target_df)):  # tix = time index
        this = target_df.iat[tix, cix]
        tix_perf = ((this - last) / last)
        last = this
        signal = target_df.iat[tix, lix]
        if ta_holding:
            perf += tix_perf
        if (signal == TARGETS[BUY]) and (not ta_holding):
            perf -= FEE  # assuming that the buy will take place in the coming minute
            ta_holding = True
        if (signal == TARGETS[SELL]) and ta_holding:
            perf -= FEE  # assuming that the sell will take place in the coming minute
            ta_holding = False
    return perf


class Targets(ccd.CryptoData):

    def __init__(self, ohlcv: ccd.Ohlcv):
        self.ohlcv = ohlcv
        super().__init__()

    def keys(self):
        "returns the list of element keys"
        return ["target", "gain"]

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        return 1

    def mnemonic(self):
        """
            returns a string that represents this class as mnemonic, e.g. to use it in file names.
        """
        return "TargetGain10up5low30min"

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Downloads or calculates new data from 'first' sample up to and including 'last'.
            This is the core method to be implemented by subclasses.
        """
        ohlc_first = first - pd.Timedelta(self.history(), unit="T")
        df = self.ohlcv.get_data(base, ohlc_first, last)
        signals_prices = target_signals_gains(df["close"].values)
        df = pd.DataFrame(data={"target": signals_prices["target"], "gain": signals_prices["gain"]}, index=df.index)
        tdf = df.loc[first:]
        return tdf


def regression_targets(ohlcvp_df, minutes, buy_threshold, sell_threshold):
    """ Returns the rolling gradient and last price of a 1D linear regression over 'minutes' future values
        normalized with the first pivot price value to express the gradient as a fraction of the pivot level.
        Using future gradient values is the main difference to cof.regression_features.
        It is assumed (and not checked) that ovphlcvp contains values of minute frequency without gaps.

        Returns a dataframe with the columns 'gain' as normalized gain per minute and 'target' as class indicator.
    """
    df = pd.DataFrame(index=ohlcvp_df.index)
    shift_minutes = -round(minutes / 2)
    df["grad"], df["lvl"] = cof.regression_line(ohlcvp_df["pivot"].to_numpy(), window=minutes)
    df["gain"] = df["grad"].shift(shift_minutes) / ohlcvp_df["pivot"]
    # shift = look 'minutes' into future values incl current minute value
    df["target"] = TARGETS[HOLD]
    df.loc[df["gain"] > buy_threshold, "target"] = TARGETS[BUY]
    df.loc[df["gain"] < sell_threshold, "target"] = TARGETS[SELL]
    return df


class TargetGrad30m1pct(Targets):

    def target_dict(self):
        """ Shall return a dict of target categories that can be used as columns for prediction data.
            The dict values are the corresponding target values.
        """
        return TARGETS

    def history(self):
        """ Returns the number of required history sample minutes excluding the minute under consideration.
            It is negative for targets because future data is needed.
        """
        return -29

    def mnemonic(self):
        """
            returns a string that represents this class as mnemonic, e.g. to use it in file names.
        """
        return "TargetGrad30m1pct"

    def keys(self):
        "returns the list of element keys"
        return ["target"]

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp):
        """ New data from 'first' sample up to and including 'last'.
            The .
        """
        ohlcv_last = last - pd.Timedelta(self.history(), unit="T")
        ohlcv_df = self.ohlcv.get_data(base, first, ohlcv_last)
        df = regression_targets(ohlcv_df, -self.history()+1, buy_threshold=0.01/30, sell_threshold=0.0)
        return df

    def new_data_test(self):
        print(f"history: {self.history()}")
        cols = ["open", "high", "low", "close", "volume"]
        dat = [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 2],
            [2, 2, 1, 1, 3],
            [3, 3, 2, 2, 4],
            [4, 4, 3, 3, 5],
            [1, 1, 0, 0, 2]]
        # [6, 6, 5, 5, 6]]
        ohlcv_df = pd.DataFrame(
            data=dat,
            columns=cols, index=pd.date_range("2012-10-08 18:15:05", periods=6, freq="T", tz="Europe/Amsterdam"))
        ohlcv_df["pivot"] = ccd.calc_pivot(ohlcv_df)
        print(ohlcv_df)
        df = regression_targets(ohlcv_df, 3, buy_threshold=0.5, sell_threshold=0.0)
        print(df)


class Target10up5low30min(Targets):

    def target_dict(self):
        """ Shall return a dict of target categories that can be used as columns for prediction data.
            The dict values are the corresponding target values.
        """
        return TARGETS

    def mnemonic(self):
        """
            returns a string that represents this class as mnemonic, e.g. to use it in file names.
        """
        return "Target10up5low30min"

    def keys(self):
        "returns the list of element keys"
        return ["target"]


class Gain10up5low30min(Targets):

    def mnemonic(self):
        """
            returns a string that represents this class as mnemonic, e.g. to use it in file names.
        """
        return "Gain10up5low30min"

    def keys(self):
        "returns the list of element keys"
        return ["gain"]


if __name__ == "__main__":
    env.test_mode()
    if False:
        close = np.array([1., 1.02, 1.015, 1.016, 1.03, 1.024, 1.025, 1.026, 1.025, 1.024,
                          1.035, 1.022, 1.021, 1.02, 1.016, 1.014, 1.012, 1.024], np.dtype(np.float))
        ratio = [(close[ix]-close[ix-1])/close[ix-1] if ix > 0 else 0
                 for ix, p in enumerate(close)]
        perc = ["{:.1%}".format((close[ix+1]-close[ix])/close[ix]) if ix < (len(close)-1) else "{:.1%}".format(0)
                for ix, p in enumerate(close)]
        trade_targets = trade_signals(close)
        signals_prices = target_signals_gains(close)
        tuples = list(zip(close, perc, [TARGET_NAMES[ix] for ix in trade_targets],
                      [TARGET_NAMES[tgt] for tgt in signals_prices["target"]],
                      ["{:.1%}".format(gain) for gain in signals_prices["gain"]]))
        [print(ix, tup) for ix, tup in enumerate(tuples)]

        ohlcv = ccd.Ohlcv()
        targets = Targets(ohlcv)
        tdf = targets.get_data(
            Env.bases[0], pd.Timestamp("2018-01-20 23:59:00+00:00"),
            pd.Timestamp("2018-01-31 23:59:00+00:00"))
        # tdf = targets.load_data(Env.bases[0])
        ccd.dfdescribe("Targets", tdf)

        targets = Target10up5low30min(ohlcv)
        tdf = targets.get_data(
            Env.bases[0], pd.Timestamp("2018-01-20 23:59:00+00:00"),
            pd.Timestamp("2018-01-31 23:59:00+00:00"))
        # tdf = targets.load_data(Env.bases[0])
        ccd.dfdescribe("Target10up5low30min", tdf)

        targets = Gain10up5low30min(ohlcv)
        tdf = targets.get_data(
            Env.bases[0], pd.Timestamp("2018-01-20 23:59:00+00:00"),
            pd.Timestamp("2018-01-31 23:59:00+00:00"))

        # tdf = targets.load_data(Env.bases[0])
        ccd.dfdescribe("Gain10up5low30min", tdf)

        # logger.debug(list(zip(close, perc, [TARGET_NAMES[ix] for ix in trade_targets])))

    else:
        TargetGrad30m1pct(ccd.Ohlcv()).new_data_test()
