import pandas as pd
import numpy as np
from env_config import Env
import cached_crypto_data as ccd


FEE = 1/1000  # in per mille, transaction fee is 0.1%
TRADE_SLIP = 0  # 1/1000  # in per mille, 0.1% trade slip
BUY_THRESHOLD = float(10/1000)  # in per mille
SELL_THRESHOLD = float(-5/1000)  # in per mille
HOLD = "hold"
BUY = "buy"
SELL = "sell"
UNDETERMINED = -2  # don't consider UNDETERMINED a valid target
TARGETS = {HOLD: 0, BUY: 1, SELL: 2}  # dict with int encoding of target labels
TARGET_CLASS_COUNT = len(TARGETS)
TARGET_NAMES = {0: HOLD, 1: BUY, 2: SELL}  # dict with int encoding of targets


def trade_signal_history():
    return 30  # max look back: 8 hours


def trade_signals(close):
    """ Receives a numpy array of close prices starting with the oldest.
        Returns a numpy array of signals.

        algorithm:

        - SELL if fardelta < sell threshold within max distance reach and minute delta negative.
        - BUY if fardelta > buy threshold within max distance reach and minute delta positive.
        - HOLD if fardelta and minute delta have different leading sign.
        - else HOLD.
    """
    if close.ndim > 1:
        print("unexpected close array dimension {close.ndim}")
    maxdistance = min(close.size, 8*60)
    dt = np.dtype([("target", "i4"), ("nowdelta", "f8"), ("fardelta", "f8"),
                   ("flag", "bool")])
    notes = np.zeros(close.size, dt)
    notes["target"] = UNDETERMINED
    notes["nowdelta"][0] = 0.
    subnotes = notes[1:]
    subnotes["nowdelta"] = (close[1:] - close[:-1]) / close[:-1]
    notes[-1]["fardelta"] = notes[-1]["nowdelta"]
    notes["target"][0] = TARGETS[HOLD]

    for bix in range(1, maxdistance):
        eix = close.size - bix  # bix = begin index,  eix = end index
        # slicing requires eix+1 because end of slice is excluded
        subnotes = notes[1:eix+1]
        subnotes["fardelta"] = (close[bix:] - close[:eix]) / close[:eix]

        subnotes["flag"] = (
            (subnotes["fardelta"] < SELL_THRESHOLD) & (subnotes["nowdelta"] < 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[SELL]

        subnotes["flag"] = (
            (subnotes["fardelta"] < SELL_THRESHOLD) & (subnotes["nowdelta"] > 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[HOLD]

        subnotes["flag"] = (
            (subnotes["fardelta"] > BUY_THRESHOLD) & (subnotes["nowdelta"] > 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[BUY]

        subnotes["flag"] = (
            (subnotes["fardelta"] > BUY_THRESHOLD) & (subnotes["nowdelta"] < 0.) &
            (subnotes["target"] == UNDETERMINED))
        subnotes["target"][subnotes["flag"]] = TARGETS[HOLD]

    # assign all UNDETERMINED targets a HOLD signal
    notes["flag"] = (notes["target"] == UNDETERMINED)
    notes["target"][notes["flag"]] = TARGETS[HOLD]

    assert ~(notes["target"] == UNDETERMINED).any()
    return notes["target"]


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
    # print(f"{datetime.now()}: calculate target_performance")
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

    def target_dict(self):
        """ Shall return a dict of target categories that can be used as columns for prediction data.
            The dict values are the corresponding target values.
        """
        pass

    def get_data_old(self, base: str, last: pd.Timestamp, minutes: int, use_cache=True):
        """ Loads and downloads/calculates new data for 'minutes' samples up to and including last.
            For Targets enforce to use_cache.
        """
        return super().get_data_old(base, last, minutes, use_cache=True)

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Loads and downloads/calculates new data from 'first' sample up to and including 'last'.
            For Targets enforce to use_cache.
        """
        return super().get_data(base, first, last, use_cache=True)


class T10up5low30min(Targets):

    def __init__(self, ohlcv: ccd.Ohlcv):
        self.ohlcv = ohlcv
        super().__init__()

    def target_dict(self):
        return TARGETS

    def history(self):
        """ Returns the number of history sample minutes
            excluding the minute under consideration required to calculate a new sample data.
        """
        return trade_signal_history()

    def keys(self):
        "returns the list of element keys"
        return ["target"]

    def mnemonic(self):
        "returns a string that represents this class as mnemonic, e.g. to use it in file names"
        return "T10up5low30min"

    def new_data_old(self, base: str, last: pd.Timestamp, minutes: int, use_cache=True):
        """ Downloads or calculates new data for 'minutes' samples up to and including last.
            This is the core method to be implemented by subclasses.
        """
        df = self.ohlcv.get_data_old(base, last, minutes + self.history())
        trade_targets = trade_signals(df["close"].values)
        df["target"] = trade_targets
        tdf = df.iloc[-minutes:]
        tdf = tdf.loc[:, self.keys()]
        return tdf

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Downloads or calculates new data from 'first' sample up to and including 'last'.
            This is the core method to be implemented by subclasses.
        """
        ohlc_first = first - pd.Timedelta(self.history(), unit="T")
        df = self.ohlcv.get_data(base, ohlc_first, last)
        trade_targets = trade_signals(df["close"].values)
        df["target"] = trade_targets
        tdf = df.loc[first:]
        tdf = tdf.loc[:, self.keys()]
        return tdf


if __name__ == "__main__":
    if True:
        close = np.array([1., 1.02, 1.015, 1.016, 1.03, 1.024, 1.025, 1.026, 1.025, 1.024,
                          1.023, 1.022, 1.021, 1.02, 1.016, 1.014, 1.012, 1.022], np.dtype(np.float))
        trade_targets = trade_signals(close)
        print(trade_targets)
    else:
        cdf = ccd.load_asset_dataframe("btc", path=Env.data_path, limit=100)
        trade_targets = trade_signals(cdf["close"].values)
        cdf["target"] = trade_targets
        print(cdf.head(5))
        print(cdf.tail(5))
