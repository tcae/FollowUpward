# import pandas as pd
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
UNDETERMINED = "undetermined"
TARGETS = {UNDETERMINED: -2, BUY: 1, HOLD: 0, SELL: -1}  # dict with int encoding of target labels
TARGET_NAMES = {-2: UNDETERMINED, 1: BUY, 0: HOLD, -1: SELL}  # dict with int encoding of targets


def __trade_signals(close):
    """ Receives a numpy array of close prices starting with the oldest.
        Returns a numpy array of signals.
    """
    if close.ndim > 1:
        print("unexpected close array dimension {close.ndim}")
    maxdistance = min(close.size, 8*60)
    dt = np.dtype([("target", "i4"), ("nowdelta", "f8"), ("fardelta", "f8"),
                   ("flag", "bool"), ("back", "i4"), ("forward", "i4"), ("holdtrace", "i4")])
    notes = np.zeros(close.size, dt)
    notes["target"] = TARGETS[UNDETERMINED]
    notes["back"] = TARGETS[UNDETERMINED]
    notes["forward"] = TARGETS[UNDETERMINED]
    notes["holdtrace"] = TARGETS[UNDETERMINED]
    notes["nowdelta"][0] = 0.
    subnotes = notes[1:]
    subnotes["nowdelta"] = (close[1:] - close[:-1]) / close[:-1]
    notes[-1]["fardelta"] = notes[-1]["nowdelta"]
    notes["target"][0] = TARGETS[HOLD]
    notes["back"][-1] = TARGETS[HOLD]  # cannot be overwritten due to missing future prices
    notes["forward"][0] = TARGETS[HOLD]  # may be overwritten by SELL or BUY

    for bix in range(1, maxdistance):
        eix = close.size - bix  # bix = begin index,  eix = end index
        # slicing requires eix+1 because end of slice is excluded
        subnotes = notes[1:eix+1]
        subnotes["fardelta"] = (close[bix:] - close[:eix]) / close[:eix]

        subnotes["flag"] = (
            (subnotes["fardelta"] < SELL_THRESHOLD) & (subnotes["nowdelta"] < 0.) &
            (subnotes["target"] == TARGETS[UNDETERMINED]))
        subnotes["target"][subnotes["flag"]] = TARGETS[SELL]
        subnotes["back"][subnotes["flag"]] = TARGETS[SELL]
        subnotes["forward"][subnotes["flag"]] = TARGETS[SELL]

        subnotes["flag"] = (
            (subnotes["fardelta"] > BUY_THRESHOLD) & (subnotes["nowdelta"] > 0.) &
            (subnotes["target"] == TARGETS[UNDETERMINED]))
        subnotes["target"][subnotes["flag"]] = TARGETS[BUY]
        subnotes["back"][subnotes["flag"]] = TARGETS[BUY]
        subnotes["forward"][subnotes["flag"]] = TARGETS[BUY]

        subnotes["flag"] = (
            (((subnotes["fardelta"] < 0.) & (subnotes["nowdelta"] > 0.)) |
             ((subnotes["fardelta"] > 0.) & (subnotes["nowdelta"] < 0.))) &
            (subnotes["target"] == TARGETS[UNDETERMINED]))
        subnotes["target"][subnotes["flag"]] = TARGETS[HOLD]

    for eix in range(close.size - 1, close.size - 1 - maxdistance, -1):
        subnotes = notes[1:eix+1]
        one_earlier = notes[:eix]
        subnotes["flag"] = (one_earlier["back"] == TARGETS[UNDETERMINED])
        if ~np.any(subnotes["flag"]):
            break
        one_earlier["back"][subnotes["flag"]] = subnotes["back"][subnotes["flag"]]

    for bix in range(1, maxdistance):
        subnotes = notes[bix:]
        one_earlier = notes[bix-1:-1]
        subnotes["flag"] = (subnotes["forward"] == TARGETS[UNDETERMINED])
        if ~np.any(subnotes["flag"]):
            break
        subnotes["forward"][subnotes["flag"]] = one_earlier["forward"][subnotes["flag"]]

    notes["flag"] = (notes["target"] == TARGETS[UNDETERMINED])
    notes["target"][notes["flag"]] = TARGETS[HOLD]

    notes["flag"] = (notes["back"] == notes["forward"]) & (notes["target"] == TARGETS[HOLD])
    notes["target"][notes["flag"]] = notes["back"][notes["flag"]]
    return notes["target"]


def crypto_trade_targets(df):
    """ df is a pandas DataFrame with at least a 'close' column that
        represents the close price.

        Adds target columns that signal the ideal trade classes (SELL, BUY, HOLD).

        The modified DataFrame is returned.
    """
    trade_targets = __trade_signals(df.close.values)
    df["target"] = trade_targets
    return df


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


if __name__ == "__main__":
    if True:
        close = np.array([1., 1.02, 1.015, 1.016, 1.03, 1.024, 1.025, 1.026, 1.025, 1.024,
                          1.023, 1.022, 1.021, 1.02, 1.016, 1.014, 1.012, 1.022], np.dtype(np.float))
        trade_targets = __trade_signals(close)
        print(trade_targets)
    else:
        cdf = ccd.load_asset_dataframe("btc", path=Env.data_path, limit=100)
        trade_targets = __trade_signals(cdf["close"].values)
        cdf["target"] = trade_targets
        print(cdf.head(5))
        print(cdf.tail(5))
