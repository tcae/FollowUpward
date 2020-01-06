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
PRELIM = "preliminary hold"
TARGETS = {UNDETERMINED: -2, BUY: 1, HOLD: 0, SELL: -1}  # dict with int encoding of target labels
TARGET_NAMES = {-2: UNDETERMINED, 1: BUY, 0: HOLD, -1: SELL}  # dict with int encoding of targets


def __signal_spike_cleanup(df, lix, cix):
    """ clean up of trade signal spikes that don't pay off

        df is a pandas DataFrame. It shall contain a targets column
        at index lix (= label index) and a close price column at
        index cix (= close index).
    """
    closeatsell = dict()
    closeatbuy = dict()
    for tix in range(len(df)):  # tix = time index
        this_close = df.iat[tix, cix]
        if df.iat[tix, lix] == TARGETS[BUY]:
            # smooth sell peaks if beneficial
            closeatbuy[tix] = this_close
            for uix in closeatsell:
                sell_buy = -2 * (FEE + TRADE_SLIP)
                holdgain = this_close - closeatsell[uix]
                if sell_buy < holdgain:
                    # if fee loss more than dip loss/gain then smoothing
                    df.iat[uix, lix] = TARGETS[HOLD]
            closeatsell = dict()
        if df.iat[tix, lix] == TARGETS[SELL]:
            # smooth buy peaks if beneficial
            closeatsell[tix] = this_close
            for uix in closeatbuy:
                buy_sell = -2 * (FEE + TRADE_SLIP) + this_close - closeatbuy[uix]
                if buy_sell < 0:
                    # if fee loss more than dip loss/gain then smoothing
                    df.iat[uix, lix] = TARGETS[HOLD]
            closeatbuy = dict()


def __close_hold_gaps_of_consequtive_equal_signals(df, lix):
    """ df is a pandas DataFrame and shall contain a targets column
        at index lix (= label index)
    """
    last_signal = TARGETS[HOLD]
    last_ix = 0
    for tix in range(len(df)):  # tix = time index
        this_signal = df.iat[tix, lix]
        if last_signal == TARGETS[HOLD]:
            if this_signal != TARGETS[HOLD]:
                last_signal = this_signal
                last_ix = tix
        else:  # last_signal != TARGETS[HOLD]:
            if this_signal == TARGETS[HOLD]:
                continue
            elif this_signal == last_signal:  # fill gap
                df.iloc[last_ix:tix, lix] = this_signal
            # opposite signals => no gap to fill or
            # equal trade signals and gap just filled
            last_ix = tix
            last_signal = this_signal


def __trade_signals_thresholds(df, lix, cix):
    """ TARGET label logic:
        - SELL when price reaches in later steps SELL_THRESHOLD before BUY_THRESHOLD
            AND it is currently decreasing
        - BUY when price reaches in later steps BUY_THRESHOLD before SELL_THRESHOLD
            AND it is currently increasing
        - HOLD when price reaches in later steps SELL_THRESHOLD before BUY_THRESHOLD
            BUT it is currently increasing
        - HOLD when price reaches in later steps BUY_THRESHOLD before SELL_THRESHOLD
            BUT it is currently decreasing
        - HOLD when price is currently increasing but in later steps price falls below current price
        - HOLD when price is currently decreasing but in later steps price gains above current price
    """
    unresolved_delta = dict()  # key = tix, value = current price delta
    last_close = df.iat[0, cix]
    for tix in range(1, len(df), 1):  # tix = time index
        urs = dict()
        this_close = df.iat[tix, cix]
        delta = (this_close - last_close) / last_close
        urs[tix] = delta
        for uix in unresolved_delta:
            last_close = df.iat[uix, cix]
            delta = (this_close - last_close) / last_close
            if delta < SELL_THRESHOLD:
                if unresolved_delta[uix] < 0.:  # price decreases
                    df.iat[uix, lix] = TARGETS[SELL]
                # else stay with df.iat[uix, lix] = TARGETS[HOLD]
            elif delta > BUY_THRESHOLD:
                if unresolved_delta[uix] > 0.:  # price increases
                    df.iat[uix, lix] = TARGETS[BUY]
                # else stay with df.iat[uix, lix] = TARGETS[HOLD]
            elif (delta > 0) and (unresolved_delta[uix] < 0.):
                pass
                # stay with df.iat[uix, lix] = TARGETS[HOLD]
            elif (delta < 0) and (unresolved_delta[uix] > 0.):
                pass
                # stay with df.iat[uix, lix] = TARGETS[HOLD]
            else:
                # keep monitoring
                urs[uix] = unresolved_delta[uix]
        last_close = this_close
        unresolved_delta = urs


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
        # subnotes["fardelta"] = (close[bix:] - close[:eix]) / close[:eix]

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
        # subnotes["fardelta"][subnotes["flag"]] = 0.

        one_earlier = notes[:eix]
        one_earlier["fardelta"] = subnotes["fardelta"] + subnotes["nowdelta"]

    for eix in range(close.size - 1, close.size - 1 - maxdistance, -1):
        subnotes = notes[1:eix+1]
        one_earlier = notes[:eix]
        subnotes["flag"] = (one_earlier["back"] == TARGETS[UNDETERMINED])
        # if ~np.any(subnotes["flag"]):
        #     break
        one_earlier["back"][subnotes["flag"]] = subnotes["back"][subnotes["flag"]]

    for bix in range(1, maxdistance):
        subnotes = notes[bix:]
        one_earlier = notes[bix-1:-1]
        subnotes["flag"] = (subnotes["forward"] == TARGETS[UNDETERMINED])
        # if ~np.any(subnotes["flag"]):
        #     break
        subnotes["forward"][subnotes["flag"]] = one_earlier["forward"][subnotes["flag"]]

    notes["flag"] = (notes["target"] == TARGETS[UNDETERMINED])
    notes["target"][notes["flag"]] = TARGETS[HOLD]

    any_undetermined = np.any(notes["back"] == TARGETS[UNDETERMINED])
    if any_undetermined:
        undetermined = np.nonzero(notes["back"] == TARGETS[UNDETERMINED])
        print("WARNING: found {} undetermined in back trace for close vector with {} elements".format(
              len(undetermined), close.size))
        print(f"indices of elements that are undetermined:")
        print(undetermined)
    any_undetermined = np.any(notes["forward"] == TARGETS[UNDETERMINED])
    if any_undetermined:
        undetermined = np.nonzero(notes["forward"] == TARGETS[UNDETERMINED])
        print("WARNING: found {} undetermined in forward trace for close vector with {} elements".format(
              len(undetermined), close.size))
        print(f"indices of elements that are undetermined:")
        print(undetermined)
    any_undetermined = np.any(notes["target"] == TARGETS[UNDETERMINED])
    if any_undetermined:
        undetermined = np.nonzero(notes["target"] == TARGETS[UNDETERMINED])
        print("WARNING: found {} undetermined in targets for close vector with {} elements".format(
              len(undetermined), close.size))
        print(f"indices of elements that are undetermined:")
        print(undetermined)

    notes["flag"] = (notes["back"] == notes["forward"]) & (notes["target"] == TARGETS[HOLD])
    notes["target"][notes["flag"]] = notes["back"][notes["flag"]]
    return notes["target"]


def crypto_trade_targets(df):
    """ df is a pandas DataFrame with at least a 'close' column that
        represents the close price.

        Adds target columns that signal the ideal trade classes (SELL, BUY, HOLD).

        The modified DataFrame is returned.

        Target label logic see TargetEvaluation.next_sample().
        Afterwards clean up of trade signal spikes that don't pay off.
        Finally close HOLD gaps between consequitive sell or buy signals.
    """
    df["target"] = TARGETS[HOLD]
    lix = df.columns.get_loc("target")
    cix = df.columns.get_loc("close")
    __trade_signals_thresholds(df, lix, cix)
    # df["target_thresholds"] = df["target"]
    __signal_spike_cleanup(df, lix, cix)
    # df["target_spike_cleanup"] = df["target"]
    __close_hold_gaps_of_consequtive_equal_signals(df, lix)
    trade_targets = __trade_signals(df.iloc[:, cix].values)
    df["target2"] = trade_targets
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
        cdf["target2"] = trade_targets
        print(cdf.head(5))
        print(cdf.tail(5))
