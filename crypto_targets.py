# import pandas as pd

FEE = 1/1000  # in per mille, transaction fee is 0.1%
TRADE_SLIP = 0  # 1/1000  # in per mille, 0.1% trade slip
BUY_THRESHOLD = 10/1000  # in per mille
SELL_THRESHOLD = -5/1000  # in per mille
HOLD = "-"
BUY = "buy"
SELL = "sell"
TARGETS = {BUY: 0, HOLD: 1, SELL: 2}  # dict with int encoding of target labels
TARGET_NAMES = {1: HOLD, 0: BUY, 2: SELL}  # dict with int encoding of targets


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
    df["target_thresholds"] = df["target"]
    __signal_spike_cleanup(df, lix, cix)
    df["target_spike_cleanup"] = df["target"]
    __close_hold_gaps_of_consequtive_equal_signals(df, lix)
    return df
