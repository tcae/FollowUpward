#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
# import numpy as np
import pandas as pd
from queue import Queue
from sklearn.utils import Bunch
import numpy as np
import env_config as env
from env_config import Env
import cached_crypto_data as ccd


PICKLE_EXT = ".pydata"  # pickle file extension
JSON_EXT = ".json"  # msgpack file extension
MSG_EXT = ".msg"  # msgpack file extension

FEE = 1/1000  # in per mille, transaction fee is 0.1%
TRADE_SLIP = 0  # 1/1000  # in per mille, 0.1% trade slip
BUY_THRESHOLD = 10/1000  # in per mille
SELL_THRESHOLD = -5/1000  # in per mille
VOL_BASE_PERIOD = "1D"
HOLD = "-"
BUY = "buy"
SELL = "sell"
NA = "not assigned"
TRAIN = "training"
VAL = "validation"
TEST = "test"
TARGETS = {HOLD: 0, BUY: 1, SELL: 2}  # dict with int encoding of target labels
TARGET_NAMES = {0: HOLD, 1: BUY, 2: SELL}  # dict with int encoding of targets
TARGET_KEY = 5
LBL = {NA: 0, TRAIN: -1, VAL: -2, TEST: -3}
MANDATORY_STEPS = 2  # number of steps for the smallest class (in general BUY)


class NoSubsetWarning(Exception):
    pass


def targets_to_features(tfv_ta_df, target_df):
    """Extracts a sample subset with targets and features of a specific time aggregation
    based on given targets. target_df and tfv_ta_df both have to share the same index basis.
    The index of target_df shall be a subset of tfv_ta_df.
    """
    df = tfv_ta_df[tfv_ta_df.index.isin(target_df.index)]
    # check compatibility of target_df.sym with
    d = len(target_df.index.difference(tfv_ta_df.index))
    c = len(df)
    b = len(target_df)
    p = len(tfv_ta_df)
    if d > 0:
        raise NoSubsetWarning(f"subset({b}) with {c}/{d} rows that are/are not in superset({p})")
    return df


def report_setsize(setname, df):
    hc = len(df[df.target == TARGETS[HOLD]])
    sc = len(df[df.target == TARGETS[SELL]])
    bc = len(df[df.target == TARGETS[BUY]])
    tc = hc + sc + bc
    print(f"buy {bc} sell {sc} hold {hc} total {tc} on {setname}")


def str_setsize(df):
    hc = len(df[df.target == TARGETS[HOLD]])
    sc = len(df[df.target == TARGETS[SELL]])
    bc = len(df[df.target == TARGETS[BUY]])
    tc = hc + sc + bc
    return f"buy {bc} sell {sc} hold {hc} total {tc}"


def smallest_dict_key(thisdict):
    smallest_key = 5000
    for k in thisdict:
        if isinstance(k, int):
            if k < smallest_key:
                smallest_key = k
    assert smallest_key != 5000, "no int in dict keys"
    return smallest_key


def to_scikitlearn(df, np_data=None, descr=None):
    """Load and return the crypto dataset (classification).
    """

    fn_list = list(df.keys())
    fn_list.remove("target")
    fn_list.remove("close")
    if np_data is None:
        # data = df[fn_list].to_numpy(dtype=float) # incompatible with pandas 0.19.2
        data = df[fn_list].values
    else:
        data = np_data
        # target = df["target"].to_numpy(dtype=float) # incompatible with pandas 0.19.2
        # tics = df.index.to_numpy(dtype=np.datetime64) # incompatible with pandas 0.19.2
    target = df["target"].values  # compatible with pandas 0.19.2
    close = df["close"].values  # compatible with pandas 0.19.2
    tics = df.index.values  # compatible with pandas 0.19.2
    feature_names = np.array(fn_list)
    target_names = np.array(TARGETS.keys())
    if descr is None:
        descr = "missing description"

    return Bunch(data=data, target=target, close=close,
                 target_names=target_names,
                 tics=tics,
                 descr=descr,
                 feature_names=feature_names)


class TargetsFeatures:
    """Receives a dict of currency pairs with associated minute candle data and
    transforms it into a dict of currency pairs with associated dicts of
    time_aggregations features. The time aggregation is the dict key with one
    special key 'CPC' that provides the summary targets

    Attributes
    ----------
    time_aggregations:
        dict with required time aggregation keys and associated number
        of periods that shall be compiled in a corresponding feature vector
    minute_data:
        currency pair (as keys) dict of input minute data as corresponding
        pandas DataFrame

    To Do
    =====
    buy - sell signals:
        now reduced to first signal
        rolling shall allow a richer buy - sell signalling that are simply mapped on a
        common timeline where the optimization problem is addressed

    >>> read dataframe from file
    >>> concatenate features to full feature vector and write it as file

    abbreviatons and terms
    ======================
    time aggregation - time period for which features are derived, e.g. open, close, high, low.
    In this context different time aggregations are used to low pass filter high frequent
    volatility.
    cpc - currency pair classifier

    """

    def __init__(self, base):
        """Receives the key attributes for feature generation

        aggregations:
            dict with required time aggregation keys and associated number
            of periods that shall be compiled in a corresponding feature vector
        target_key:
            has to be a key of aggregations. Targets are only calculated for that target_key


        """
        assert(env.config_ok())
        self.base = base
        self.quote = Env.quote
        self.minute_data = None
        self.vec = None
        self.target_key = TARGET_KEY

    def load_classifier_features(self):
        """ Loads the historic data temporarily, calculates the features and targets
            and releases the original data afterwards
        """
        try:
            df = ccd.load_asset_dataframe(self.base)
        except env.MissingHistoryData:
            raise
        else:
            self.calc_features_and_targets(df)
            report_setsize(self.base, self.vec)

    def __derive_features(self, df):
        """derived features in relation to price based on the provided
        time aggregated dataframe df with the exception of the derived feature 'delta'
        that is calculated together with targets
        """
        # price deltas in 1/1000
        df["height"] = (df["high"] - df["low"]) / df["close"] * 1000
        df.loc[df["close"] > df["open"],
               "top"] = (df["high"] - df["close"]) / df["close"] * 1000
        df.loc[df["close"] <= df["open"],
               "top"] = (df["high"] - df["open"]) / df["close"] * 1000
        df.loc[df["close"] > df["open"],
               "bottom"] = (df["open"] - df["low"]) / df["close"] * 1000
        df.loc[df["close"] <= df["open"],
               "bottom"] = (df["close"] - df["low"]) / df["close"] * 1000

    def __calc_aggregation(self, minute_df, time_aggregations):
        """Time aggregation through rolling aggregation with the consequence that new data is
        generated every minute and even long time aggregations reflect all minute bumps in their
        features

        in:
            dataframe of minute data of a currency pair
            with the columns: open, high, low, close, volume
        out:
            dict of dataframes of aggregations with features and targets
        """
        tf_aggs = dict()  # feature and target aggregations
        mdf = minute_df  # .copy()
        df = pd.DataFrame(minute_df)  # .copy()
        df["vol"] = (mdf["volume"] - mdf.volume.rolling(VOL_BASE_PERIOD).mean()) \
            / mdf.volume.rolling(VOL_BASE_PERIOD).mean()
        df = df.fillna(value={"vol": 0.000001})
        maxmin = 0
        for time_agg in time_aggregations:
            if isinstance(time_agg, int):
                if (time_agg * time_aggregations[time_agg]) > maxmin:
                    maxmin = time_agg * time_aggregations[time_agg]
        if maxmin > len(df.index):
            raise env.MissingHistoryData("History data has {} samples but should have >= {}".format(
                    len(df.index), maxmin))
        for time_agg in time_aggregations:
            # print(f"{datetime.now()}: time_aggregation {time_agg}")
            if time_agg > 1:
                df = pd.DataFrame()
                df["open"] = mdf.open.shift(time_agg-1)
                df["high"] = mdf.high.rolling(time_agg).max()
                df["low"] = mdf.low.rolling(time_agg).min()
                df["close"] = mdf.close
                df["vol"] = mdf.vol.rolling(time_agg).mean()
            df["delta"] = (mdf.close - mdf.close.shift(time_agg)) / mdf.close.shift(time_agg)
            tf_aggs[time_agg] = df
            self.__derive_features(df)
        return tf_aggs

    def __expand_target_feature_vectors(self, tf_aggs, target_key):
        """Builds a target and feature vector for just the target_key with
        1 minute DHTBV and D*V feature sequences and the remaining D sequences of
        n time steps (tics) as configured in time_aggregations in T units.
        The most important step in expand_target_feature_vectors is
        1) the concatenation of feature vectors per sample to provide a history
        for the classifier
        2) discarding the original currency values that are not used
        as features (except 'close')

        Result:
            a self.vecs dict with the single target_key that is
            referring to a DataFrame with feature vectors as rows. The column name indicates
            the type of feature, i.e. either 'target', 'close' or 'D|H|T|B|V|DV' in case of
            1 minute aggregation or just 'D' for all other aggregations with aggregation+'T_'
            as column prefix
        """
        df = pd.DataFrame(tf_aggs[target_key], columns=["close"])
        skey = smallest_dict_key(tf_aggs)
        for ta in tf_aggs:
            for tics in range(Env.time_aggs[ta]):
                ctitle = str(ta) + "T_" + str(tics) + "_"
                offset = tics*ta
                # now add feature columns according to aggregation
                df[ctitle + "D"] = tf_aggs[ta].delta.shift(offset)
                if ta == skey:  # full set only for smallest aggregation (minute data)
                    df[ctitle + "H"] = tf_aggs[ta].height.shift(offset)
                    df[ctitle + "T"] = tf_aggs[ta].top.shift(offset)
                    df[ctitle + "B"] = tf_aggs[ta].bottom.shift(offset)
                    df[ctitle + "V"] = tf_aggs[ta].vol.shift(offset)
                    df[ctitle + "DV"] = tf_aggs[ta].vol.shift(offset) *\
                        tf_aggs[ta].delta.shift(offset)
        df = df.dropna()
        if df.empty:
            raise env.MissingHistoryData("empty dataframe from expand_target_feature_vectors")
        return df

    class TargetEvaluation:

        def __init__(self, minute_df, start_ix, target_col_ix, close_col_ix):
            self.df = minute_df
            self.lix = target_col_ix  # label ix
            self.cix = close_col_ix  # close ix
            self.unresolved_delta = dict()  # key = tix, value = close
            self.last_close = self.df.iat[0, self.cix]

        def next_sample(self, tix):
            this_close = self.df.iat[tix, self.cix]
            delta = (this_close - self.last_close) / self.last_close
            self.unresolved_delta[tix] = delta
            for uix in self.unresolved_delta:
                if uix == tix:
                    continue
                last_close = self.df.iat[uix, self.cix]
                delta = (this_close - last_close) / last_close
                if delta < SELL_THRESHOLD:
                    if self.unresolved_delta[uix] < 0:
                        self.df.iat[uix, self.lix] = TARGETS[SELL]
                    # else stay with df.iat[uix, lix] = TARGETS[HOLD]
                    del self.unresolved_delta[uix]
                if delta > BUY_THRESHOLD:
                    if self.unresolved_delta[uix] > 0:
                        self.df.iat[uix, self.lix] = TARGETS[BUY]
                    # else stay with df.iat[uix, lix] = TARGETS[HOLD]
                    del self.unresolved_delta[uix]
            self.last_close = this_close

    def __add_targets2(self, time_agg, df):
        df['target2'] = TARGETS[HOLD]
        lix = df.columns.get_loc('target2')
        cix = df.columns.get_loc('close')
        te = {slot: TargetsFeatures.TargetEvaluation(df, slot, lix, cix) for slot in range(0, time_agg)}
        closeatsell = closeatbuy = 0
        for tix in range(time_agg, len(df), 1):  # tix = time index
            slot = (tix % time_agg)
            te[slot].next_sample(tix)

        # now correct peaks due to time_agg approach
        closeatsell = dict()
        closeatbuy = dict()
        for tix in range(time_agg, len(df), 1):  # tix = time index
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
                    del closeatsell[uix]
            if df.iat[tix, lix] == TARGETS[SELL]:
                # smooth buy peaks if beneficial
                closeatsell[tix] = this_close
                for uix in closeatbuy:
                    buy_sell = -2 * (FEE + TRADE_SLIP) + this_close - closeatbuy[uix]
                    if buy_sell < 0:
                        # if fee loss more than dip loss/gain then smoothing
                        df.iat[uix, lix] = TARGETS[HOLD]
                    del closeatbuy[uix]

    def __add_targets(self, time_agg, df):
        """ target = achieved if improvement > 1% without intermediate loss of more than 0.2%

            `time_agg` in minutes is used to ignore losses within these time ranges looking forward.
            To achieve that `time_agg` parallel monitoring paths are evaluated that look forward
            in `time_agg` minutes jumps to evaluate gains/losses.
        """
        # print(f"{datetime.now()}: self.add_targets {time_agg}")
        df['target'] = TARGETS[HOLD]
        lix = df.columns.get_loc('target')
        cix = df.columns.get_loc('close')
        win = dict()
        loss = dict()
        lossix = dict()
        winix = dict()
        ixfifo = Queue()  # will hold all sell ix to smooth out if dip sell does no tpay off
        closeatsell = closeatbuy = 0
        lasttarget = dict()
        for slot in range(0, time_agg):
            win[slot] = loss[slot] = 0.
            winix[slot] = lossix[slot] = slot
            lasttarget[slot] = TARGETS[HOLD]
        for tix in range(time_agg, len(df), 1):  # tix = time index
            slot = (tix % time_agg)
            last_close = df.iat[tix - time_agg, cix]
            this_close = df.iat[tix, cix]
            delta = (this_close - last_close) / last_close  # * 1000 no longer in per mille
            if delta < 0:
                if loss[slot] < 0:  # loss monitoring is running
                    loss[slot] += delta
                else:  # first time bar of decrease period
                    lossix[slot] = tix
                    loss[slot] = delta
                if win[slot] > 0:  # win monitoring is running
                    win[slot] += delta
                    if win[slot] < 0:  # reset win monitor because it is below start price
                        win[slot] = 0.
                if loss[slot] < SELL_THRESHOLD:  # reset win monitor -> dip exceeded threshold
                    win[slot] = 0.
                    df.iat[lossix[slot], lix] = lasttarget[slot] = TARGETS[SELL]
                    lossix[slot] += 1  # allow multiple signals if conditions hold => FIX this changes slot!
                    # FIX loss[slot] is not corrected with the index change
                    #  here comes the smooth execution for BUY peaks:
                    if closeatbuy > 0:  # smoothing is active
                        buy_sell = -2 * (FEE + TRADE_SLIP) + this_close - closeatbuy
                        while not ixfifo.empty():
                            smooth_ix = ixfifo.get()
                            if buy_sell < 0:
                                # if fee loss more than dip loss/gain then smoothing
                                df.iat[smooth_ix, lix] = TARGETS[HOLD]
                        closeatbuy = 0
                    #  here comes the smooth preparation for SELL dips:
                    if closeatsell == 0:
                        closeatsell = this_close
                    ixfifo.put(tix)  # prep after execution due to queue reuse
            elif delta > 0:
                if win[slot] > 0:  # win monitoring is running
                    win[slot] += delta
                else:  # first time bar of increase period
                    winix[slot] = tix
                    win[slot] = delta
                if loss[slot] < 0:  # loss monitoring is running
                    loss[slot] += delta
                    if loss[slot] > 0:
                        loss[slot] = 0.  # reset loss monitor -> recovered before sell threshold
                if win[slot] > BUY_THRESHOLD:  # reset win monitor -> dip exceeded threshold
                    loss[slot] = 0.
                    df.iat[winix[slot], lix] = lasttarget[slot] = TARGETS[BUY]
                    winix[slot] += 1  # allow multiple signals if conditions hold => FIX this changes slot!
                    # FIX win[slot] is not corrected with the index change
                    #  here comes the smooth execution for SELL dips:
                    if closeatsell > 0:  # smoothing is active
                        sell_buy = -2 * (FEE + TRADE_SLIP)
                        holdgain = this_close - closeatsell
                        while not ixfifo.empty():
                            smooth_ix = ixfifo.get()
                            if sell_buy < holdgain:
                                # if fee loss more than dip loss/gain then smoothing
                                df.iat[smooth_ix, lix] = TARGETS[HOLD]
                        closeatsell = 0
                    #  here comes the smooth preparation for BUY peaks:
                    if closeatbuy == 0:
                        closeatbuy = this_close
                    ixfifo.put(tix)  # prep after execution due to queue reuse
        # report_setsize("complete set", df)

    def calc_features_and_targets(self, minute_dataframe):
        """Assigns minute_dataframe to attribute *minute_data*.
        Calculates features and assigns them to attribute *vec*.
        If minute_dataframe is None
        then an earlier assigned *minute_data* is used the recalculate features.
        If *minute_data* has no 'target' column then targets are calculated and added to
        *minute_data*.

        Releasing feature data by assigning *vec* None and recalculating those later is a valid
        use case to free up memory temporary.

        minute_dataframe shall have
        the columns: open, high, low, close, volume and timestamps as index
        """

        if minute_dataframe is None:
            if self.minute_data is None:
                raise env.MissingHistoryData("{}–{} target {}min without minute data ({})".format(
                                         self.base, self.quote, self.target_key, self.vec))
        else:
            self.minute_data = minute_dataframe
        if self.minute_data.empty is None:
            self.minute_data = None
            raise env.MissingHistoryData("{}–{} target {}min with empty minute data".format(
                                     self.base, self.quote, self.target_key))
        tf_aggs = self.__calc_aggregation(self.minute_data, Env.time_aggs)
        if "target" not in self.minute_data:
            self.__add_targets(self.target_key, tf_aggs[self.target_key])  # add aggregation targets
            self.__add_targets2(self.target_key, tf_aggs[self.target_key])  # add aggregation targets
            self.minute_data["target"] = tf_aggs[self.target_key]["target"]
            # print("calculating targets")
        else:
            # print("reusing targets")
            pass
        self.vec = self.__expand_target_feature_vectors(tf_aggs, self.target_key)
        if "target" not in self.vec:
            self.vec["target"] = self.minute_data["target"]

        # print(f"{len(self.vec)} feature vectors of {len(self.vec.iloc[0])-2} features")

    def __append_minute_df_with_targets(self, minute_df):
        """ unused?
        """
        self.vec = None
        if "target" not in minute_df:
            raise ValueError("append_minute_df_with_targets: missing target column")
        if self.minute_data is None:
            self.minute_data = minute_df
        else:
            self.minute_data = pd.concat([self.minute_data, minute_df], sort=False)

    def __target_performance(self):
        """ calculates the time aggregation specific performance of target_key
            unused?
        """
        # print(f"{datetime.now()}: calculate target_performance")
        target_df = self.minute_data
        perf = 0.
        ta_holding = False
        col_ix = target_df.columns.get_loc("target")
        assert col_ix > 0, f"did not find column {col_ix} of {self.target_key}"
        close_ix = target_df.columns.get_loc("close")

        assert target_df.index.is_unique, "unexpected not unique index"
        last = target_df.iat[0, close_ix]
        for tix in range(len(target_df)):  # tix = time index
            this = target_df.iat[tix, close_ix]
            tix_perf = ((this - last) / last)  # no longer in per mille * 1000)
            last = this
            signal = target_df.iat[tix, col_ix]
            if ta_holding:
                perf += tix_perf
            if (signal == TARGETS[BUY]) and (not ta_holding):
                perf -= FEE
                ta_holding = True
            if (signal == TARGETS[SELL]) and ta_holding:
                perf -= FEE
                ta_holding = False
        return perf


class TargetsFeatures2(TargetsFeatures):
    """ Alternative feature set. Approach: less features and closer to intuitive performance correlation
        (what would I look at?).

        - regression line percentage increase per hour calculated on last 10d basis
        - regression line percentage increase per hour calculated on last 12h basis
        - regression line percentage increase per hour calculated on last 4h basis
        - regression line percentage increase per hour calculated on last 0.5h basis
        - regression line percentage increase per hour calculated on last 5min basis for last 2x5min periods
        - percentage of last 5min mean volume compared to last 1h mean 5min volume
        - not directly used: SDU = absolute standard deviation of price points above regression line
        - SDU - (current price - regression price) (== up potential) for 12h, 4h, 0.5h, 5min regression
        - not directly used: SDD = absolute standard deviation of price points below regression line
        - SDD + (current price - regression price) (== down potential) for 12h, 4h, 0.5h, 5min regression
    """

    def __init__(self, base):
        """Receives the key attributes for feature generation

        aggregations:
            dict with required time aggregation keys and associated number
            of periods that shall be compiled in a corresponding feature vector
        target_key:
            has to be a key of aggregations. Targets are only calculated for that target_key


        """
        assert(env.config_ok())
        self.base = base
        self.quote = Env.quote
        self.minute_data = None
        self.vec = None
        self.target_key = TARGET_KEY

    def calc_features_and_targets(self, minute_dataframe):
        """Assigns minute_dataframe to attribute *minute_data*.
        Calculates features and assigns them to attribute *vec*.
        If minute_dataframe is None
        then an earlier assigned *minute_data* is used the recalculate features.
        If *minute_data* has no 'target' column then targets are calculated and added to
        *minute_data*.

        Releasing feature data by assigning *vec* None and recalculating those later is a valid
        use case to free up memory temporary.

        minute_dataframe shall have
        the columns: open, high, low, close, volume and timestamps as index
        """

        if minute_dataframe is None:
            if self.minute_data is None:
                raise env.MissingHistoryData("{}–{} target {}min without minute data ({})".format(
                                         self.base, self.quote, self.target_key, self.vec))
        else:
            self.minute_data = minute_dataframe
        if self.minute_data.empty is None:
            self.minute_data = None
            raise env.MissingHistoryData("{}–{} target {}min with empty minute data".format(
                                     self.base, self.quote, self.target_key))
        # to be modified: tf_aggs = self.__calc_aggregation(self.minute_data, Env.time_aggs)
        if "target" not in self.minute_data:
            super().__add_targets(self.target_key, tf_aggs[self.target_key])  # add aggregation targets
            super().__add_targets2(self.target_key, tf_aggs[self.target_key])  # add aggregation targets
            self.minute_data["target"] = tf_aggs[self.target_key]["target"]
            # print("calculating targets")
        else:
            # print("reusing targets")
            pass
        # to be modified: self.vec = self.__expand_target_feature_vectors(tf_aggs, self.target_key)
        if "target" not in self.vec:
            self.vec["target"] = self.minute_data["target"]

        # print(f"{len(self.vec)} feature vectors of {len(self.vec.iloc[0])-2} features")

    def load_classifier_features(self):
        """ Loads the historic data temporarily, calculates the features and targets
            and releases the original data afterwards
        """
        try:
            df = ccd.load_asset_dataframe(self.base)
        except env.MissingHistoryData:
            raise
        else:
            self.calc_features_and_targets(df)
            report_setsize(self.base, self.vec)
