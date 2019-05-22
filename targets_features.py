#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
# import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from sklearn.utils import Bunch
import numpy as np
from queue import Queue


DATA_KEYS = ['open', 'high', 'low', 'close', 'volume']  # , 'price'

# DATA_PATH = os.getcwd() # local execution - to be avoided due to Git sync size
# DATA_PATH = '/Users/tc/crypto/Features'  # local execution
DATA_PATH = '/Users/tc/crypto/TestFeatures'  # local execution
# DATA_PATH = '/content/gdrive/My Drive/Features' # Colab execution

PICKLE_EXT = ".pydata"  # pickle file extension
JSON_EXT = ".json"  # msgpack file extension
MSG_EXT = ".msg"  # msgpack file extension

DT_FORMAT = '%Y-%m-%d_%H:%M'
FEE = 1/1000  # in per mille, transaction fee is 0.1%
TRADE_SLIP = 0 # 1/1000  # in per mille, 0.1% trade slip
BUY_THRESHOLD = 10/1000  # in per mille
SELL_THRESHOLD = -5/1000  # in per mille
VOL_BASE_PERIOD = '1D'
CPC = 'CPC'
HOLD = '-'
BUY = 'buy'
SELL = 'sell'
NA = 'not assigned'
TRAIN = 'training'
VAL = 'validation'
TEST = 'test'
TARGETS = {HOLD: 0, BUY: 1, SELL: 2, NA: 11}  # dict with int encoding of target labels
TARGET_NAMES = {0: HOLD, 1: BUY, 2: SELL, 11: NA}  # dict with int encoding of targets
TARGET_KEY = 5
TIME_AGGS = {CPC: 0, 1: 10, 5: 10, 15: 10, 60: 10, 4*60: 10}
# TIME_AGGS = {1: 10, 5: 10}
LBL = {NA: 0, TRAIN: -1, VAL: -2, TEST: -3}
# BASES = ['xrp', 'eos', 'bnb', 'btc', 'eth', 'neo', 'ltc', 'trx']
BASES = ['bnb', 'xrp']

def timestr(ts=None):
    if ts is None:
        return datetime.now().strftime(DT_FORMAT)
    else:
        return ts.strftime(DT_FORMAT)

def time_in_index(dataframe_with_timeseriesindex, tic):
    return True in dataframe_with_timeseriesindex.index.isin([tic])

def targets_to_features(tfv_ta_df, target_df):
    """Extracts a sample subset with targets and features of a specific time aggregation
    based on given targets. target_df and tfv_ta_df both have to share the same index basis.
    The index of target_df shall be a subset of tfv_ta_df.
    """
    return tfv_ta_df[tfv_ta_df.index.isin(target_df.index)]

def load_sample_set_split_config(config_fname):
    # fname = '/Users/tc/tf_models/crypto' + '/' + 'sample_set_split.config'
    try:
        cdf = pd.read_csv(config_fname, skipinitialspace=True, \
                          converters={'set_start': np.datetime64})
        print(f"sample set split loaded from {config_fname}")
        cdf.sort_values(by=['set_start'], inplace=True)
        cdf = cdf.reset_index(drop=True)
        cdf['set_stop'] = cdf.set_start.shift(-1)
        cdf['set_start'] = cdf.set_start.dt.tz_localize(None)
        cdf['set_stop'] = cdf.set_stop.dt.tz_localize(None)
        cdf['diff'] = cdf['set_stop'] - cdf['set_start']
        print(cdf)
        # print(cdf.dtypes)
        return cdf
    except IOError:
        print(f"pd.read_csv({config_fname}) IO error")
        return None

def save_asset_dataframe(df, path, cur_pair):
    # "saves the object via msgpack"
    cur_pair = cur_pair.replace('/', '_')
    fname = path +  '/' + cur_pair + '_DataFrame.msg'
    print("{}: writing {} {} tics ({} - {})".format(
        datetime.now().strftime(DT_FORMAT), cur_pair, len(df), df.index[0].strftime(DT_FORMAT),
        df.index[len(df)-1].strftime(DT_FORMAT)))
    df.to_msgpack(fname)

def load_asset_dataframefile(fname):
    # "loads the object via msgpack"
    df = None
    try:
        df = pd.read_msgpack(fname)
        print("{}: load {} {} tics ({} - {})".format(
            datetime.now().strftime(DT_FORMAT), fname, len(df), df.index[0].strftime(DT_FORMAT),
            df.index[len(df)-1].strftime(DT_FORMAT)))
    except IOError:
        print(f"{timestr()} load_asset_dataframe ERROR: cannot load {fname}")
    except ValueError:
        return None
    return df

def dfdescribe(desc, df):
    print(desc)
    print(df.describe())
    print(df.head())
    print(df.tail())

def merge_asset_dataframe(path, base):
    # "loads the object via msgpack"
    fname = path +  '/' + 'btc_usdt' + '_DataFrame.msg'
    btcusdt = load_asset_dataframefile(fname)
    if base != 'btc':
        fname = path +  '/' + base + '_btc' + '_DataFrame.msg'
        basebtc = load_asset_dataframefile(fname)
        dfdescribe(f"{base}-btc", basebtc)
        fname = path +  '/' + base + '_usdt' + '_DataFrame.msg'
        baseusdt = load_asset_dataframefile(fname)
        dfdescribe(f"{base}-usdt", baseusdt)
        if (baseusdt.index[0] <= basebtc.index[0]) or (baseusdt.index[0] <= btcusdt.index[0]):
            basemerged = baseusdt
        else:
            basebtc = basebtc[basebtc.index.isin(btcusdt.index)]
            basemerged = pd.DataFrame(btcusdt)
            basemerged = basemerged[basemerged.index.isin(basebtc.index)]
            for key in DATA_KEYS:
                if key != 'volume':
                    basemerged[key] = basebtc[key] * btcusdt[key]
            basemerged['volume'] = basebtc.volume
            dfdescribe(f"{base}-btc-usdt", basemerged)

            baseusdt = baseusdt[baseusdt.index.isin(basemerged.index)]
            assert not baseusdt.empty
            basemerged.loc[baseusdt.index] = baseusdt[:]  # take values of cusdt where available
    else:
        basemerged = btcusdt
    dfdescribe(f"{base}-merged", basemerged)

    save_asset_dataframe(basemerged, DATA_PATH, base + 'usdt')

    return basemerged

def load_asset_dataframe(path, base):
    # "loads the object via msgpack"
    fname = path +  '/' + base + 'usdt' + '_DataFrame.msg'
    dfbu = load_asset_dataframefile(fname)
    return dfbu

def report_setsize(setname, df):
    hc = len(df[df.target == TARGETS[HOLD]])
    sc = len(df[df.target == TARGETS[SELL]])
    bc = len(df[df.target == TARGETS[BUY]])
    tc = hc + sc + bc
    print(f"buy {bc} sell {sc} hold {hc} total {tc} on {setname}")


class MissingHistoryData(Exception):
    pass


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

    def __init__(self, aggregations=TIME_AGGS, target_key=TARGET_KEY, cur_pair=None):
        """Receives the key attributes for feature generation

        aggregations:
            dict with required time aggregation keys and associated number
            of periods that shall be compiled in a corresponding feature vector
        target_key:
            has to be a key of aggregations. Targets are only calculated for that target_key


        """
        self.time_aggregations = aggregations  # aggregation in minutes of trailing DHTBV values
        self.tf_aggs = dict()  # feature and target aggregations
        self.cur_pair = cur_pair
        self.tf_vectors = None
        self.minute_data = None
        self.target_key = target_key
        assert target_key in aggregations, "target key {} not in aggregations {}".format(
            target_key, aggregations)

    def derive_features(self, time_agg):
        """derived features in relation to price based on the provided time aggregated dataframe df
        with the exception of the derived feature 'delta' that is calculated together with targets
        """
        # price deltas in 1/1000
        df = self.tf_aggs[time_agg]
        df['height'] = (df['high'] - df['low']) / df['close'] * 1000
        df.loc[df['close'] > df['open'], 'top'] = (df['high'] - df['close']) / df['close'] * 1000
        df.loc[df['close'] <= df['open'], 'top'] = (df['high'] - df['open']) / df['close'] * 1000
        df.loc[df['close'] > df['open'], 'bottom'] = (df['open'] - df['low']) / df['close'] * 1000
        df.loc[df['close'] <= df['open'], 'bottom'] = (df['close'] - df['low']) / df['close'] * 1000


    def calc_aggregation(self):
        """Time aggregation through rolling aggregation with the consequence that new data is
        generated every minute and even long time aggregations reflect all minute bumps in their
        features

        in:
            dataframe of minute data of a currency pair;
        out:
            dict of dataframes of aggregations with features and targets
        """
        mdf = df = self.minute_data  # .copy()
        mdf['vol'] = (mdf['volume'] - mdf.volume.rolling(VOL_BASE_PERIOD).mean()) \
                 / mdf.volume.rolling(VOL_BASE_PERIOD).mean()
        mdf = mdf.fillna(value={'vol': 0.000001})
        maxmin = 0
        for time_agg in self.time_aggregations:
            if isinstance(time_agg, int):
                if (time_agg * self.time_aggregations[time_agg]) > maxmin:
                    maxmin = time_agg * self.time_aggregations[time_agg]
        if maxmin > len(df.index):
            raise MissingHistoryData("History data has {} samples but should have >= {}".format(
                    len(df.index), maxmin))
        for time_agg in self.time_aggregations:
            # print(f"{datetime.now()}: time_aggregation {time_agg}")
            if isinstance(time_agg, int):
                if time_agg > 1:
                    df = pd.DataFrame()
                    df['open'] = mdf.open.shift(time_agg-1)
                    df['high'] = mdf.high.rolling(time_agg).max()
                    df['low'] = mdf.low.rolling(time_agg).min()
                    df['close'] = mdf.close
                    df['vol'] = mdf.vol.rolling(time_agg).mean()
                df['delta'] = (mdf.close - mdf.close.shift(time_agg)) / mdf.close.shift(time_agg)
                self.tf_aggs[time_agg] = df
                self.derive_features(time_agg)
            else:
                self.tf_aggs[time_agg] = pd.DataFrame(self.minute_data.close)  # CPC dummy df

    def calc_features_and_targets(self, minute_dataframe):
        assert not minute_dataframe.empty, "empty dataframe"
        self.minute_data = minute_dataframe
        self.calc_aggregation()  # calculate features and time aggregations of features
        if self.target_key == CPC:
            for time_agg in self.time_aggregations:
                if isinstance(time_agg, int):
                    self.add_period_specific_targets(time_agg)  # add aggregation targets
            best_perf = self.cpc_best_path()
            print(f"best potential performance among all aggregations: {best_perf:%}")
        else:
            self.add_period_specific_targets(self.target_key)  # add aggregation targets
        self.tf_vectors = TfVectors(tf=self)

    def add_period_specific_targets(self, time_agg):
        "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"
        #print(f"{datetime.now()}: add_period_specific_targets {time_agg}")
        df = self.tf_aggs[time_agg]
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
                if loss[slot] < SELL_THRESHOLD:  # reset win monitor because dip exceeded threshold
                    win[slot] = 0.
                    df.iat[lossix[slot], lix] = lasttarget[slot] = TARGETS[SELL]
                    lossix[slot] += 1  # allow multiple signals if conditions hold
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
                        loss[slot] = 0.  # reset loss monitor as it recovered before  sell threshold
                if win[slot] > BUY_THRESHOLD:  # reset win monitor because dip exceeded threshold
                    loss[slot] = 0.
                    df.iat[winix[slot], lix] = lasttarget[slot] = TARGETS[BUY]
                    winix[slot] += 1  # allow multiple signals if conditions hold
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

    def cpc_best_path(self):
        """identify best path through all aggregations - use all buy and sell signals.
        The search space is limited because costs increase the same for all paths.
        Hence, one has only to check the most recent 2 buy signals and
        the most recent 2 sell signals for maximization.

        1. no holding with buy signal:
            open potential buy
        2. holding, old open potential sell with sell signal:
            current performance - fee >= potential transaction performance:
                discard old potential sell and open new potential sell
            else:
                ignore new sell signal
                if potential transaction performance > 0:
                    execute potential sell and remove corresponding buy and sell
        3. holding, no open potential sell with sell signal:
            open potential sell
        4. holding, with buy signal:
            current performance < -fee:
                if potential transaction performance > 0:
                    execute potential sell and remove corresponding buy and sell
                discard old potential buy and open new potential buy
                discard any potential sell including new ones
                current performance = -fee
            current performance >= -fee: do nothing, i.e.
                discard new buy and hold on to old potential buy
                hold on to any potential sells including new ones

        """

        # cpc_df = pd.DataFrame(self.minute_data, columns=['close']) # already done
        if self.target_key != CPC:
            return self.target_performance()
        cpc_df = self.tf_aggs[CPC]
        cpc_df[CPC+'_target'] = TARGETS[HOLD]
        target_ix = cpc_df.columns.get_loc(CPC+'_target')
        close_ix = cpc_df.columns.get_loc('close')
        col_ix = dict()
        for time_agg in self.time_aggregations:
            if isinstance(time_agg, int):
                c_name = str(time_agg) + '_target'
                cpc_df[c_name] = self.tf_aggs[time_agg].target
                col_ix[time_agg] = cpc_df.columns.get_loc(c_name)
        assert cpc_df.index.is_unique, "unexpected not unique index"
        print(f"{datetime.now()}: best path")
        holding = False
        pot_transaction = False  # becomes true if potential buy and potential sell are detected
        transaction_perf = current_perf = best_perf = 0.
        buy_tix = sell_tix = 0
        last = cpc_df.iat[0, close_ix]
        for tix in range(len(cpc_df)):  # tix = time index
            this = cpc_df.iat[tix, close_ix]
            tix_perf = ((this - last) / last) # no longer in per mille * 1000)
            last = this
            sell_sig = False
            buy_sig = False
            for six in col_ix:
                signal = cpc_df.iat[tix, col_ix[six]]
                if signal == TARGETS[BUY]:
                    buy_sig = True
                if signal == TARGETS[SELL]:
                    sell_sig = True
            if holding:
                current_perf += tix_perf
                if sell_sig:
                    if pot_transaction:  # use case 2
                        if (current_perf - FEE) >= transaction_perf:  # reset transaction
                            transaction_perf = current_perf - FEE
                            sell_tix = tix
                            pot_transaction = True  # remains True
                        else:
                            if transaction_perf > 0:  # execute transaction
                                # assert buy_tix < sell_tix, \
                                # f"inconsistent tix marks, buy: {buy_tix} sell: {sell_tix}"
                                best_perf += transaction_perf
                                cpc_df.iat[buy_tix, target_ix] = TARGETS[BUY]
                                cpc_df.iat[sell_tix, target_ix] = TARGETS[SELL]
                            pot_transaction = False
                            holding = False
                            current_perf = transaction_perf = 0.
                    else:  # use case 3 = no potential transaction
                        transaction_perf = current_perf - FEE
                        sell_tix = tix
                        pot_transaction = True  # remains True
                        # assert buy_tix < sell_tix, "inconsistent buy/sell tix marks"
                if buy_sig:  # use case 4, from different time aggs there can be sell and buy!
                    if current_perf < -FEE:  # set new buy_tix but check for a potential tr.
                        if pot_transaction and (transaction_perf > 0):  # execute transaction
                            # assert buy_tix < sell_tix, \
                            # f"inconsistent tix marks, buy: {buy_tix} sell: {sell_tix}"
                            best_perf += transaction_perf
                            cpc_df.iat[buy_tix, target_ix] = TARGETS[BUY]
                            cpc_df.iat[sell_tix, target_ix] = TARGETS[SELL]
                            pot_transaction = False
                            holding = False
                            transaction_perf = 0.
                        buy_tix = tix
                        current_perf = -FEE
                        holding = True
            else:  # not holding
                if buy_sig:  # use case 1 = not holding with buy signal
                    buy_tix = tix
                    current_perf = -FEE
                    holding = True
                    pot_transaction = False
        # self.tf_aggs[CPC] = cpc_df # already done
        return best_perf

    def target_performance(self):
        """calculates the time aggregation specific performance of target_key
        """
        print(f"{datetime.now()}: calculate target_performance")
        target_df = self.tf_aggs[self.target_key]
        perf = 0.
        ta_holding = False
        col_ix = target_df.columns.get_loc('target')
        assert col_ix > 0, f"did not find column {col_ix} of {self.target_key}"
        close_ix = target_df.columns.get_loc('close')

        assert target_df.index.is_unique, "unexpected not unique index"
        last = target_df.iat[0, close_ix]
        for tix in range(len(target_df)):  # tix = time index
            this = target_df.iat[tix, close_ix]
            tix_perf = ((this - last) / last) # no longer in per mille * 1000)
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

    def calc_performances(self):
        """calculate all time aggregation specific performances
        as well as the CPC summary best path performance.
        """
        perf = dict()
        if self.target_key == CPC:
            print(f"{datetime.now()}: calculate performances")
            cpc_df = self.tf_aggs[CPC]
            col_ix = dict()
            ta_holding = dict()
            for time_agg in self.time_aggregations:
                perf[time_agg] = 0.
                ta_holding[time_agg] = False
                t_name = str(time_agg) + '_target'
                col_ix[time_agg] = cpc_df.columns.get_loc(t_name)
                assert col_ix[time_agg] > 0, f"did not find column {col_ix[time_agg]} of {time_agg}"
            close_ix = cpc_df.columns.get_loc('close')

            assert cpc_df.index.is_unique, "unexpected not unique index"
            last = cpc_df.iat[0, close_ix]
            for tix in range(len(cpc_df)):  # tix = time index
                this = cpc_df.iat[tix, close_ix]
                tix_perf = ((this - last) / last) # no longer in per mille * 1000)
                last = this
                for ta_ix in perf:
                    signal = cpc_df.iat[tix, col_ix[ta_ix]]
                    if ta_holding[ta_ix]:
                        perf[ta_ix] += tix_perf
                    if (signal == TARGETS[BUY]) and (not ta_holding[ta_ix]):
                        perf[ta_ix] -= FEE
                        ta_holding[ta_ix] = True
                    if (signal == TARGETS[SELL]) and ta_holding[ta_ix]:
                        perf[ta_ix] -= FEE
                        ta_holding[ta_ix] = False
        else:
            perf[self.target_key] = self.target_performance()
        return perf

def smallest_dict_key(thisdict):
    smallest_key = 5000
    for k in thisdict:
        if isinstance(k, int):
            if k < smallest_key:
                smallest_key = k
    assert smallest_key != 5000, "no int in dict keys"
    return smallest_key

class TfVectors:
    """Container class for targets and features of a currency pair

    attributes:
        currency pair - pair used to derive targets and features
        vecs - dict with either CPC as summary label or time aggregation in  minutes as keys
        and corresponding data frames as dict items.


    feature vector dataframe columns:
        timeseries index in minutes of data used

        buy - 1 if buy signal, 0 otherwise

        sell - 1 if sell signal, 0 otherwise

        series of delta (D), height (H), top(T), bottom (B), volume (V)
        each column name encodes the aggregation in minutes first, followed by the series nbr,
        followed by a single character DHTBV indicating the data element
    """

#    def build_classifier_vectors(self):
    def __init__(self, tf):
        """Builds a target and feature vector sequence with DHTBV feature sequences of
        n time steps (tics) as configured in time_aggregations in T units.
        While TargetFeatures calculates features and targets per minute,
        the most important step in TfVectors is
        1) the concatenation of feature vectors per sample to provide a history for the classifier
        2) discarding the original currency values that are not used as features (except 'close')

        Result:
            a self.vecs dict with keys as in time_aggregations plus 'CPC' that
            are referring to DataFrames with feature vectors as rows. The column name indicates
            the type of feature, i.e. either 'target' or 'D|H|T|B|V' with aggregation+'T_'
            as prefix
        """
        self.cur_pair = tf.cur_pair
        assert tf is not None, "request for feature vectors without basis data"
        self.vecs = dict()  # full fleged feature vectors and their targets
        if tf.target_key == CPC:
            self.expand_CPC_feature_vectors(tf)
        else:
            self.expand_target_feature_vectors(tf)
#        print("target: {} has {} features x {} samples".format(tf.cur_pair,
#              len(self.vecs[tf.target_key].columns)-2, len(self.vecs[tf.target_key].index)))

    def expand_target_feature_vectors(self, tf):
        """Builds a target and feature vector for just the target_key with
        1 minute DHTBV and D*V feature sequences and the remaining D sequences of
        n time steps (tics) as configured in time_aggregations in T units.
        While TargetFeatures calculates features and targets per minute,
        the most important step in TfVectors is
        1) the concatenation of feature vectors per sample to provide a history for the classifier
        2) discarding the original currency values that are not used as features (except 'close')

        Result:
            a self.vecs dict with the single target_key that is
            referring to a DataFrame with feature vectors as rows. The column name indicates
            the type of feature, i.e. either 'target', 'close' or 'D|H|T|B|V|DV' in case of
            1 minute aggregation or just 'D' for all other aggregations with aggregation+'T_'
            as column prefix
        """
        # print(f"{datetime.now()}: processing {self.cur_pair} for target classifier {tf.target_key}")
        trgt = tf.target_key
        df = pd.DataFrame(tf.tf_aggs[trgt], columns=['close', 'target'])
        # df['target'] = tf.tf_aggs[trgt]['target'].astype(int)
        skey = smallest_dict_key(tf.tf_aggs)
        sts = tf.tf_aggs[skey].index[0]
        ets = tf.tf_aggs[skey].index[len(tf.tf_aggs[skey].index)-1]
        mindiff = (ets - sts) / timedelta(minutes=1) + 1
        maxos = 1440 * tf.time_aggregations[1440]
        # print(f"expand from {timestr(sts)} to {timestr(ets)} = {mindiff} minutes in {len(tf.tf_aggs[skey].index)} rows, required {maxos+1}")
        for ta in tf.tf_aggs:
            for tics in range(tf.time_aggregations[ta]):
                ctitle = str(ta) + 'T_' + str(tics) + '_'
                if isinstance(ta, int):
                    offset = tics*ta
                else:
                    offset = tics
                # now add feature columns according to aggregation
                df[ctitle + 'D'] = tf.tf_aggs[ta].delta.shift(offset)
                if ta == skey:
                    df[ctitle + 'H'] = tf.tf_aggs[ta].height.shift(offset)
                    df[ctitle + 'T'] = tf.tf_aggs[ta].top.shift(offset)
                    df[ctitle + 'B'] = tf.tf_aggs[ta].bottom.shift(offset)
                    df[ctitle + 'V'] = tf.tf_aggs[ta].vol.shift(offset)
                    df[ctitle + 'DV'] = tf.tf_aggs[ta].vol.shift(offset) *\
                                                     tf.tf_aggs[ta].delta.shift(offset)
        self.vecs[trgt] = df.dropna()
        if self.vecs[trgt].empty:
            print("empty dataframe from TargetsFeatures")
        self.cut_back_to_same_sample_tics(tf)

    def expand_CPC_feature_vectors(self, tf):
        """Builds a target and feature vector sequence with DHTBV feature sequences of
        n time steps (tics) as configured in time_aggregations in T units.
        While TargetFeatures calculates features and targets per minute,
        the most important step in TfVectors is
        1) the concatenation of feature vectors per sample to provide a history for the classifier
        2) discarding the original currency values that are not used as features (except 'close')

        Result:
            a self.vecs dict with keys as in time_aggregations plus 'CPC' that
            are referring to DataFrames with feature vectors as rows. The column name indicates
            the type of feature, i.e. either 'target' or 'D|H|T|B|V' with aggregation+'T_'
            as prefix
        """
        print(f"{datetime.now()}: processing {self.cur_pair}")
        for ta in tf.tf_aggs:
            print(f"{datetime.now()}: build classifier vectors {ta}")
            t_name = str(ta) + '_target'
            self.vecs[ta] = pd.DataFrame(tf.minute_data, columns=['close', 'target'])
            self.vecs[ta]['target'] = tf.tf_aggs[CPC][t_name].astype(int)

            for tics in range(tf.time_aggregations[ta]):
                tgt = str(ta) + 'T_' + str(tics) + '_'
                if isinstance(ta, int):
                    offset = tics*ta
                else:
                    offset = tics
                # now add feature columns according to aggregation
                self.vecs[ta][tgt + 'D'] = tf.tf_aggs[ta].delta.shift(offset)
                self.vecs[ta][tgt + 'H'] = tf.tf_aggs[ta].height.shift(offset)
                self.vecs[ta][tgt + 'T'] = tf.tf_aggs[ta].top.shift(offset)
                self.vecs[ta][tgt + 'B'] = tf.tf_aggs[ta].bottom.shift(offset)
                self.vecs[ta][tgt + 'V'] = tf.tf_aggs[ta].vol.shift(offset)
            self.vecs[ta].dropna(inplace=True)
            assert not self.vecs[ta].empty, "empty dataframe from TargetsFeatures"
        self.cut_back_to_same_sample_tics(tf)

    def cut_back_to_same_sample_tics(self, tf):
        """
            There will be a different number of feature vectors per aggregation due to the
            nan in the beginning rows that have missing history vectors before them.
            This leads to issues of different array lengths when constructing the CPC features
            out of the time aggregation specific probability result vectors, which is corrected
            with this function.
        """
        df = None
        for ta in self.vecs:
            if df is None:
                df = self.vecs[ta]
            elif len(self.vecs[ta]) < len(df):
                df = self.vecs[ta]
        for ta in self.vecs:
            self.vecs[ta] = self.vecs[ta][self.vecs[ta].index.isin(df.index)]

    def vec(self, key):
        "Returns the dataframe of the given key"
        return self.vecs[key]

    def most_recent_features_only(self):
        """Reduces rows to just the last one"""
        lastts = None
        for ta in self.vecs:
            tfv_ta = self.vecs[ta]
            if tfv_ta.empty:
                print("most_recent_features_only ERROR: unexpected empty self.vecs(ta)")
                return None
            if lastts is None:
                lastts = tfv_ta.index[len(tfv_ta.index)-1]
            if tfv_ta.empty:
                print("most_recent_features_only ERROR: unexpected empty self.vecs(ta)")
                return None
            df = pd.DataFrame(tfv_ta.loc[lastts]).transpose()
            if len(df) != 1:
                print(f"most_recent_features_only ERROR: unexpected self.vecs(ta) len {len(df)}")
            self.vecs[ta] = df

    def any_key(self):
        """Returns the key of any TfVectors dataframe that has a complete timestamp index
            and minute close data. It may not have a 'target' column
        """
        return list(self.vecs.keys())[0]

    def signal_sequences(self, key):
        "provides a histogram of consecutive signals for the given key data"
        c_df = self.vecs[key]
        b_seq = dict()
        s_seq = dict()
        h_seq = dict()
        seq = {BUY: b_seq, SELL: s_seq, HOLD: h_seq}
        b_count = s_count = h_count = 0
        b_col = c_df.columns.get_loc(TARGETS[BUY])
        s_col = c_df.columns.get_loc(TARGETS[SELL])
        for t_ix in range(len(c_df)):
            b_sig = c_df.iat[t_ix, b_col]
            if b_sig > 0:
                b_count += 1
                assert b_sig == 1, f"unexpected buy signal value: {b_sig}"
            else:
                if b_count > 0:
                    if b_count in b_seq:
                        b_seq[b_count] += 1
                    else:
                        b_seq[b_count] = 1
                    b_count = 0
            s_sig = c_df.iat[t_ix, s_col]
            if s_sig > 0:
                s_count += 1
                assert s_sig == 1, f"unexpected sell signal value: {s_sig}"
            else:
                if s_count > 0:
                    if s_count in s_seq:
                        s_seq[s_count] += 1
                    else:
                        s_seq[s_count] = 1
                    s_count = 0
            assert not ((b_sig > 0) and (s_sig > 0)), f"unexpected double signal sell and buy"
            if (b_sig == 0) and (s_sig == 0):
                h_count += 1
            else:
                if h_count > 0:
                    if h_count in h_seq:
                        h_seq[h_count] += 1
                    else:
                        h_seq[h_count] = 1
                    h_count = 0
        return seq

    def timeslice_targets_as_configured(self, key, config_fname):
        """Splits the data set into training set, validation set and test set as configured
        in the given file.

        Returns:
        ========
        The dataframe receives a column 'slc' that identifies the assignment of samples to
        training, validation, or test set and a column 'slcix' with the slice index to
        enable slicewise adaptation and evaluation.
        """
        sdf = load_sample_set_split_config(config_fname)
        c_df = pd.DataFrame(self.vecs[key].target) #only use targets for filter logic
        c_df['slc'] = LBL[NA]
        sdf_ix = -1  # increment in while loop to 0 == start
        c_df['slcix'] = sdf_ix
        sdf_lbl = NA
        sdf_pair = "no pair"
        sdf_stop = np.datetime64(c_df.index[0])  # force while loop
        slc_col = c_df.columns.get_loc('slc')
        slcix_col = c_df.columns.get_loc('slcix')
        trgt_col = c_df.columns.get_loc('target')
        hc = bc = sc = nc = uc = 0
        hct = bct = sct = nct = uct = 0
        for t_ix in range(len(c_df)):
            dt_stamp = np.datetime64(c_df.index[t_ix])
            while sdf_stop <= dt_stamp:
                tc = hc + bc + sc
                if tc > 0:
                    print("buy {} sell {} hold {} total {} na {} unknown {} on {} set {} {}".format(
                            bc, sc, hc, tc, nc, uc, sdf_pair,sdf_ix, sdf_lbl))
                    hct += hc
                    bct += bc
                    sct += sc
                    nct += nc
                    uct += uc
                    hc = bc = sc = nc = uc = 0
                sdf_ix += 1
                sdf_pair = sdf.loc[sdf_ix, 'set_pair']
                sdf_start = sdf.loc[sdf_ix, 'set_start']
                sdf_stop = sdf.loc[sdf_ix, 'set_stop']
                sdf_lbl = sdf.loc[sdf_ix, 'sample_set']
                if sdf_lbl  in LBL:
                    sdf_tag = LBL[sdf_lbl]
                else:
                    sdf_tag = LBL[NA]
                    print(f"unknwon config label '{sdf_lbl}'")
            if (sdf_start <= dt_stamp) and (sdf_stop > dt_stamp):
                trgt = c_df.iat[t_ix, trgt_col]
                if trgt == TARGETS[HOLD]:
                    hc += 1
                elif trgt == TARGETS[SELL]:
                    sc += 1
                elif trgt == TARGETS[BUY]:
                    bc += 1
                elif trgt == TARGETS[NA]:
                    nc += 1
                else:
                    uc += 1
                c_df.iat[t_ix, slc_col] = sdf_tag
                c_df.iat[t_ix, slcix_col] = sdf_ix
        print("buy {} sell {} hold {} total {} na {} unknown {} on whole set".format(
                bct, sct, hct, bct + sct + hct, nct, uct))

        train_df = c_df[c_df.slc == LBL[TRAIN]]
        val_df = c_df[c_df.slc == LBL[VAL]]
        test_df = c_df[c_df.slc == LBL[TEST]]
        seq = {TRAIN: train_df, VAL: val_df, TEST: test_df}

        report_setsize(TRAIN, seq[TRAIN])
        report_setsize(VAL, seq[VAL])
        report_setsize(TEST, seq[TEST])
        return seq

    def reduce_target_sequences(self, c_df, key):
        """
        Sequences of buy or sell samples will be reduced every n of the sequence according
        to the seq_red value(local variable).

        Returns:
        ========
        dataframe with reduced target sequences

        """

        def reduce_seq_targets(signal, target, c_df, t_ix, t_col, last, seq_count):
            if signal == target:
                if last == signal:
                    seq_count += 1
                else:
                    seq_count = 0
                    last = signal
                if (seq_count % seq_red) != 0:
                    c_df.iat[t_ix, t_col] = TARGETS[NA] # change to non valid target
            return last, seq_count

        if isinstance(key, int):
            seq_red = key # use only every aggregation sample in signal sequences
        else:
            seq_red = 4 # use only every seq_red sample in signal sequences
        t_col = c_df.columns.get_loc('target')
        seq_count = 0
        last = 'x'
        for t_ix in range(len(c_df)):
            sig = c_df.iat[t_ix, t_col]
            last, seq_count = reduce_seq_targets(sig, TARGETS[BUY], c_df,
                                                 t_ix, t_col, last, seq_count)
            last, seq_count = reduce_seq_targets(sig, TARGETS[SELL], c_df,
                                                 t_ix, t_col, last, seq_count)
            last, seq_count = reduce_seq_targets(sig, TARGETS[HOLD], c_df,
                                                 t_ix, t_col, last, seq_count)

    def balance_targets(self, l_df):
        t_col = l_df.columns.get_loc('target')
        # l_df = l_df.drop(['slc'], axis = 1) # not required as this is only a temp df
        h_count = len(l_df[l_df.target == TARGETS[HOLD]])
        s_count = len(l_df[l_df.target == TARGETS[SELL]])
        b_count = len(l_df[l_df.target == TARGETS[BUY]])
        smallest = min([h_count, b_count, s_count])
        if smallest == 0:
            print("warning: cannot balance as not all classes have representatives")
            report_setsize("val or train", l_df)
            return l_df
        h_ratio = smallest / h_count
        s_ratio = smallest / s_count
        b_ratio = smallest / b_count

        h_lvl = b_lvl = s_lvl = 0.0
        for t_ix in range(len(l_df)):
            this = l_df.iat[t_ix, t_col]
            if this == TARGETS[BUY]:
                if b_lvl <= 0:
                    b_lvl += 1 # keep that target
                else:
                    l_df.iat[t_ix, t_col] = TARGETS[NA] # dump that target
                b_lvl -= b_ratio
            elif this == TARGETS[SELL]:
                if s_lvl <= 0:
                    s_lvl += 1 # keep that target
                else:
                    l_df.iat[t_ix, t_col] = TARGETS[NA] # dump that target
                s_lvl -= s_ratio
            elif this == TARGETS[HOLD]:
                if h_lvl <= 0:
                    h_lvl += 1 # keep that target
                else:
                    l_df.iat[t_ix, t_col] = TARGETS[NA] # dump that target
                h_lvl -= h_ratio
        l_df = l_df[(l_df.target == TARGETS[BUY])| \
                    (l_df.target == TARGETS[SELL])| \
                    (l_df.target == TARGETS[HOLD])]
        return l_df

    def reduce_sequences_balance_targets(self, key, set_df, balance):
        subset_df = pd.DataFrame(set_df)
        self.reduce_target_sequences(subset_df, key)
        subset_df = subset_df[(subset_df.target == TARGETS[BUY])| \
                         (subset_df.target == TARGETS[SELL])| \
                         (subset_df.target == TARGETS[HOLD])]
        if balance:
            subset_df = self.balance_targets(subset_df)
        return subset_df

    def to_scikitlearn(self, df, np_data=None, descr=None):
        """Load and return the crypto dataset (classification).
        """

        fn_list = list(df.keys())
        fn_list.remove('target')
        fn_list.remove('close')
        if np_data is None:
#            data = df[fn_list].to_numpy(dtype=float) # incompatible with pandas 0.19.2
            data = df[fn_list].values
        else:
            data = np_data
#        target = df['target'].to_numpy(dtype=float) # incompatible with pandas 0.19.2
#        tics = df.index.to_numpy(dtype=np.datetime64) # incompatible with pandas 0.19.2
        target = df['target'].values # compatible with pandas 0.19.2
        close = df['close'].values # compatible with pandas 0.19.2
        tics = df.index.values # compatible with pandas 0.19.2
        feature_names = np.array(fn_list)
        target_names = np.array(TARGETS.keys())
        if descr is None:
            descr =self.cur_pair

        return Bunch(data=data, target=target, close=close,
                     target_names=target_names,
                     tics=tics,
                     descr=descr,
                     feature_names=feature_names)


    def timeslice_data_to_scikitlearn(self, key, train_ratio, val_ratio, balance, days):
        """Load and return the crypto dataset (classification).
        """

        seq = self.timeslice_and_select_targets(key, train_ratio, val_ratio, balance, days)
        for elem in seq:
            # h=a[a.index.isin(f.index)] use only the data that is also in the target df
            data_df = self.vecs[key][self.vecs[key].index.isin(seq[elem].index)]
            seq[elem] = self.to_scikitlearn(data_df)
            seq[elem].descr = self.cur_pair + " aggregation: " + str(key) + " " + elem
        return seq


def load_classifier_features(aggs, target, base):
    df = load_asset_dataframe(DATA_PATH, base)
    if df is None:
        tf = None
    else:
        cur_pair = base + "_usdt"
        tf = TargetsFeatures(aggregations=aggs, target_key=target, cur_pair=cur_pair)
        tf.calc_features_and_targets(df)
        # test = tf.calc_performances()
        # for p in test:
        #     print(f"performance potential for aggregation {p}: {test[p]:%}")
    return tf

class TrainingSet:

    def __init__(self, history_sets):
        self.hs = history_sets
        self.bases_iter = None
        self.base = None
        self.tfv = None

    def __iter__(self):
        self.bases_iter = iter(self.hs.baselist)
        return self

    def __next__(self):
        try:
            if self.hs.trainctrl.empty:
                print("no training set in HistorySets")
                raise StopIteration()
            while True:
                self.base = self.bases_iter.__next__()
                sym = self.base + "_usdt"
                # tf = load_classifier_features(self.hs.tf_aggs, self.hs.target_key, self.base)
                # self.tfv = tf.tf_vectors
                try:
                    train_base_df = self.hs.trainctrl.loc[self.hs.trainctrl.sym == sym]
                    print(f"training set with {len(train_base_df)} samples for {sym}")
                    return train_base_df
                except KeyError:
                    print(f"no training set for {sym}")
                    pass
        except StopIteration:
            raise StopIteration()


class HistorySets:
    """Container class for targets and features of a currency pair
    timeblock is the time window in minutes that is analyzed equally for all bases to
    avoid label leakage. timeblocks are then distributed to train, validate and test to
    balance buy and sell signals.
    """

    def __init__(self, target_key, aggs):
        """Uses history data of baselist/USDT as history set to train and evaluate.

        training control:
        =================
        - df[common timeline index, base, target, training_count, buy_prob, sell_prop, hold_prop,
           train_next]
        - iteration within an epoch, every iternation-th class of a sym is used
        - tcount is incremented with every training cycle usage
        - buy_prob, sell_prop, hold_prop are the class probabilities of the last evaluation
        - use is an earmark that this sample shall be used for the next training epoch/validation
        """
        self.baselist = BASES
        self.target_key = target_key
        self.tf_aggs = aggs
        self.timeblock = 4*7*24*60  # time window in minutes that is analyzed equally for all bases
        self.fixtic = None  # tic as fixpoint for timeblock
        self.analysis = pd.DataFrame(columns=['sym', 'set_type', 'start', 'end', 'tics',
                                              'buys', 'sells', 'avg_vol', 'novol_count'])

        self.trainctrl = pd.DataFrame(columns=['sym', 'timestamp', 'target', 'use',
                                               'buy_prob', 'sell_prop', 'hold_prop',
                                               'iteration', 'tcount'])
        self.training_iterations = 10
        self.valctrl = pd.DataFrame(columns=['sym', 'timestamp', 'target', 'use',
                                             'buy_prob', 'sell_prop', 'hold_prop'])
        self.testctrl = pd.DataFrame(columns=['sym', 'timestamp', 'target', 'use',
                                              'buy_prob', 'sell_prop', 'hold_prop'])

        # for base in self.baselist:
        #     merge_asset_dataframe(DATA_PATH, base)
        self.load_sets_config(self.sets_config_fname(target_key))
        assert not self.analysis.empty, f"{timestr()}: missing sets config - generating it now"
        # self.analyze_bases(target_key, aggs)

    def load_sets_config(self, config_fname):

        def use_settype_total():
            """Uses the set_type of 'total' and apllies it to all sets with such timeblock.
            """
            cdf = self.analysis.set_index('end')
            cdf['set_type'] = cdf.loc[cdf.sym == 'total']['set_type']
            # cdf['end'] = cdf.index  # is already doen by reset_index
            self.analysis = cdf.reset_index()

        # fname = '/Users/tc/tf_models/crypto' + '/' + 'sample_set_split.config'
        try:
            self.analysis = pd.read_csv(config_fname, skipinitialspace=True, sep='\t')
        except IOError:
            print(f"pd.read_csv({config_fname}) IO error")
            return None
        # use_settype_total()
        # self.analysis.to_csv(config_fname, sep='\t', index=False)
        self.prepare_training()

    def prepare_training(self):
        """Prepares training, validation and test sets with targets and admin info
        (see class description). These determine the samples per set_type and whether
        they are used in an iteration.

        It is assumed that load_sets_config was called and self.analysis contains the
        proper config fiel content.
        """

        def label_iterations(df, buys, sells, holds):
            """Each sample is assigned to an iteration. The smallest class has
            self.training_iterations iterations, i.e. each sample is labeled
            with an iteration between 0 and self.training_iterations-1.
            The larger classes have more iteration labels according to
            their ratio with the smallest class. This is determines per sym,
            timeblock and target class.

            This sample distribution over iterations shall balance the training
            of the classifiers and not bias the classifier too much in an early learning cycle.
            """
            min_iter = min([buys, sells, holds])
            b_iter = buys / min_iter * self.training_iterations
            s_iter = sells / min_iter * self.training_iterations
            h_iter = holds / min_iter * self.training_iterations
            bi = si = hi = 0
            tix = df.columns.get_loc('target')
            iix = df.columns.get_loc('iteration')
            for ix in range(len(df.index)):
                target = df.iat[ix, tix]
                if target == TARGETS[BUY]:
                    df.iat[ix, iix] = bi
                    bi = (bi + 1) % b_iter
                elif target == TARGETS[SELL]:
                    df.iat[ix, iix] = si
                    si = (si + 1) % s_iter
                elif target == TARGETS[HOLD]:
                    df.iat[ix, iix] = hi
                    hi = (hi + 1) % h_iter
                else: # hold
                    print(f"error: unexpected target {target}")

        def samples_concat(target, to_be_added):
            if target.empty:
                target = to_be_added
                print("target empty --> target = to_be_added", target.head(), target.tail())
                return to_be_added
            if False:
                # debugging output
                elen = len(target)
                xdf = target.tail()
                if ('iteration' in xdf.columns):
                    xdf = xdf[['target', 'timestamp', 'iteration']]
                else:
                    xdf = xdf[['target', 'timestamp']]
                print(f'target len: {elen}', xdf)
                ydf = to_be_added.head()
                if ('iteration' in ydf.columns):
                    ydf = ydf[['target', 'timestamp', 'iteration']]
                else:
                    ydf = ydf[['target', 'timestamp']]
                print(f'time agg timeblock len: {len(to_be_added)}', ydf)
            target = pd.concat([target, to_be_added], sort=False)
            if False:
                # debugging output
                zdf = target.iloc[range(elen-5,elen+5)]
                elen = len(target)
                if ('iteration' in zdf.columns):
                    zdf = zdf[['target', 'timestamp', 'iteration']]
                else:
                    zdf = zdf[['target', 'timestamp']]
                print(f'concat with new len {elen} result at interface: ', zdf)
            return target

        def extract_set_type_targets(sym, tf, set_type):
            try:
                print(f"extracting {set_type} for {sym}")
                dfcfg = self.analysis.loc[(self.analysis.set_type == set_type) &
                                          (self.analysis.sym == sym)]
            except KeyError:
                print(f"no {set_type} set for {sym}")
                return None
            dft = tf.tf_aggs[self.target_key]
            extract = None
            for block in dfcfg.index:
                df = dft.loc[(dft.index >= dfcfg.at[block, 'start']) &
                              (dft.index <= dfcfg.at[block, 'end']), ['target']]
                df['timestamp'] = df.index
                df['sym'] = sym
                df['use'] = True
                df['buy_prob'] = float(0)
                df['sell_prop'] = float(0)
                df['hold_prop'] = float(0)
                if set_type == TRAIN:
                    df['tcount'] = int(0)
                    df['iteration'] = int(0)
                    buys = dfcfg.at[block, 'buys']
                    sells = dfcfg.at[block, 'sells']
                    holds = dfcfg.at[block, 'tics'] - sells - buys
                    if False:  # False for SVM regression test
                        label_iterations(df, buys, sells, holds)
                if extract is None:
                    extract = df
                else:
                    extract = samples_concat(extract, df)
            return extract

        for base in self.baselist:
            sym = base + "_usdt"
            tf = load_classifier_features(self.tf_aggs, self.target_key, base)
            if tf is None:
                print(f"no datafile found for {sym}")
                continue
            tdf = extract_set_type_targets(sym, tf, TRAIN)
            self.trainctrl = samples_concat(self.trainctrl, tdf)
            vdf = extract_set_type_targets(sym, tf, VAL)
            self.valctrl = samples_concat(self.valctrl, vdf)
            tstdf = extract_set_type_targets(sym, tf, TEST)
            self.testctrl = samples_concat(self.testctrl, tstdf)

    def sets_config_fname(self, target_key):
        cfname = DATA_PATH + "/target_" + str(target_key) +"_sets_split.config"
        return cfname

    def analyze_bases(self, target_key, aggs):
        """Analyses the baselist and creates a config file of sets split into equal timeblock
        length. Additionally an artificial 'total' set is created with the sum of all bases for
        each timeblock.
        """

        def first_week_ts(tf):
            df = tf.tf_vectors.vec(tf.target_key)
            assert df is not None
            first_tic = df.index[0].to_pydatetime()
            last_tic = df.index[len(df.index)-1].to_pydatetime()
            if self.fixtic is None:
                self.fixtic = last_tic
            fullweeks = math.ceil((self.fixtic - first_tic) / timedelta(minutes=self.timeblock))
            assert fullweeks > 0, \
                   f"{tf.cur_pair} {len(df)} {len(df.index)} {first_tic} {last_tic} {fullweeks}"
            wsts = self.fixtic - timedelta(minutes=fullweeks*self.timeblock-1)
            wets = wsts + timedelta(minutes=self.timeblock-1)
            return (wsts, wets)

        def next_week_ts(tf, wets):
            wsts = wets + timedelta(minutes=1)
            wets = wsts + timedelta(minutes=self.timeblock-1)
            df = tf.tf_vectors.vec(tf.target_key)
            assert df is not None
            last_tic = df.index[len(df.index)-1].to_pydatetime()
            if last_tic < wsts:
                return (None, None)
            else:
                return (wsts, wets)  # first and last tic of a block

        def analyze_timeblock(tf, wsts, wets, sym):
            dft = tf.tf_aggs[tf.target_key]
            dfm = tf.tf_aggs[1]
            assert dft is not None
            assert dfm is not None
            dft = dft.loc[(dft.index >= wsts) & (dft.index <= wets), ['target']]
            buys = len(dft.loc[dft.target == TARGETS[BUY]])
            sells = len(dft.loc[dft.target == TARGETS[SELL]])
            dfm = dfm.loc[(dfm.index >= wsts) & (dfm.index <= wets), ['volume']]
            vcount = len(dfm)
            avgvol = int(dfm['volume'].mean())
            novol = len(dfm.loc[dfm.volume <= 0])
            lastix = len(self.analysis)
            self.analysis.loc[lastix] = [sym, NA, wsts, wets, vcount, buys, sells, avgvol, novol]

        blocks = set()
        for base in self.baselist:
            sym = base + "_usdt"
            tf = load_classifier_features(aggs, target_key, base)
            (wsts, wets) = first_week_ts(tf)
            while wets is not None:
                blocks.add(wets)
                analyze_timeblock(tf, wsts, wets, sym)
                (wsts, wets) = next_week_ts(tf, wets)
        blocklist = list(blocks)
        blocklist.sort()
        for wets in blocklist:
            df = self.analysis.loc[self.analysis.end == wets]
            buys = int(df['buys'].sum())
            sells = int(df['sells'].sum())
            avgvol = int(df['avg_vol'].mean())
            novol = int(df['novol_count'].sum())
            vcount = int(df['tics'].sum())
            wsts = wets - timedelta(minutes=self.timeblock)
            lastix = len(self.analysis)
            sym = 'total'
            self.analysis.loc[lastix] = [sym, NA, wsts, wets, vcount, buys, sells, avgvol, novol]
        cfname = self.sets_config_fname(target_key)
        self.analysis.to_csv(cfname, sep='\t', index=False)



#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
#    currency_data = FeaturesTargets()
#    print(currency_data.performances())
