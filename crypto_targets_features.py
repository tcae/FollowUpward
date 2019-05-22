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
TARGETS = {HOLD: 0, BUY: 1, SELL: 2}  # dict with int encoding of target labels
TARGET_NAMES = {0: HOLD, 1: BUY, 2: SELL}  # dict with int encoding of targets
TARGET_KEY = 5
TIME_AGGS = {1: 10, 5: 10, 15: 10, 60: 10, 4*60: 10, 24*60: 10}
# TIME_AGGS = {1: 10, 5: 10}
LBL = {NA: 0, TRAIN: -1, VAL: -2, TEST: -3}
# BASES = ['xrp', 'eos', 'bnb', 'btc', 'eth', 'neo', 'ltc', 'trx']
BASES = ['bnb', 'xrp']
QUOTE = 'usdt'
MANDATORY_ITERATIONS = 2  # number of iterations for the smallest class (in general BUY)


def timestr(ts=None):
    if ts is None:
        return datetime.now().strftime(DT_FORMAT)
    else:
        return pd.to_datetime(ts).strftime(DT_FORMAT)

def time_in_index(dataframe_with_timeseriesindex, tic):
    return True in dataframe_with_timeseriesindex.index.isin([tic])

class NoSubsetWarning(Exception):
    pass

def targets_to_features(tfv_ta_df, target_df):
    """Extracts a sample subset with targets and features of a specific time aggregation
    based on given targets. target_df and tfv_ta_df both have to share the same index basis.
    The index of target_df shall be a subset of tfv_ta_df.
    """
    df = tfv_ta_df[tfv_ta_df.index.isin(target_df.index)]
    l = len(target_df.index.difference(tfv_ta_df.index))
    if l > 0:
        raise NoSubsetWarning(f"subset with {l} rows that are not in superset")
    return df

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
        print(f"{timestr()} load_asset_dataframefile ERROR: cannot load {fname}")
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
    if dfbu is None:
        raise MissingHistoryData("Cannot load {}".format(fname))
    return dfbu

def report_setsize(setname, df):
    hc = len(df[df.target == TARGETS[HOLD]])
    sc = len(df[df.target == TARGETS[SELL]])
    bc = len(df[df.target == TARGETS[BUY]])
    tc = hc + sc + bc
    print(f"buy {bc} sell {sc} hold {hc} total {tc} on {setname}")

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
        descr = "missing description"

    return Bunch(data=data, target=target, close=close,
                 target_names=target_names,
                 tics=tics,
                 descr=descr,
                 feature_names=feature_names)



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

    def __init__(self, base, quote):
        """Receives the key attributes for feature generation

        aggregations:
            dict with required time aggregation keys and associated number
            of periods that shall be compiled in a corresponding feature vector
        target_key:
            has to be a key of aggregations. Targets are only calculated for that target_key


        """
        self.base = base
        self.quote = quote
        self.minute_data = None
        self.vec = None
        self.target_key = TARGET_KEY

    def cur_pair(self):
        return self.base + '_' + self.quote

    def load_classifier_features(self):
        try:
            df = load_asset_dataframe(DATA_PATH, self.base)
        except MissingHistoryData:
            raise
        else:
            self.calc_features_and_targets(df)

    def calc_features_and_targets(self, minute_dataframe):

        def derive_features(df):
            """derived features in relation to price based on the provided
            time aggregated dataframe df with the exception of the derived feature 'delta'
            that is calculated together with targets
            """
            # price deltas in 1/1000
            df['height'] = (df['high'] - df['low']) / df['close'] * 1000
            df.loc[df['close'] > df['open'],
                   'top'] = (df['high'] - df['close']) / df['close'] * 1000
            df.loc[df['close'] <= df['open'],
                   'top'] = (df['high'] - df['open']) / df['close'] * 1000
            df.loc[df['close'] > df['open'],
                   'bottom'] = (df['open'] - df['low']) / df['close'] * 1000
            df.loc[df['close'] <= df['open'],
                   'bottom'] = (df['close'] - df['low']) / df['close'] * 1000


        def calc_aggregation(minute_df, time_aggregations):
            """Time aggregation through rolling aggregation with the consequence that new data is
            generated every minute and even long time aggregations reflect all minute bumps in their
            features

            in:
                dataframe of minute data of a currency pair;
            out:
                dict of dataframes of aggregations with features and targets
            """
            tf_aggs = dict()  # feature and target aggregations
            mdf = df = minute_df  # .copy()
            mdf['vol'] = (mdf['volume'] - mdf.volume.rolling(VOL_BASE_PERIOD).mean()) \
                     / mdf.volume.rolling(VOL_BASE_PERIOD).mean()
            mdf = mdf.fillna(value={'vol': 0.000001})
            maxmin = 0
            for time_agg in time_aggregations:
                if isinstance(time_agg, int):
                    if (time_agg * time_aggregations[time_agg]) > maxmin:
                        maxmin = time_agg * time_aggregations[time_agg]
            if maxmin > len(df.index):
                raise MissingHistoryData("History data has {} samples but should have >= {}".format(
                        len(df.index), maxmin))
            for time_agg in time_aggregations:
                # print(f"{datetime.now()}: time_aggregation {time_agg}")
                if time_agg > 1:
                    df = pd.DataFrame()
                    df['open'] = mdf.open.shift(time_agg-1)
                    df['high'] = mdf.high.rolling(time_agg).max()
                    df['low'] = mdf.low.rolling(time_agg).min()
                    df['close'] = mdf.close
                    df['vol'] = mdf.vol.rolling(time_agg).mean()
                df['delta'] = (mdf.close - mdf.close.shift(time_agg)) / mdf.close.shift(time_agg)
                tf_aggs[time_agg] = df
                derive_features(df)
            return tf_aggs

        def expand_target_feature_vectors(tf_aggs, target_key):
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
            df = pd.DataFrame(tf_aggs[target_key], columns=['close', 'target'])
            skey = smallest_dict_key(tf_aggs)
            for ta in tf_aggs:
                for tics in range(TIME_AGGS[ta]):
                    ctitle = str(ta) + 'T_' + str(tics) + '_'
                    offset = tics*ta
                    # now add feature columns according to aggregation
                    df[ctitle + 'D'] = tf_aggs[ta].delta.shift(offset)
                    if ta == skey:  # full set only for smallest aggregation (minute data)
                        df[ctitle + 'H'] = tf_aggs[ta].height.shift(offset)
                        df[ctitle + 'T'] = tf_aggs[ta].top.shift(offset)
                        df[ctitle + 'B'] = tf_aggs[ta].bottom.shift(offset)
                        df[ctitle + 'V'] = tf_aggs[ta].vol.shift(offset)
                        df[ctitle + 'DV'] = tf_aggs[ta].vol.shift(offset) *\
                                                         tf_aggs[ta].delta.shift(offset)
            df = df.dropna()
            if df.empty:
                print("empty dataframe from expand_target_feature_vectors")
            return df

        def add_targets(time_agg, df):
            "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"
            #print(f"{datetime.now()}: add_targets {time_agg}")
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
                            loss[slot] = 0.  # reset loss monitor -> recovered before sell threshold
                    if win[slot] > BUY_THRESHOLD:  # reset win monitor -> dip exceeded threshold
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

        # here comes the core of calc_features_and_targets
        self.minute_data = minute_dataframe
        if minute_dataframe is None:
            return
        if minute_dataframe.empty is None:
            minute_dataframe = None
            return
        tf_aggs = calc_aggregation(self.minute_data, TIME_AGGS)
        add_targets(self.target_key, tf_aggs[self.target_key])  # add aggregation targets
        self.minute_data['target'] = tf_aggs[self.target_key]['target']
        self.vec = expand_target_feature_vectors(tf_aggs, self.target_key)


    def target_performance(self):
        """calculates the time aggregation specific performance of target_key
        """
        print(f"{datetime.now()}: calculate target_performance")
        target_df = self.minute_data
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


class HistorySets:
    """Container class for targets and features of a currency pair
    timeblock is the time window in minutes that is analyzed equally for all bases to
    avoid label leakage. timeblocks are then distributed to train, validate and test to
    balance buy and sell signals.
    """

    def __init__(self):
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
        self.hs_name = None
        self.baselist = BASES
        self.timeblock = 4*7*24*60  # time window in minutes that is analyzed equally for all bases
        self.fixtic = None  # tic as fixpoint for timeblock
        self.analysis = pd.DataFrame(columns=['sym', 'set_type', 'start', 'end', 'tics',
                                              'buys', 'sells', 'avg_vol', 'novol_count'])

        self.ctrl = dict()
        self.ctrl[TRAIN] = pd.DataFrame(columns=['sym', 'timestamp', 'target', 'use',
                                               'buy_prob', 'sell_prop', 'hold_prop',
                                               'iteration', 'tcount'])
        self.ctrl[VAL] = pd.DataFrame(columns=['sym', 'timestamp', 'target', 'use',
                                             'buy_prob', 'sell_prop', 'hold_prop'])
        self.ctrl[TEST] = pd.DataFrame(columns=['sym', 'timestamp', 'target', 'use',
                                              'buy_prob', 'sell_prop', 'hold_prop'])

        # state attributes
        self.iterations = 0  # actual training iterations
        self.max_iter = dict()  # max labeled iterations for all used samples per HOLD, BUY, SELL


        # for base in self.baselist:
        #     merge_asset_dataframe(DATA_PATH, base)
        self.load_sets_config(self.sets_config_fname())
        assert not self.analysis.empty, f"{timestr()}: missing sets config - generating it now"
        # self.analyze_bases()

    def set_of_type(self, base, set_type):
        sym = base + "_" + QUOTE
        try:
            base_df = self.ctrl[set_type].loc[(self.ctrl[set_type].sym == sym) &
                                                 (self.ctrl[set_type].use == True)]
            print(f"{set_type} set with {len(base_df)} samples for {sym}")
            return base_df
        except KeyError:
            print(f"no {self.set_type} set for {sym}")
            pass

    def trainset_iteration(self, base, iteration):
        sym = base + "_" + QUOTE
        try:
            hold_iter = iteration % self.max_iter[HOLD]
            buy_iter = iteration % self.max_iter[BUY]
            sell_iter = iteration % self.max_iter[SELL]
            base_df = self.ctrl[TRAIN].loc[(self.ctrl[TRAIN].sym == sym) &
                                              (((self.ctrl[TRAIN].target == TARGETS[HOLD]) &
                                              (self.ctrl[TRAIN].iteration == hold_iter)) |
                                              ((self.ctrl[TRAIN].target == TARGETS[BUY]) &
                                              (self.ctrl[TRAIN].iteration == buy_iter)) |
                                              ((self.ctrl[TRAIN].target == TARGETS[SELL]) &
                                              (self.ctrl[TRAIN].iteration == sell_iter))) &
                                              (self.ctrl[TRAIN].use == True)]
            report_setsize(f"{sym} {TRAIN} set iteration {iteration}", base_df)
            return base_df
        except KeyError:
            print(f"no {self.set_type} set for {sym}")
            pass

    def load_sets_config(self, config_fname):

        def use_settype_total():
            """Uses the set_type of 'total' and apllies it to all sets with such timeblock.
            """
            cdf = self.analysis.set_index('end')
            cdf['set_type'] = cdf.loc[cdf.sym == 'total']['set_type']
            # cdf['end'] = cdf.index  # is already doen by reset_index
            self.analysis = cdf.reset_index()

        # fname = '/Users/tc/tf_models/crypto' + '/' + 'sample_set_split.config'
        self.hs_name = config_fname
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

        def samples_concat(target, to_be_added):
            if target.empty:
                target = to_be_added
                # print("target empty --> target = to_be_added", target.head(), target.tail())
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

        def extract_set_type_targets(base, tf, set_type):
            sym = base + '_' + QUOTE
            try:
                print(f"extracting {set_type} for {sym}")
                dfcfg = self.analysis.loc[(self.analysis.set_type == set_type) &
                                          (self.analysis.sym == sym)]
            except KeyError:
                print(f"no {set_type} set for {sym}")
                return None
            dft = tf.minute_data
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
                if extract is None:
                    extract = df
                else:
                    extract = samples_concat(extract, df)
            return extract

        # here comes the core of prepare_training()
        for base in self.baselist:
            tf = TargetsFeatures(base, QUOTE)
            try:
                tf.load_classifier_features()
            except MissingHistoryData:
                continue
            tfv = tf.vec
            tdf = extract_set_type_targets(base, tf, TRAIN)
            tdf = tdf[tdf.index.isin(tfv.index)]
            self.ctrl[TRAIN] = samples_concat(self.ctrl[TRAIN], tdf)
            vdf = extract_set_type_targets(base, tf, VAL)
            vdf = vdf[vdf.index.isin(tfv.index)]
            self.ctrl[VAL] = samples_concat(self.ctrl[VAL], vdf)
            tstdf = extract_set_type_targets(base, tf, TEST)
            tstdf = tstdf[tstdf.index.isin(tfv.index)]
            self.ctrl[TEST] = samples_concat(self.ctrl[TEST], tstdf)

    def use_training_mistakes(self):
        df = self.ctrl[TRAIN]
        df['use'] = False
        df.loc[(df.target == TARGETS[HOLD]) &
               ((df.buy_prob >=  df.hold_prop) | (df.sell_prop >=  df.hold_prop)), 'use'] = True
        df.loc[(df.target == TARGETS[BUY]) &
               ((df.hold_prop >=  df.buy_prob) | (df.sell_prop >=  df.buy_prob)), 'use'] = True
        df.loc[(df.target == TARGETS[SELL]) &
               ((df.buy_prob >=  df.sell_prop) | (df.hold_prop >=  df.sell_prop)), 'use'] = True
        return len(df[df.use == True])

    def label_iterations(self):
        """Each sample is assigned to an iteration. The smallest class has
        MANDATORY_ITERATIONS, i.e. each sample is labeled
        with an iteration between 0 and MANDATORY_ITERATIONS - 1.
        The larger classes have more iteration labels according to
        their ratio with the smallest class. This is determined per sym,
        timeblock and target class.

        This sample distribution over iterations shall balance the training of classes
        not bias the classifier too much in an early learning cycle.
        """
        df = self.ctrl[TRAIN]
        holds = len(df[(df.target == TARGETS[HOLD]) & (df.use == True)])
        sells = len(df[(df.target == TARGETS[SELL]) & (df.use == True)])
        buys = len(df[(df.target == TARGETS[BUY]) & (df.use == True)])
        samples = holds + sells + buys
        print(f"buy {buys} sell {sells} hold {holds} total {samples} on {TRAIN}")
        min_iter = min([buys, sells, holds])
        b_iter = int(buys / min_iter * MANDATORY_ITERATIONS)
        self.max_iter[BUY] = b_iter
        s_iter = int(sells / min_iter * MANDATORY_ITERATIONS)
        self.max_iter[SELL] = s_iter
        h_iter = int(holds / min_iter * MANDATORY_ITERATIONS)
        self.max_iter[HOLD] = h_iter
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

    def sets_config_fname(self):
        cfname = DATA_PATH + "/target_5_sets_split.config"
        return cfname

    def analyze_bases(self):
        """Analyses the baselist and creates a config file of sets split into equal timeblock
        length. Additionally an artificial 'total' set is created with the sum of all bases for
        each timeblock.
        """

        def first_week_ts(tf):
            df = tf.vec
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
            df = tf.vec
            assert df is not None
            last_tic = df.index[len(df.index)-1].to_pydatetime()
            if last_tic < wsts:
                return (None, None)
            else:
                return (wsts, wets)  # first and last tic of a block

        def analyze_timeblock(tf, wsts, wets, sym):
            dfm = tf.minute_data
            assert dfm is not None
            dfm = dfm.loc[(dfm.index >= wsts) & (dfm.index <= wets), ['target', 'volume']]
            buys = len(dfm.loc[dfm.target == TARGETS[BUY]])
            sells = len(dfm.loc[dfm.target == TARGETS[SELL]])
            vcount = len(dfm)
            avgvol = int(dfm['volume'].mean())
            novol = len(dfm.loc[dfm.volume <= 0])
            lastix = len(self.analysis)
            self.analysis.loc[lastix] = [sym, NA, wsts, wets, vcount, buys, sells, avgvol, novol]

        blocks = set()
        for base in self.baselist:
            sym = base + "_usdt"
            tf = TargetsFeatures(base, QUOTE)
            tf.load_classifier_features()
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
        cfname = self.sets_config_fname()
        self.analysis.to_csv(cfname, sep='\t', index=False)



#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
#    currency_data = FeaturesTargets()
#    print(currency_data.performances())
