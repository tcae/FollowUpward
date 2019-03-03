#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
#import numpy as np
import pandas as pd
from datetime import datetime
import math
import json
import re

# DATA_PATH = os.getcwd() # local execution - to be avoided due to Git sync size
DATA_PATH = '/Users/tc/Features' # local execution
# DATA_PATH = '/content/gdrive/My Drive/Features' # Colab execution

PICKLE_EXT = ".pydata" # pickle file extension
JSON_EXT = ".json" # msgpack file extension
MSG_EXT = ".msg" # msgpack file extension

BUY = 'buy'
HOLD = '-'
SELL = 'sell'
FEE = 1 #  in per mille, transaction fee is 0.1%
BUY_THRESHOLD = 10 # in per mille
SELL_THRESHOLD = -2 # in per mille
VOL_BASE_PERIOD = '1D'
CPC = 'CPC'

def time_in_index(dataframe_with_timeseriesindex, tic):
    return True in dataframe_with_timeseriesindex.index.isin([tic])

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

    def __init__(self, minute_dataframe, aggregation=None, cur_pair=None):
        assert not minute_dataframe.empty, "empty dataframe"
        if aggregation:
            self.time_aggregations = aggregation
        else:
            self.time_aggregations = {CPC: 0, 1: 10, 5: 10, \
                                      15: 10, 60: 10, 4*60: 10}
            # keys is aggregation in minutes, value is nbr of trailing DHTBV values
        self.performance = self.time_aggregations.copy()
        self.minute_data = pd.DataFrame()
        self.tf_aggs = dict() # feature and target aggregations
        self.minute_data = minute_dataframe


        self.time_aggregation() # calculate features and time aggregations of features
        self.missed_buy_end = 0 # currently unused
        self.missed_sell_start = 0 # currently unused
        for time_agg in self.time_aggregations:
            if isinstance(time_agg, int):
                self.add_period_specific_targets(time_agg) # add aggregation targets
        print(self.cpc_best_path()) # add summary targets for this currency pair
        self.calc_performances() # calculate performances based on targets
        self.tf_vectors = TfVectors(tf=self, currency_pair=cur_pair)


    def add_period_specific_targets(self, time_agg):
#    def add_period_specific_targets_rolling(self, time_agg):
        "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

        print(f"{datetime.now()}: add_period_specific_targets {time_agg}")
        df = self.tf_aggs[time_agg]
        df['delta'] = 0.
        df['target'] = "-"
        pix = df.columns.get_loc('delta') # performance column index
        lix = df.columns.get_loc('target')
        cix = df.columns.get_loc('close')
        win = dict()
        loss = dict()
        lossix = dict()
        winix = dict()
        lasttarget = dict()
        for slot in range(0, time_agg):
            win[slot] = loss[slot] = 0.
            winix[slot] = lossix[slot] = slot
            lasttarget[slot] = "-"
#        for tix in range(((len(df)-1) % time_agg)+time_agg, len(df), time_agg): # tix = time index
        for tix in range(time_agg, len(df), 1): # tix = time index
            slot = (tix % time_agg)
            last_close = df.iat[tix - time_agg, cix]
            delta = (df.iat[tix, cix] - last_close) / last_close * 1000 #  in per mille: 1% == 10
            df.iat[tix, pix] = delta
            if delta < 0:
                if loss[slot] < 0: # loss monitoring is running
                    loss[slot] += delta
                else: # first time bar of decrease period
                    lossix[slot] = tix
                    loss[slot] = delta
                if win[slot] > 0: # win monitoring is running
                    win[slot] += delta
                    if win[slot] < 0: # reset win monitor because it is below start price
                        win[slot] = 0.
                if loss[slot] < SELL_THRESHOLD: # reset win monitor because dip exceeded threshold
                    win[slot] = 0.
#                    if lasttarget[slot] != "sell": # only one signal without further repeat
                    df.iat[lossix[slot], lix] = lasttarget[slot] = "sell"
                    lossix[slot] += 1 # allow multiple signals if conditions hold
            elif delta > 0:
                if win[slot] > 0: # win monitoring is running
                    win[slot] += delta
                else: # first time bar of increase period
                    winix[slot] = tix
                    win[slot] = delta
                if loss[slot] < 0: # loss monitoring is running
                    loss[slot] += delta
                    if loss[slot] > 0:
                        loss[slot] = 0. # reset loss monitor as it recovered before  sell threshold
                if win[slot] > BUY_THRESHOLD: # reset win monitor because dip exceeded threshold
                    loss[slot] = 0.
#                    if lasttarget[slot] != "buy": # only one signal without further repeat
                    df.iat[winix[slot], lix] = lasttarget[slot] = "buy"
                    winix[slot] += 1 # allow multiple signals if conditions hold



    def derive_features(self, time_agg):
        "derived features in relation to price based on the provided time aggregated dataframe df"
        # price deltas in 1/1000
        df = self.tf_aggs[time_agg]
        df['height'] = (df['high'] - df['low']) / df['close'] * 1000
        df.loc[df['close'] > df['open'], 'top'] = (df['high'] - df['close']) / df['close'] * 1000
        df.loc[df['close'] <= df['open'], 'top'] = (df['high'] - df['open']) / df['close'] * 1000
        df.loc[df['close'] > df['open'], 'bottom'] = (df['open'] - df['low']) / df['close'] * 1000
        df.loc[df['close'] <= df['open'], 'bottom'] = (df['close'] - df['low']) / df['close'] * 1000



#    def time_aggregation_rolling(self):
    def time_aggregation(self):
        """Time aggregation through rolling aggregation with the consequence that new data is
        generated every minute and even long time aggregations reflect all minute bumps in their
        features

        in:
            dataframe of minute data of a currency pair;
        out:
            dict of dataframes of aggregations with features and targets
        """
        for time_agg in self.time_aggregations:
            print(f"{datetime.now()}: time_aggregation {time_agg}")
            if isinstance(time_agg, int):
                if time_agg == 1:
                    mdf = df = self.minute_data  # .copy()
                    df['vol'] = (df['volume'] - df.volume.rolling(VOL_BASE_PERIOD).median()) \
                                 / df.volume.rolling(VOL_BASE_PERIOD).median()
                else:
                    df = pd.DataFrame()
                    df['close'] = mdf.close
                    df['high'] = mdf.high.rolling(time_agg).max()
                    df['low'] = mdf.low.rolling(time_agg).min()
                    df['open'] = mdf.open.shift(time_agg-1)
                    df['vol'] = mdf.vol.rolling(time_agg).mean()
                self.tf_aggs[time_agg] = df
                self.derive_features(time_agg)



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

        cpc_df = pd.DataFrame(self.minute_data, columns=['close'])
        cpc_df[CPC+'_target'] = HOLD
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
        pot_transaction = False # becomes true if potential buy and potential sell are detected
        transaction_perf = current_perf = best_perf = 0.
        buy_tix = sell_tix = 0
        last = cpc_df.iat[0, close_ix]
        for tix in range(len(cpc_df)): # tix = time index
            this = cpc_df.iat[tix, close_ix]
            tix_perf = ((this - last)/ last * 1000)
            last = this
            sell_sig = False
            buy_sig = False
            for six in col_ix:
                signal = cpc_df.iat[tix, col_ix[six]]
                if signal == BUY:
                    buy_sig = True
                if signal == SELL:
                    sell_sig = True
            if holding:
                current_perf += tix_perf
                if sell_sig:
                    if pot_transaction: # use case 2
                        if (current_perf - FEE) >= transaction_perf: # reset transaction
                            transaction_perf = current_perf - FEE
                            sell_tix = tix
                            pot_transaction = True # remains True
                        else:
                            if transaction_perf > 0: # execute transaction
#                                assert buy_tix < sell_tix, \
#                                    f"inconsistent tix marks, buy: {buy_tix} sell: {sell_tix}"
                                best_perf += transaction_perf
                                cpc_df.iat[buy_tix, target_ix] = BUY
                                cpc_df.iat[sell_tix, target_ix] = SELL
                            pot_transaction = False
                            holding = False
                            current_perf = transaction_perf = 0.
                    else: # use case 3 = no potential transaction
                        transaction_perf = current_perf - FEE
                        sell_tix = tix
                        pot_transaction = True # remains True
#                        assert buy_tix < sell_tix, "inconsistent buy/sell tix marks"
                if buy_sig: # use case 4, from different time aggs there can be sell and buy!
                    if current_perf < -FEE: # set new buy_tix but check for a potential tr.
                        if pot_transaction and (transaction_perf > 0): # execute transaction
#                            assert buy_tix < sell_tix, \
#                                f"inconsistent tix marks, buy: {buy_tix} sell: {sell_tix}"
                            best_perf += transaction_perf
                            cpc_df.iat[buy_tix, target_ix] = BUY
                            cpc_df.iat[sell_tix, target_ix] = SELL
                            pot_transaction = False
                            holding = False
                            transaction_perf = 0.
                        buy_tix = tix
                        current_perf = -FEE
                        holding = True
            else: # not holding
                if buy_sig: # use case 1 = not holding with buy signal
                    buy_tix = tix
                    current_perf = -FEE
                    holding = True
                    pot_transaction = False
        self.tf_aggs[CPC] = cpc_df
        return best_perf



    def calc_performances(self):
        """calculate all time aggregation specific performances
        as well as the CPC summary best path performance.
        """

        perf = dict()
        print(f"{datetime.now()}: calculate performances")
        cpc_df = self.tf_aggs[CPC]
        col_ix = dict()
        ta_holding = dict()
        for time_agg in self.time_aggregations:
            perf[time_agg] = 0.
            ta_holding[time_agg] = False
#            self.tf_aggs[time_agg].loc[:, 'perf'] = 0.
            t_name = str(time_agg) + '_target'
            col_ix[time_agg] = cpc_df.columns.get_loc(t_name)
            assert col_ix[time_agg] > 0, f"did not find column {col_ix[time_agg]} of {time_agg}"
        close_ix = cpc_df.columns.get_loc('close')

        assert cpc_df.index.is_unique, "unexpected not unique index"
        last = cpc_df.iat[0, close_ix]
        for tix in range(len(cpc_df)): # tix = time index
            this = cpc_df.iat[tix, close_ix]
            tix_perf = ((this - last)/ last * 1000)
            last = this
            for ta_ix in perf:
                signal = cpc_df.iat[tix, col_ix[ta_ix]]
                if ta_holding[ta_ix]:
                    perf[ta_ix] += tix_perf
#                    self.tf_aggs[time_agg].at[tic, 'perf'] = perf[time_agg]
                if (signal == BUY) and (not ta_holding[ta_ix]):
                    perf[ta_ix] -= FEE
#                    self.tf_aggs[time_agg].at[tic, 'perf'] = perf[time_agg]
                    ta_holding[ta_ix] = True
                if (signal == SELL) and ta_holding[ta_ix]:
                    perf[ta_ix] -= FEE
#                    self.tf_aggs[time_agg].at[tic, 'perf'] = perf[time_agg]
                    ta_holding[ta_ix] = False
        self.performance = perf



import pickle

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
    def __init__(self, tf=None, filename=None, currency_pair=None):
        """Builds a target and feature vector sequence with DHTBV feature sequences of
        n time steps (tics) as configured in time_aggregations in T units

        Result:
            a self.vecs dict with keys as in time_aggregations plus 'CPC' that
            are referring to DataFrames with feature vectors as rows. The column name indicates
            the type of feature, i.e. either 'target' or 'aggregation_tic_D|H|T|B|V'

            There will be a different number of feature vectors per aggregation due to the
            nan in the beginning rows that have missing history vectors before them.
        """
        self.cur_pair = currency_pair
        self.vecs = dict() # full fleged feature vectors and their targets
        self.data_version = 1.0
        self.filename = filename
        if tf is not None:
            if currency_pair is not None:
                print(f"{datetime.now()}: processing {self.cur_pair}")
            self.aggregations = tf.time_aggregations
            for ta in tf.time_aggregations:
                print(f"{datetime.now()}: build classifier vectors {ta}")
                t_name = str(ta) + '_target'
                self.vecs[ta] = pd.DataFrame(tf.minute_data, columns=['close', 'buy', 'sell'])
                self.vecs[ta]['buy'] = (tf.tf_aggs[CPC][t_name] == 'buy').astype(float)
                self.vecs[ta]['sell'] = (tf.tf_aggs[CPC][t_name] == 'sell').astype(float)
                for tics in range(tf.time_aggregations[ta]):
                    tgt = str(ta) + 'T_' + str(tics) + '_'
                    if isinstance(ta, int):
                        offset = tics*ta
                    else:
                        offset = tics
                    self.vecs[ta][tgt + 'D'] = tf.tf_aggs[ta].delta.shift(offset)
                    self.vecs[ta][tgt + 'H'] = tf.tf_aggs[ta].height.shift(offset)
                    self.vecs[ta][tgt + 'T'] = tf.tf_aggs[ta].top.shift(offset)
                    self.vecs[ta][tgt + 'B'] = tf.tf_aggs[ta].bottom.shift(offset)
                    self.vecs[ta][tgt + 'V'] = tf.tf_aggs[ta].vol.shift(offset)
                self.vecs[ta].dropna(inplace=True)
                assert not self.vecs[ta].empty, "empty dataframe from TargetsFeatures"
        elif filename is not None:
            self.filename = None
            if filename.endswith(PICKLE_EXT):
                df_f = open(filename, 'rb')
                self.data_version, self.cur_pair, self.aggregations = pickle.load(df_f)
                self.vecs = pickle.load(df_f)
                print(f"{datetime.now()}: processing {self.cur_pair}")
                print("{}: read tf vectors with {} tics x {} aggregations from {}".format( \
                         datetime.now(), len(self.vecs[CPC]), len(self.vecs), filename))
                df_f.close()
            elif filename.endswith(JSON_EXT):
#                df_f = open(filename, 'r')
                with open(filename) as df_f:
#                    self.data_version, self.cur_pair, self.aggregations = json.load(df_f)
                    for line in df_f:
                        self.data_version, self.cur_pair, self.aggregations = json.loads(line)
                        break
                    for ta in self.aggregations:
                        self.vecs[ta] = pd.read_json(df_f)
                print(f"{datetime.now()}: processing {self.cur_pair}")
                print("{}: read tf vectors with {} tics x {} aggregations from {}".format( \
                         datetime.now(), len(self.vecs[CPC]), len(self.vecs), filename))
                df_f.close()
            elif filename.endswith(MSG_EXT):
                df_f = open(filename, 'rb')
                self.data_version, self.cur_pair, self.aggregations = pickle.load(df_f)
                df_f.close()
                for ta in self.aggregations:
                    ext_fname = re.sub(MSG_EXT, '_'+str(ta)+MSG_EXT, filename)
                    self.vecs[ta] = pd.read_msgpack(ext_fname)
#                for ta in self.aggregations:
#                    self.vecs[ta] = pd.read_msgpack(df_f)
                print(f"{datetime.now()}: processing {self.cur_pair}")
                print("{}: read tf vectors with {} tics x {} aggregations from {}".format( \
                         datetime.now(), len(self.vecs[CPC]), len(self.vecs), filename))
            else:
                print(f"TfVectors init from file {filename}: unknown file extension")

    def vec(self, key):
        "Returns the dataframe of the given key"
        return self.vecs[key]

    def set_pair_name(self, currency_pair):
        "sets the currency pair name"
        self.cur_pair = currency_pair

    def pair_name(self):
        "returns the currency pair name"
        return self.cur_pair

    def save(self, fname):
        if fname.endswith(PICKLE_EXT):
            "saves the object via pickle"
            print("{}: writing tf vectors with {} tics ({} - {}) x {} aggregations to {}".format( \
                     datetime.now(), len(self.vecs[CPC]), self.vecs[CPC].index[0],\
                     self.vecs[CPC].index[len(self.vecs[CPC])-1], len(self.vecs), fname))
            self.filename = fname
            df_f = open(fname, 'wb')
            pickle.dump((self.data_version, self.cur_pair, self.aggregations), df_f)
            pickle.dump(self.vecs, df_f)
            df_f.close()
            print(f"{datetime.now()}: tf vectors saved")
        elif fname.endswith(JSON_EXT):
            "saves the object via json"
            print("{}: writing tf vectors with {} tics ({} - {}) x {} aggregations to {}".format( \
                     datetime.now(), len(self.vecs[CPC]), self.vecs[CPC].index[0],\
                     self.vecs[CPC].index[len(self.vecs[CPC])-1], len(self.vecs), fname))
            self.filename = fname
            df_f = open(fname, 'w')
            json.dump((self.data_version, self.cur_pair, self.aggregations), df_f)
#            json.dump(self.vecs, df_f)
            for ta in self.aggregations:
                self.vecs[ta].to_json(df_f)
            df_f.close()
            print(f"{datetime.now()}: tf vectors saved")
        elif fname.endswith(MSG_EXT):
            "saves the object via msgpack"
            print("{}: writing tf vectors with {} tics ({} - {}) x {} aggregations to {}".format( \
                     datetime.now(), len(self.vecs[CPC]), self.vecs[CPC].index[0],\
                     self.vecs[CPC].index[len(self.vecs[CPC])-1], len(self.vecs), fname))
            self.filename = fname
            df_f = open(fname, 'wb')
            pickle.dump((self.data_version, self.cur_pair, self.aggregations), df_f)
            df_f.close()
            for ta in self.aggregations:
                newext = '_'+str(ta)+MSG_EXT
                ext_fname = re.sub(MSG_EXT, newext, fname)
                self.vecs[ta].to_msgpack(ext_fname)
#            df_f = open(fname, 'wb')
#            pickle.dump((self.data_version, self.cur_pair, self.aggregations), df_f)
#            for ta in self.aggregations:
#                self.vecs[ta].to_msgpack(df_f, append=True)
#            df_f.close()
            print(f"{datetime.now()}: tf vectors saved")
        else:
            print("TfVectors save: unknown file extension")

    def signal_sequences(self, key):
        "provides a histogram of consecutive signals for the given key data"
        c_df = self.vecs[key]
        b_seq = dict()
        s_seq = dict()
        h_seq = dict()
        seq = {'buy': b_seq, 'sell': s_seq, 'hold': h_seq}
        b_count = s_count = h_count = 0
        b_col = c_df.columns.get_loc('buy')
        s_col = c_df.columns.get_loc('sell')
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


    def split_data(self, key, train_ratio=0.6, val_ratio=0.2, hold_ratio=1):
        """Splits the data set into training set, validation set and test set such that
        training set receives at least the specified ratio of buy and sell samples and
        validation set also receives at least the specified ratio of buy and sell samples.
        sequences of buy or sell samples will be reduced every n of the sequence according
        to the aggregation value, e.g. if 1:10 is specified then every 10th buy or sell sample
        is used from a consecutive sequence.
        hold_ratio is the ratio of hold / (sell + buy)

        Returns:
        ========
        dict with dataframes for a training, a validation and a test set

        Attention for possible label leakage:
        =====================================
        buy, sell and hold samples can overlap in time range
        """
        c_df = self.vecs[key]
        b_count = s_count = h_count = 0
        bt_count = st_count = 0
        bv_count = sv_count = seq_count = 0
#        agg = max(self.aggregations[key], 1) # use only every agg sample in signal sequences
        agg = 4 # use only every agg sample in signal sequences
        b_col = c_df.columns.get_loc('buy')
        s_col = c_df.columns.get_loc('sell')
        c_df['select'] = '-'
        sel_col = c_df.columns.get_loc('select')
        last = 'x'
        for t_ix in range(len(c_df)):
            b_sig = c_df.iat[t_ix, b_col]
            if b_sig > 0:
                if last == 'buy':
                    seq_count += 1
                else:
                    seq_count = 0
                    last = 'buy'
                if (seq_count % agg) == 0:
                    c_df.iat[t_ix, sel_col] = 'buy_select'
                    b_count += 1

            s_sig = c_df.iat[t_ix, s_col]
            if s_sig > 0:
                if last == 'sell':
                    seq_count += 1
                else:
                    seq_count = 0
                    last = 'sell'
                if (seq_count % agg) == 0:
                    c_df.iat[t_ix, sel_col] = 'sell_select'
                    s_count += 1

            if (b_sig == 0) and (s_sig == 0):
                if last == 'hold':
                    seq_count += 1
                else:
                    seq_count = 0
                    last = 'hold'
                if (seq_count % agg) == 0:
                    c_df.iat[t_ix, sel_col] = 'hold_select'
                    h_count += 1

        h_ratio = math.floor((h_count / (b_count + s_count)) / hold_ratio)
        h_count = 0
        for t_ix in range(len(c_df)):
            this = c_df.iat[t_ix, sel_col]
            if this == 'buy_select':
                if (bt_count / b_count) < train_ratio:
                    c_df.iat[t_ix, sel_col] = 'train'
                    bt_count += 1
                elif (bv_count / b_count) < val_ratio:
                    c_df.iat[t_ix, sel_col] = 'val'
                    bv_count += 1
                else:
                    c_df.iat[t_ix, sel_col] = 'test'
            elif this == 'sell_select':
                if (st_count / s_count) < train_ratio:
                    c_df.iat[t_ix, sel_col] = 'train'
                    st_count += 1
                elif (sv_count / s_count) < val_ratio:
                    c_df.iat[t_ix, sel_col] = 'val'
                    sv_count += 1
                else:
                    c_df.iat[t_ix, sel_col] = 'test'
            elif this == 'hold_select':
                if (h_count % h_ratio) == 0:
                    if ((bt_count + st_count) / (b_count + s_count)) < train_ratio:
                        c_df.iat[t_ix, sel_col] = 'train'
                    elif ((bv_count + sv_count) / (b_count + s_count)) < val_ratio:
                        c_df.iat[t_ix, sel_col] = 'val'
                    else:
                        c_df.iat[t_ix, sel_col] = 'test'
                h_count += 1

        train_df = c_df.loc[c_df['select'] == 'train']
        val_df = c_df.loc[c_df['select'] == 'val']
        test_df = c_df.loc[c_df['select'] == 'test']
        c_df.drop(['select'], axis = 1, inplace = True)
        train_df.drop(['select'], axis = 1, inplace = True)
        val_df.drop(['select'], axis = 1, inplace = True)
        test_df.drop(['select'], axis = 1, inplace = True)
        seq = {'training': train_df, 'validation': val_df, 'test': test_df}
        return seq



#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
#    currency_data = FeaturesTargets()
#    print(currency_data.performances())

