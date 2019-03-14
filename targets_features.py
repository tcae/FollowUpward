#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
# import numpy as np
import pandas as pd
from datetime import datetime
import math
import json
import re
from sklearn.utils import Bunch
import numpy as np
import pickle

# DATA_PATH = os.getcwd() # local execution - to be avoided due to Git sync size
DATA_PATH = '/Users/tc/Features'  # local execution
# DATA_PATH = '/content/gdrive/My Drive/Features' # Colab execution

PICKLE_EXT = ".pydata"  # pickle file extension
JSON_EXT = ".json"  # msgpack file extension
MSG_EXT = ".msg"  # msgpack file extension

FEE = 1  # in per mille, transaction fee is 0.1%
BUY_THRESHOLD = 10  # in per mille
SELL_THRESHOLD = -2  # in per mille
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
TIME_AGGS = {CPC: 0, 1: 10, 5: 10, 15: 10, 60: 10, 4*60: 10}
LBL = {NA:0, TRAIN:-1, VAL:-2, TEST:-3}


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

    def __init__(self, aggregation=TIME_AGGS, cur_pair=None):
        self.time_aggregations = aggregation  # aggregation in minutes of trailing DHTBV values
        self.performance = self.time_aggregations.copy()
        self.tf_aggs = dict()  # feature and target aggregations
        self.cur_pair = cur_pair

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
            else:
                self.tf_aggs[time_agg] = pd.DataFrame(self.minute_data.close)  # CPC dummy df

    def calc_features_and_targets(self, minute_dataframe):
        assert not minute_dataframe.empty, "empty dataframe"
        self.minute_data = minute_dataframe
        self.calc_aggregation()  # calculate features and time aggregations of features
        for time_agg in self.time_aggregations:
            if isinstance(time_agg, int):
                self.add_period_specific_targets(time_agg)  # add aggregation targets
        print(self.cpc_best_path())  # add summary targets for this currency pair
        # self.calc_performances() # calculate performances based on targets
        self.tf_vectors = TfVectors(tf=self, currency_pair=self.cur_pair)

    def add_period_specific_targets(self, time_agg):
        "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

        print(f"{datetime.now()}: add_period_specific_targets {time_agg}")
        df = self.tf_aggs[time_agg]
        df['delta'] = 0.
        df['target'] = TARGETS[HOLD]
        pix = df.columns.get_loc('delta')  # performance column index
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
            lasttarget[slot] = TARGETS[HOLD]
        for tix in range(time_agg, len(df), 1):  # tix = time index
            slot = (tix % time_agg)
            last_close = df.iat[tix - time_agg, cix]
            delta = (df.iat[tix, cix] - last_close) / last_close * 1000  # in per mille: 1% == 10
            df.iat[tix, pix] = delta
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
            tix_perf = ((this - last) / last * 1000)
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
            t_name = str(time_agg) + '_target'
            col_ix[time_agg] = cpc_df.columns.get_loc(t_name)
            assert col_ix[time_agg] > 0, f"did not find column {col_ix[time_agg]} of {time_agg}"
        close_ix = cpc_df.columns.get_loc('close')

        assert cpc_df.index.is_unique, "unexpected not unique index"
        last = cpc_df.iat[0, close_ix]
        for tix in range(len(cpc_df)):  # tix = time index
            this = cpc_df.iat[tix, close_ix]
            tix_perf = ((this - last) / last * 1000)
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
        self.performance = perf


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
        self.cur_pair = currency_pair
        self.vecs = dict()  # full fleged feature vectors and their targets
        self.data_version = 1.0
        self.filename = filename
        if tf is not None:
            if currency_pair is not None:
                print(f"{datetime.now()}: processing {self.cur_pair}")
            self.aggregations = tf.time_aggregations
            for ta in tf.time_aggregations:
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
        elif filename is not None:
            self.filename = None
            if filename.endswith(PICKLE_EXT):
                df_f = open(filename, 'rb')
                self.data_version, self.cur_pair, self.aggregations = pickle.load(df_f)
                self.vecs = pickle.load(df_f)
                print(f"{datetime.now()}: processing {self.cur_pair}")
                print("{}: read tf vectors with {} tics x {} aggregations from {}".format(
                         datetime.now(), len(self.vecs[CPC]), len(self.vecs), filename))
                df_f.close()
            elif filename.endswith(JSON_EXT):
                with open(filename, 'r') as df_f:
                    for line in df_f:
                        self.data_version, self.cur_pair, self.aggregations = json.loads(line)
                        break
                    for ta in self.aggregations:
                        self.vecs[ta] = pd.read_json(df_f)
                print(f"{datetime.now()}: processing {self.cur_pair}")
                print("{}: read tf vectors with {} tics x {} aggregations from {}".format(
                         datetime.now(), len(self.vecs[CPC]), len(self.vecs), filename))
                df_f.close()
            elif filename.endswith(MSG_EXT):
                df_f = open(filename, 'rb')
                self.data_version, self.cur_pair, self.aggregations = pickle.load(df_f)
                df_f.close()
                for ta in self.aggregations:
                    ext_fname = re.sub(MSG_EXT, '_'+str(ta)+MSG_EXT, filename)
                    self.vecs[ta] = pd.read_msgpack(ext_fname)
                print("{}: read tf vectors with {} tics x {} aggregations from {}".format(
                         datetime.now(), len(self.vecs[CPC]), len(self.vecs), filename))
            else:
                print(f"TfVectors init from file {filename}: unknown file extension")
        self.cut_back_to_same_sample_tics()

    def cut_back_to_same_sample_tics(self):
        """
            There will be a different number of feature vectors per aggregation due to the
            nan in the beginning rows that have missing history vectors before them.
            This leads to issues of different array lengths when constructing the CPC features
            out of the time aggregation specific probability result vectors, which is corrected
            with this function.
        """
        df = None
        for ta in self.aggregations:
            if df is None:
                df = self.vecs[ta]
            elif len(self.vecs[ta]) < len(df):
                df = self.vecs[ta]
#            print("agg {} with {} tics, first tic: {}  last tic: {}".format( \
#                  ta, len(self.vecs[ta]), \
#                  self.vecs[ta].index[0], self.vecs[ta].index[len(self.vecs[ta])-1]))
        for ta in self.aggregations:
            # h=a[a.index.isin(f.index)]
            self.vecs[ta] = self.vecs[ta][self.vecs[ta].index.isin(df.index)]
#            print("agg {} with {} tics, first tic: {}  last tic: {}".format( \
#                  ta, len(self.vecs[ta]), \
#                  self.vecs[ta].index[0], self.vecs[ta].index[len(self.vecs[ta])-1]))

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
            print("{}: writing tf vectors with {} tics ({} - {}) x {} aggregations to {}".format(
                     datetime.now(), len(self.vecs[CPC]), self.vecs[CPC].index[0],
                     self.vecs[CPC].index[len(self.vecs[CPC])-1], len(self.vecs), fname))
            self.filename = fname
            df_f = open(fname, 'wb')
            pickle.dump((self.data_version, self.cur_pair, self.aggregations), df_f)
            pickle.dump(self.vecs, df_f)
            df_f.close()
            print(f"{datetime.now()}: tf vectors saved")
        elif fname.endswith(JSON_EXT):
            "saves the object via json"
            print("{}: writing tf vectors with {} tics ({} - {}) x {} aggregations to {}".format(
                     datetime.now(), len(self.vecs[CPC]), self.vecs[CPC].index[0],
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
            print("{}: writing tf vectors with {} tics ({} - {}) x {} aggregations to {}".format(
                     datetime.now(), len(self.vecs[CPC]), self.vecs[CPC].index[0],
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

    def timeslice_targets(self, key, train_ratio=0.6, val_ratio=0.2, days=30):
        """Splits the data set into training set, validation set and test set such that
        training, validation and test sets are created that receive their time slice share
        of samples according to the given ratio.
        Days is the number of days of a single time slice of samples.

        Returns:
        ========
        The dataframe receives a column 'slice' that identifies the assignment of samples to
        training, validation, or test set

        """

        c_df = pd.DataFrame(self.vecs[key].target) #only use targets for filter logic
        c_df['slice'] = 0
        slc_col = c_df.columns.get_loc('slice')
        slice_end = c_df.index[0] + pd.Timedelta(days=days)
        slice_cnt = 1
        for t_ix in range(len(c_df)):
            if c_df.index[t_ix] < slice_end:
                c_df.iat[t_ix, slc_col] = slice_cnt
            else:
                slice_cnt += 1
                slice_end = c_df.index[t_ix] + pd.Timedelta(days=days)

        # now assign slices to training, validation and test set
        slices = {k: LBL[NA] for k in range(slice_cnt)}
        train_lvl = val_lvl = test_lvl = 0
        test_ratio = 1 - train_ratio - val_ratio
        ix = 0
        for ix in range(slice_cnt):
            train_lvl += train_ratio
            val_lvl += val_ratio
            test_lvl += test_ratio
            if train_lvl >= 0:
                slices[ix] = LBL[TRAIN]
                train_lvl -= 1
            elif val_lvl >= 0:
                slices[ix] = LBL[VAL]
                val_lvl -= 1
            elif test_lvl >= 0:
                slices[ix] = LBL[TEST]
                test_lvl -= 1
            assert slices[ix] != LBL[NA], "missing slice assignment"

        for k in range(slice_cnt):
            c_df.loc[(c_df['slice'] == k), 'slice'] = slices[k]
        return c_df

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
        # l_df = l_df.drop(['slice'], axis = 1) # not required as this is only a temp df
        h_count = len(l_df[l_df.target == TARGETS[HOLD]])
        s_count = len(l_df[l_df.target == TARGETS[SELL]])
        b_count = len(l_df[l_df.target == TARGETS[BUY]])
        smallest = min([h_count, b_count, s_count])
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

    def extract_subset(self, key, set_df, balance):
        subset_df = pd.DataFrame(set_df)
        self.reduce_target_sequences(subset_df, key)
        subset_df = subset_df[(subset_df.target == TARGETS[BUY])| \
                         (subset_df.target == TARGETS[SELL])| \
                         (subset_df.target == TARGETS[HOLD])]
        if balance:
            subset_df = self.balance_targets(subset_df)
        return subset_df

    def report_setsize(self, setname, df):
        hc = len(df[df.target == TARGETS[HOLD]])
        sc = len(df[df.target == TARGETS[SELL]])
        bc = len(df[df.target == TARGETS[BUY]])
        tc = hc + sc + bc
        print(f"buy {bc} sell {sc} hold {hc} total {tc} on {setname}")

    def timeslice_and_select_targets(self, key, train_ratio, val_ratio, balance, days):
        """Splits the data set into training set, validation set and test set such that
        training set receives the specified ratio of buy and sell samples and
        validation set also receives at least the specified ratio of buy and sell samples.
        sequences of buy or sell samples will be reduced every n of the sequence according
        to the seq_red value(local variable).
        days is the number of days of a slice

        Returns:
        ========
        dict with dataframes containing only targets for a training, a validation and a test set

        """

        c_df = self.timeslice_targets(key, train_ratio, val_ratio, days)
        train_df = c_df[c_df.slice == LBL[TRAIN]]
        val_df = c_df[c_df.slice == LBL[VAL]]
        test_df = c_df[c_df.slice == LBL[TEST]]

        self.report_setsize(f"{self.cur_pair} {str(key)} total", c_df)
        self.report_setsize(TRAIN, train_df)
        self.report_setsize(VAL, val_df)
        self.report_setsize(TEST, test_df)

        train_df = self.extract_subset(key, train_df, balance)
        self.report_setsize(f"{TRAIN} subset", train_df)

        val_df = self.extract_subset(key, val_df, balance)
        self.report_setsize(f"{VAL} subset", val_df)

        seq = {TRAIN: train_df, VAL: val_df, TEST: test_df}
        return seq


    def to_sklearn(self, df, np_data=None):
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
        target = df['target'].values # incompatible with pandas 0.19.2
        tics = df.index.values # incompatible with pandas 0.19.2
        feature_names = np.array(fn_list)
        target_names = np.array(TARGETS.keys())

        return Bunch(data=data, target=target,
                     target_names=target_names,
                     tics=tics,
                     descr=self.cur_pair,
                     feature_names=feature_names)


    def timeslice_data_to_sklearn(self, key, train_ratio, val_ratio, balance, days):
        """Load and return the crypto dataset (classification).
        """

        seq = self.timeslice_and_select_targets(key, train_ratio, val_ratio, balance, days)
        for elem in seq:
            # h=a[a.index.isin(f.index)] use only the data that is also in the target df
            data_df = self.vecs[key][self.vecs[key].index.isin(seq[elem].index)]
            seq[elem] = self.to_sklearn(data_df)
            seq[elem].descr = self.cur_pair + " aggregation: " + str(key)
        return seq

#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
#    currency_data = FeaturesTargets()
#    print(currency_data.performances())

