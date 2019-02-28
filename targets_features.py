#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
#import numpy as np
import pandas as pd
from datetime import datetime

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

    def __init__(self, minute_dataframe, aggregation=None):
        assert not minute_dataframe.empty, "empty dataframe"
        if aggregation:
            self.time_aggregations = aggregation
        else:
            self.time_aggregations = {'CPC': 0, 1: 10, 5: 10, \
                                      15: 10, 60: 10, 4*60: 10} # keys in minutes
        self.performance = self.time_aggregations.copy()
        self.minute_data = pd.DataFrame()
        self.tf_aggs = dict() # feature and target aggregations
        self.tf_vectors = dict() # full fleged feature vectors and their targets
        self.minute_data = minute_dataframe


        self.time_aggregation() # calculate features and time aggregations of features
        self.missed_buy_end = 0 # currently unused
        self.missed_sell_start = 0 # currently unused
        for time_agg in self.time_aggregations:
            if isinstance(time_agg, int):
                self.add_period_specific_targets(time_agg) # add aggregation targets
        print(self.cpc_best_path()) # add summary targets for this currency pair
        self.calc_performances() # calculate performances based on targets
        self.build_classifier_vectors()


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
                    df['vol'] = (df['volume']  - \
                                          df.volume.rolling(VOL_BASE_PERIOD).median()) / \
                                          df.volume.rolling(VOL_BASE_PERIOD).median() * 100 # in %
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
        cpc_df['target'] = HOLD
        assert cpc_df.index.is_unique, "unexpected not unique index"
        print(f"{datetime.now()}: best path")
        holding = False
        pot_transaction = False # becomes true if potential buy and potential sell are detected
        transaction_perf = current_perf = best_perf = 0.
        buy_tic = sell_tic = self.minute_data.index[0]
        last = self.minute_data.at[self.minute_data.index[0], 'close']
        for tic in self.minute_data.index:
            this = self.minute_data.at[tic, 'close']
            tic_perf = ((this - last)/ last * 1000)
            last = this
            sell_sig = False
            buy_sig = False
            for time_agg in self.time_aggregations:
                if isinstance(time_agg, int):
#                    assert self.tf_aggs[time_agg].index.contains(tic), "missing timestamp"
                    assert time_in_index(self.tf_aggs[time_agg], tic), "missing timestamp"
                    signal = self.tf_aggs[time_agg].at[tic, 'target']
                    if signal == BUY:
                        buy_sig = True
                    if signal == SELL:
                        sell_sig = True
            if holding:
                current_perf += tic_perf
                if sell_sig:
                    if pot_transaction: # use case 2
                        if (current_perf - FEE) >= transaction_perf: # reset transaction
                            transaction_perf = current_perf - FEE
                            sell_tic = tic
                            pot_transaction = True # remains True
                        else:
                            if transaction_perf > 0: # execute transaction
                                assert buy_tic < sell_tic, \
                                    f"inconsistent tic marks, buy: {buy_tic} sell: {sell_tic}"
                                best_perf += transaction_perf
                                cpc_df.at[buy_tic, 'target'] = BUY
                                cpc_df.at[sell_tic, 'target'] = SELL
                            pot_transaction = False
                            holding = False
                            current_perf = transaction_perf = 0.
                    else: # use case 3 = no potential transaction
                        transaction_perf = current_perf - FEE
                        sell_tic = tic
                        pot_transaction = True # remains True
                        assert buy_tic < sell_tic, "inconsistent buy/sell tic marks"
                if buy_sig: # use case 4, from different time aggs there can be sell and buy!
                    if current_perf < -FEE: # set new buy_tic but check for a potential tr.
                        if pot_transaction and (transaction_perf > 0): # execute transaction
                            assert buy_tic < sell_tic, \
                                f"inconsistent tic marks, buy: {buy_tic} sell: {sell_tic}"
                            best_perf += transaction_perf
                            cpc_df.at[buy_tic, 'target'] = BUY
                            cpc_df.at[sell_tic, 'target'] = SELL
                            pot_transaction = False
                            holding = False
                            transaction_perf = 0.
                        buy_tic = tic
                        current_perf = -FEE
                        holding = True
            else: # not holding
                if buy_sig: # use case 1 = not holding with buy signal
                    buy_tic = tic
                    current_perf = -FEE
                    holding = True
                    pot_transaction = False
        self.tf_aggs[CPC] = cpc_df
        return best_perf


    def calc_performances(self):
        """calculate all time aggregation specific performances
        as well as the CPC summary best path performance.
        """

        self.performance = dict()
        print(f"{datetime.now()}: calculate performances")
        ta_holding = dict()
        for time_agg in self.time_aggregations:
            self.performance[time_agg] = 0.
            ta_holding[time_agg] = False
            self.tf_aggs[time_agg].loc[:, 'perf'] = 0.
        last = self.minute_data.at[self.minute_data.index[0], 'close']
        for tic in self.minute_data.index:
            this = self.minute_data.at[tic, 'close']
            tic_perf = ((this - last)/ last * 1000)
            last = this
            for time_agg in self.time_aggregations:
#                assert self.tf_aggs[time_agg].index.contains(tic), "missing timestamp"
                assert time_in_index(self.tf_aggs[time_agg], tic), "missing timestamp"
                signal = self.tf_aggs[time_agg].at[tic, 'target']
                if ta_holding[time_agg]:
                    self.performance[time_agg] += tic_perf
                    self.tf_aggs[time_agg].at[tic, 'perf'] = self.performance[time_agg]
                if (signal == BUY) and (not ta_holding[time_agg]):
                    self.performance[time_agg] -= FEE
                    self.tf_aggs[time_agg].at[tic, 'perf'] = self.performance[time_agg]
                    ta_holding[time_agg] = True
                if (signal == SELL) and ta_holding[time_agg]:
                    self.performance[time_agg] -= FEE
                    self.tf_aggs[time_agg].at[tic, 'perf'] = self.performance[time_agg]
                    ta_holding[time_agg] = False


    def build_classifier_vectors(self):
        """Builds a target and feature vector sequence with DHTBV feature sequences of
        n time steps (tics) as configured in time_aggregations in T units

        Result:
            a self.tf_vectors dict with keys as in time_aggregations plus 'CPC' that
            are referring to DataFrames with feature vectors as rows. The column name indicates
            the type of feature, i.e. either 'target' or 'aggregation_tic_D|H|T|B|V'

            There will be a different number of feature vectors per aggregation due to the
            nan in the beginning rows that have missing history vectors before them.
        """
        for time_agg in self.time_aggregations:
            print(f"{datetime.now()}: build_classifier_vectors {time_agg}")
            self.tf_vectors[time_agg] = pd.DataFrame(self.minute_data, columns=['buy', 'sell'])
            self.tf_vectors[time_agg]['buy'] = \
                (self.tf_aggs[time_agg].target == 'buy').astype(float)
            self.tf_vectors[time_agg]['sell'] = \
                (self.tf_aggs[time_agg].target == 'sell').astype(float)
            tics = self.time_aggregations[time_agg]
            for tf in range(tics):
                tgt = str(time_agg) + 'T_' + str(tf) + '_'
                if isinstance(time_agg, int):
                    offset = tf*time_agg
                else:
                    offset = tf
                self.tf_vectors[time_agg][tgt + 'D'] = self.tf_aggs[time_agg].delta.shift(offset)
                self.tf_vectors[time_agg][tgt + 'H'] = self.tf_aggs[time_agg].height.shift(offset)
                self.tf_vectors[time_agg][tgt + 'T'] = self.tf_aggs[time_agg].top.shift(offset)
                self.tf_vectors[time_agg][tgt + 'B'] = self.tf_aggs[time_agg].bottom.shift(offset)
                self.tf_vectors[time_agg][tgt + 'V'] = self.tf_aggs[time_agg].vol.shift(offset)
            self.tf_vectors[time_agg].dropna(inplace=True)


#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
#    currency_data = FeaturesTargets()
#    print(currency_data.performances())
