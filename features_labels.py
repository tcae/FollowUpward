#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
import numpy as np
import pandas as pd

BUY = 'buy'
HOLD = '-'
SELL = 'sell'
FEE = 1 #  in per mille, transaction fee is 0.1%

class FeaturesLabels:
    """Receives a dict of currency pairs with associated minute candle data and
    transforms it into a dict of currency pairs with associated dicts of
    time_aggregations features. The time aggregation is the dict key with one
    special key 'CPC' that provides the summary labels

    Attributes
    ----------
    time_aggregations:
        dict with required time aggrefation keys and associated number
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



    """

    def __init__(self, currency_pair: str, minute_filename=None, minute_dataframe=None):
        assert (minute_filename is not None) or (minute_dataframe is not None), \
            "either filename or dataframe but not both"
        assert len(currency_pair) > 1, "unexpected empty string as currency pair"
        self.time_aggregations = {1: 4, 2: 4} # keys in minutes
        self.performance = self.time_aggregations.copy()
        self.cpc_performance = 0.
        self.minute_data = pd.DataFrame()
        self.fl_aggs = dict() # feature and label aggregations
        self.vol_base_period = '1D'
        self.sell_threshold = -2 # in per mille
        self.buy_threshold = 10 # in per mille
        self.best_n = 2 #10
        self.missed_buy_end = 0
        self.missed_sell_start = 0
        self.pmax_ix = 0

        if minute_filename is not None:
            pass
        elif not minute_dataframe.empty:
            self.minute_data = minute_dataframe
            self.pairs = pd.DataFrame()
            self.pairs.loc[0, 'bts'] = self.minute_data.index[0]
#            self.pairs.loc[0, 'sts'] = self.minute_data.index[len(self.minute_data.index)-1]
            self.pairs.loc[0, 'sts'] = self.minute_data.index[1]
            self.pairs.loc[0, 'lvl'] = int(0)
            self.pairs.loc[0, 'child1'] = int(0)
            self.pairs.loc[0, 'child2'] = int(0)
            self.pairs.loc[0, 'perf'] = 0.
            self.pairs.loc[0, 'best'] = False
            self.currency_pair = currency_pair
            self.time_aggregation()
            for time_agg in self.time_aggregations.keys():
                self.add_period_specific_labels(time_agg)
            print(self.cpc_best_path())
            self.calc_performances()
#            self.add_asset_summary_labels() # pairs approach
        else:
            print("warning: neither filename nor dataframe => no action")


    def add_period_specific_labels(self, time_agg):
#    def add_period_specific_labels_rolling(self, time_agg):
        "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

        df = self.fl_aggs[time_agg]
        df['change'] = 0.
        df['label'] = "-"
        pix = df.columns.get_loc('change') # performance column index
        lix = df.columns.get_loc('label')
        cix = df.columns.get_loc('close')
        win = dict()
        loss = dict()
        lossix = dict()
        winix = dict()
        lastlabel = dict()
        for slot in range(0, time_agg):
            win[slot] = loss[slot] = 0.
            winix[slot] = lossix[slot] = slot
            lastlabel[slot] = "-"
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
                if loss[slot] < self.sell_threshold: # reset win monitor because dip exceeded threshold
                    win[slot] = 0.
#                    if lastlabel[slot] != "sell": # only one signal without further repeat
                    df.iat[lossix[slot], lix] = lastlabel[slot] = "sell"
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
                        loss[slot] = 0. # reset loss monitor as it recovered before triggered sell threshold
                if win[slot] > self.buy_threshold: # reset win monitor because dip exceeded threshold
                    loss[slot] = 0.
#                    if lastlabel[slot] != "buy": # only one signal without further repeat
                    df.iat[winix[slot], lix] = lastlabel[slot] = "buy"
                    winix[slot] += 1 # allow multiple signals if conditions hold



    def derive_features(self, time_agg):
        "derived features in relation to price based on the provided time aggregated dataframe df"
        # price changes in 1/1000
        df = self.fl_aggs[time_agg]
        df['height'] = (df['high'] - df['low']) / df['close'] * 1000
        df.loc[df['close'] > df['open'], 'top'] = (df['high'] - df['close']) / df['close'] * 1000
        df.loc[df['close'] <= df['open'], 'top'] = (df['high'] - df['open']) / df['close'] * 1000
        df.loc[df['close'] > df['open'], 'bottom'] = (df['open'] - df['low']) / df['close'] * 1000
        df.loc[df['close'] <= df['open'], 'bottom'] = (df['close'] - df['low']) / df['close'] * 1000



#    def time_aggregation_rolling(self):
    def time_aggregation(self):
        """Time aggregation through rolling aggregation with the consequence that new data is
        generated every minute and even long period reflect all minute bumps in their features

        in:
            dataframe of minute data of a currency pair;
        out:
            dict of dataframes of aggregations with features and targets
        """
        time_aggs = list(self.time_aggregations.keys())
        for time_agg in time_aggs:
            if time_agg == 1:
                mdf = df = self.minute_data  # .copy()
                df['volume_change'] = (df['volume']  - \
                                      df.volume.rolling(self.vol_base_period).median()) / \
                                      df.volume.rolling(self.vol_base_period).median() * 100 # in %
            else:
                df = pd.DataFrame()
                df['close'] = mdf.close
                df['high'] = mdf.high.rolling(time_agg).max()
                df['low'] = mdf.low.rolling(time_agg).min()
                df['open'] = mdf.open.shift(time_agg-1)
                df['volume_change'] = mdf.volume_change.rolling(time_agg).mean()
            self.fl_aggs[time_agg] = df
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

        self.cpc_labels = pd.DataFrame(self.minute_data, columns=['close'])
        self.cpc_labels['label'] = HOLD
        assert self.cpc_labels.index.is_unique, "unexpected not unique index"
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
            for time_agg in self.time_aggregations.keys():
                assert self.fl_aggs[time_agg].index.contains(tic), "missing timestamp(resample on?)"
                signal = self.fl_aggs[time_agg].at[tic, 'label']
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
                                self.cpc_labels.at[buy_tic,'label'] = BUY
                                self.cpc_labels.at[sell_tic,'label'] = SELL
                            pot_transaction = False
                            holding = False
                            current_perf = transaction_perf = 0.
                    else: # use case 3 = no potential transaction
                        transaction_perf = current_perf - FEE
                        sell_tic = tic
                        pot_transaction = True # remains True
                        assert buy_tic < sell_tic, "inconsistent buy/sell tic marks"
                if buy_sig: # use case 4, from different periods there can be sell and buy!
                    if (current_perf < -FEE): # set new buy_tic but check for a potential tr.
                        if pot_transaction and (transaction_perf > 0): # execute transaction
                            assert buy_tic < sell_tic, \
                                f"inconsistent tic marks, buy: {buy_tic} sell: {sell_tic}"
                            best_perf += transaction_perf
                            self.cpc_labels.at[buy_tic,'label'] = BUY
                            self.cpc_labels.at[sell_tic,'label'] = SELL
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
        return best_perf


    def calc_performances(self):
        """calculate all time aggregation specific performances
        as well as the CPC summary best path performance.
        """

        self.performance = dict()
        ta_holding = dict()
        for time_agg in self.time_aggregations.keys():
            self.performance[time_agg] = 0.
            ta_holding[time_agg]= False
            self.fl_aggs[time_agg].loc[:,'perf'] = 0.
        self.cpc_labels.loc[:,'perf'] = 0.
        holding = False
        assert self.cpc_labels.index.is_unique, "unexpected not unique index"
        last = self.minute_data.at[self.minute_data.index[0], 'close']
        for tic in self.minute_data.index:
            this = self.minute_data.at[tic, 'close']
            tic_perf = ((this - last)/ last * 1000)
            last = this
            signal = self.cpc_labels.at[tic, 'label']
            if holding:
                self.cpc_performance += tic_perf
                self.cpc_labels.at[tic, 'perf'] = self.cpc_performance
            if (signal == BUY) and (not holding):
                self.cpc_performance -= FEE
                self.cpc_labels.at[tic, 'perf'] = self.cpc_performance
                holding = True
            if (signal == SELL) and holding:
                self.cpc_performance -= FEE
                self.cpc_labels.at[tic, 'perf'] = self.cpc_performance
                holding = False
            for time_agg in self.time_aggregations.keys():
                assert self.fl_aggs[time_agg].index.contains(tic), "missing timestamp(resample on?)"
                signal = self.fl_aggs[time_agg].at[tic, 'label']
                if ta_holding[time_agg]:
                    self.performance[time_agg] += tic_perf
                    self.fl_aggs[time_agg].at[tic, 'perf'] = self.performance[time_agg]
                if (signal == BUY) and (not ta_holding[time_agg]):
                    self.performance[time_agg] -= FEE
                    self.fl_aggs[time_agg].at[tic, 'perf'] = self.performance[time_agg]
                    ta_holding[time_agg] = True
                if (signal == SELL) and ta_holding[time_agg]:
                    self.performance[time_agg] -= FEE
                    self.fl_aggs[time_agg].at[tic, 'perf'] = self.performance[time_agg]
                    ta_holding[time_agg]= False


    def build_feature_vectors(self):
        print("rolling4")
        a = pd.DataFrame(np.arange(6), columns=['created'],\
                         index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
        a['s1'] = a.created.shift(1)
        a['s2'] = a.created.shift(2)
        a['s3'] = a.created.shift(3)
        a['t1'] = a.created.tshift(1)
        a['t2'] = a.created.tshift(2)
        a['t3'] = a.created.tshift(3)
        print(a)




#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
#    currency_data = FeaturesLabels()
#    print(currency_data.performances())
