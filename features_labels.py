#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
import numpy as np
import pandas as pd


def derive_features(df: pd.DataFrame):
    "derived features in relation to price based on the provided time aggregated dataframe df"
    # price changes in 1/1000
    df['height'] = (df['high'] - df['low']) / df['close'] * 1000
    df.loc[df['close'] > df['open'], 'top'] = (df['high'] - df['close']) / df['close'] * 1000
    df.loc[df['close'] <= df['open'], 'top'] = (df['high'] - df['open']) / df['close'] * 1000
    df.loc[df['close'] > df['open'], 'bottom'] = (df['open'] - df['low']) / df['close'] * 1000
    df.loc[df['close'] <= df['open'], 'bottom'] = (df['close'] - df['low']) / df['close'] * 1000


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

    def add_period_specific_labels_resampling(self, time_agg):
#    def add_period_specific_labels(self, time_agg):
        "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

        df = self.fl_aggs[time_agg]
        df['change'] = 0.
        df['label'] = lastlabel = "-"
        pix = df.columns.get_loc('change') # performance column index
        lix = df.columns.get_loc('label')
        cix = df.columns.get_loc('close')
        win = loss = 0.
        for tix in range(1, len(df)): # tix = time index
            last_close = df.iat[tix-1, cix]
            delta = (df.iat[tix, cix] - last_close) / last_close * 1000 #  in per mille: 1% == 10
            df.iat[tix, pix] = delta
            if delta < 0:
                if loss < 0: # loss monitoring is running
                    loss += delta
                else: # first time bar of decrease period
                    lossix = tix
                    loss = delta
                if win > 0: # win monitoring is running
                    win += delta
                    if win < 0: # reset win monitor because it is below start price
                        win = 0.
                if loss < self.sell_threshold: # reset win monitor because dip exceeded threshold
                    win = 0.
                    if lastlabel != "sell": # only one signal without further repeat
                        df.iat[lossix, lix] = "sell"
                        lastlabel = "sell"
            elif delta > 0:
                if win > 0: # win monitoring is running
                    win += delta
                else: # first time bar of increase period
                    winix = tix
                    win = delta
                if loss < 0: # loss monitoring is running
                    loss += delta
                    if loss > 0:
                        loss = 0. # reset loss monitor as it recovered before triggered sell threshold
                if win > self.buy_threshold: # reset win monitor because dip exceeded threshold
                    loss = 0.
                    if lastlabel != "buy": # only one signal without further repeat
                        df.iat[winix, lix] = "buy"
                        lastlabel = "buy"




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
            derive_features(df)
            self.fl_aggs[time_agg] = df


    def time_aggregation_resampling(self):
#    def time_aggregation(self):
        """Time aggregation through downsampling with the consequence that new data is only
        generated after a specific period and overlooks minute bumps

        in:
            dataframe of minute data of a currency pair;
        out:
            dict of dataframes of aggregations with features and targets
        """
        global df, mdf
        time_aggs = list(self.time_aggregations.keys())
        for time_agg in time_aggs:
            mstr = str(time_agg) + "T"
            if time_agg == 1:
                mdf = df = self.minute_data
                df['volume_change'] = (df['volume']  - \
                                      df.volume.rolling(self.vol_base_period).median()) / \
                                      df.volume.rolling(self.vol_base_period).median() * 100 # in %
            else:
                mdf = self.minute_data
                df = pd.DataFrame()
                df['close'] = mdf.close.resample(mstr, label='right', closed='right').last()
                df['high'] = mdf.high.resample(mstr, label='right', closed='right').max()
                df['low'] = mdf.low.resample(mstr, label='right', closed='right').min()
                df['open'] = mdf.open.resample(mstr, label='right', closed='right').first()
                df['volume_change'] = mdf.volume_change.resample(mstr, label='right', \
                                                                       closed='right').mean()
            derive_features(df)
            self.fl_aggs[time_agg] = df

    def check_pairs(self):
        problem = self.pairs.isna()
        for pix in problem.index:
            if problem.at[pix, 'sts'] is True:
                print(f"problem at pix {pix} for sts")
            if problem.at[pix, 'bts'] is True:
                print(f"problem at pix {pix} for bts")
            if problem.at[pix, 'lvl'] is True:
                print(f"problem at pix {pix} for lvl")
            if problem.at[pix, 'child1'] is True:
                print(f"problem at pix {pix} for child1")
            if problem.at[pix, 'child2'] is True:
                print(f"problem at pix {pix} for child2")
            if problem.at[pix, 'perf'] is True:
                print(f"problem at pix {pix} for perf")
            if problem.at[pix, 'best'] is True:
                print(f"problem at pix {pix} for best")



    def combine_all_pairs(self):
        "build all pair combinations on level 0 between all time aggregations"

        first = self.pairs.loc[self.pairs.lvl == 0]
        if not first.empty:
            for fp in first.index:
                fp_bts = first.at[fp, 'bts']
                second = first.loc[(first.sts > first.at[fp, 'bts'])] # only a sell after the fp buy
                for sp in second.index:
                    sp_sts = second.at[sp, 'sts']
                    already_there = self.pairs.loc[(self.pairs.bts == fp_bts) & (self.pairs.sts == sp_sts)]
                    if already_there.empty: # otherwise pair already exists
                        self.pmax_ix += 1
                        self.pairs.loc[self.pmax_ix, \
                                       ['bts','sts','lvl','perf', 'child1', 'child2', 'best']] = \
                                       [ fp_bts, sp_sts, 0, 0, -2, -2, False]
        assert self.pairs.index.is_unique, "unexpected not unique index"


    def next_pairs_level(self, level):
        "build pairs of pairs"
        paired = False
        first = self.pairs.loc[self.pairs.lvl < level] # childs can be on different lower levels!
        if not first.empty:
            for fp in first.index:
                fp_bts = first.at[fp, 'bts']
                fp_perf = first.at[fp, 'perf']
                second = first.loc[(first.bts > first.at[fp, 'sts'])] # only pairs after each other
                for sp in second.index:
                    assert sp != fp, "unexpected detection of same pair"
                    sp_sts = second.at[sp, 'sts']
                    sp_perf = second.at[sp, 'perf']
                    already_there = first.loc[(first.child1 == fp) & (first.child2 == sp)]
                    try:
                        self.check_pairs()
                        if (self.pairs.at[fp, 'lvl'] >= level) or (self.pairs.at[sp, 'lvl'] >= level):
                            print("inconsistency")
                    except:
                        print("key error")
                        raise
#                    assert self.pairs.at[fp, 'lvl'] < level, "unexpected level"
#                    assert self.pairs.at[sp, 'lvl'] < level, "unexpected level"
                    if already_there.empty: # otherwise pair already exists
                        paired = True
                        self.pmax_ix += 1
                        self.pairs.loc[self.pmax_ix, \
                                       ['bts','sts','lvl','perf', 'child1', 'child2', 'best']] = \
                                       [ fp_bts, sp_sts, level, fp_perf + sp_perf, fp, sp, False]
            if paired:
                self.next_pairs_level(int(level + 1))
        assert self.pairs.index.is_unique, "unexpected not unique index"


    def calc_performance(self, row):
        "calculates the performance of each pair"
        diff = (self.minute_data.at[row.sts, 'close'] - self.minute_data.at[row.bts, 'close'])
        return diff / self.minute_data.at[row.bts, 'close'] * 1000 - 2 * self.transaction_fee

    def mark_childs(self, bix):
        "recursively marks all pairs involved in the best path with best=True"
        self.pairs.at[bix, 'best'] = True
        my_lvl = self.pairs.at[bix, 'lvl']
        if my_lvl > 0:
            c1_ix = int(self.pairs.at[bix, 'child1'])
            c1_lvl = self.pairs.at[c1_ix, 'lvl']
            c2_ix = int(self.pairs.at[bix, 'child2'])
            c2_lvl = self.pairs.at[c2_ix, 'lvl']
            assert my_lvl > c1_lvl, "inconsistent levels with child1"
            self.mark_childs(c1_ix)
            assert my_lvl > c2_lvl, "inconsistent levels with child2"
            self.mark_childs(c2_ix)


    def build_period_pairs(self, start_ts, end_ts):
        "builds level 0 pairs based on aggregated time data"
        for p in iter(self.fl_aggs.keys()):
            buy_sigs = self.fl_aggs[p].loc[(self.fl_aggs[p].label == 'buy') & \
                                   (self.fl_aggs[p].index >= start_ts) & \
                                   (self.fl_aggs[p].index < end_ts)]
            for bs in buy_sigs.index:
                sell_sigs = self.fl_aggs[p].loc[(self.fl_aggs[p].label == 'sell') & \
                                        (self.fl_aggs[p].index <= end_ts) & \
                                        (self.fl_aggs[p].index > bs)]
                for s in sell_sigs.index:
                    already_there = self.pairs.loc[(self.pairs.bts == bs) \
                                                   & (self.pairs.sts == s)& (self.pairs.lvl == 0)]
                    if already_there.empty: # otherwise pair already exists
                        self.pmax_ix += 1
                        self.pairs.loc[self.pmax_ix, \
                                       ['bts','sts','lvl','perf', 'child1', 'child2', 'best']] = \
                                       [ bs, s, 0, 0, -1, -1, False]


    def add_asset_summary_labels(self):
        "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

        max_period = 1
        for p in iter(self.fl_aggs.keys()):
            assert self.fl_aggs[p].index.is_unique, "unexpected not unique index"
            if self.fl_aggs[max_period].index[0].freq < self.fl_aggs[p].index[0].freq:
                max_period = p
        start_ts = self.minute_data.index[0]
        sell_ixs = self.fl_aggs[max_period].loc[self.fl_aggs[max_period].label == 'sell']
        for end_ts in sell_ixs.index:
            self.build_period_pairs(start_ts, end_ts)
            assert self.pairs.index.is_unique, "unexpected not unique index"
            self.combine_all_pairs()
            self.pairs['perf'] = self.pairs.apply(self.calc_performance, axis=1)

            # 1st level of pairs created
            self.next_pairs_level(int(1)) # recursively create all pair levels
            assert self.pairs.index.is_unique, "unexpected not unique index"

            # now select only those pairs that are part of the best n paths and work with those
            n = min(self.best_n, len(self.pairs.index))
            best_perf = self.pairs.nlargest(n, 'perf')
            assert best_perf.index.is_unique, "unexpected not unique index"
            assert min(self.best_n, len(self.pairs.index)) == len(best_perf.index), \
                "unexpected long best_perf data frame"
            st = best_perf.nsmallest(1, 'sts')
            assert st.index.is_unique, "unexpected not unique index"
            start_ts = st.iat[0, st.columns.get_loc('sts')]

            self.pairs['best'] = False
            for bix in best_perf.index:
                self.mark_childs(int(bix))
            self.pmax_ix = self.pairs.index.max()
            self.pairs = self.pairs.loc[self.pairs.best] # reduce pairs to best_n path pairs
            assert self.pairs.index.is_unique, "unexpected not unique index"

        best_perf = self.pairs.nlargest(1, 'perf')
        assert best_perf.index.is_unique, "unexpected not unique index"
        self.pairs['best'] = False
        self.mark_childs(int(best_perf.index[0]))
        self.check_result_consistency()
        self.pairs = self.pairs.loc[self.pairs.best & (self.pairs.lvl == 0)] # reduce pairs to best_n path pairs on level 0
        assert self.pairs.index.is_unique, "unexpected not unique index"
        cpc_labels = pd.DataFrame(self.minute_data, columns=['close'])
        cpc_labels['label'] = '-'
        for bix in self.pairs.index:
            assert cpc_labels.at[self.pairs.at[bix, 'bts'], 'label'] == '-', \
                'buy inconsistency due to unexpected value {0} instead of hold at timestamp'\
                .format(cpc_labels.at[self.pairs.at[bix, 'bts'], 'label'])
            cpc_labels.at[self.pairs.at[bix, 'bts'], 'label'] = 'buy'

            assert cpc_labels.at[self.pairs.at[bix, 'sts'], 'label'] == '-', \
                'error sell: inconsistency due to unexpected value {} instead of hold at timestamp'\
                .format(cpc_labels.at[self.pairs.at[bix, 'sts'], 'label'])
            cpc_labels.at[self.pairs.at[bix, 'sts'], 'label'] = 'sell'
        assert cpc_labels.index.is_unique, "unexpected not unique index"
        self.fl_aggs[self.cpc_label_key] = cpc_labels


    def check_period_consistency(self):
        "count non matching buy and sell signals and calculates the performance per period"
        for perf_elem in iter(self.performance.keys()):
            self.performance[perf_elem] = 0

        for agg in iter(self.fl_aggs.keys()):
            tdf = self.fl_aggs[agg]
            lastbuy_close = 0.
            sigs = tdf.loc[(tdf.label == 'buy') | (tdf.label == 'sell')]
            self.missed_sell_start = 0
            self.missed_buy_end = -1
            for sig in sigs.index:
                if sigs.at[sig, 'label'] == 'buy':
                    if lastbuy_close == 0.:
                        lastbuy_close = sigs.at[sig, 'close']
                        self.missed_buy_end = 1
                    else:
                        self.missed_buy_end += 1 # buy is following buy
                elif sigs.at[sig, 'label'] == 'sell':
                    if lastbuy_close > 0.:
                        sell_close = sigs.at[sig, 'close']
                        p = (sell_close - lastbuy_close) / \
                            lastbuy_close * 1000 - 2 * self.transaction_fee
                        self.performance[agg] += p
                        lastbuy_close = 0.
                        self.missed_buy_end = 0
                    else:
                        if self.missed_buy_end < 0: # no buy signal yet seen
                            self.missed_sell_start += 1
                        else:
                            pass # sell is following sell
                else:
                    assert False, ("unexpected signal - neither buy nor sell")


    def check_result_consistency(self):
        "consistency checks"

        best_perf = 0.
        tdf = self.fl_aggs[1]
        self.pairs.sort_values(by=['lvl', 'bts'])
        for p in self.pairs.index:
            bts = self.pairs.at[p, 'bts']
            sts = self.pairs.at[p, 'sts']
            check_perf = (tdf.at[sts, 'close'] - tdf.at[bts, 'close']) / tdf.at[bts, 'close'] \
                         * 1000 - 2 * self.transaction_fee
            if check_perf > best_perf:
                best_perf = check_perf

            assert bts < sts, f"intra pair sequence {bts} >= {sts} incorrect"
            assert self.pairs.at[p, 'lvl'] >= 0, "unexpectd level in pairs"

            if self.pairs.at[p, 'lvl'] > 0:
                assert (self.pairs.at[p, 'bts'] \
                        == self.pairs.at[int(self.pairs.at[p, 'child1']), 'bts']) and \
                       (self.pairs.at[p, 'sts'] \
                        == self.pairs.at[int(self.pairs.at[p, 'child2']), 'sts']), \
                    "can't find consistent childs"

        self.check_period_consistency()
        for perf_elem in iter(self.performance.keys()):
            assert self.performance[perf_elem] <= best_perf
        self.performance[self.cpc_label_key] = best_perf



    def __init__(self, currency_pair: str, minute_filename=None, minute_dataframe=None):
        assert (minute_filename is not None) or (minute_dataframe is not None), \
            "either filename or dataframe but not both"
        self.currency_pair = '?'
        self.cpc_label_key = 'CPC'
        self.time_aggregations = {1: 4, 2: 4} # keys in minutes
        self.performance = self.time_aggregations.copy()
        self.performance[self.cpc_label_key] = 0.
        self.minute_data = pd.DataFrame()
        self.fl_aggs = dict() # feature and label aggregations
        self.vol_base_period = '1D'
        self.sell_threshold = -2 # in per mille
        self.buy_threshold = 10 # in per mille
        self.transaction_fee = 1 # in per mille, i.e. 0,1%
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
            self.add_asset_summary_labels()
            t1 = self.fl_aggs[1]
            t2 = self.fl_aggs[2]
            print(f"T1 length: {len(t1.index)}, T2 length: {len(t2.index)}")
        else:
            print("warning: neither filename nor dataframe => no action")



#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
#    currency_data = FeaturesLabels()
#    print(currency_data.performances())
