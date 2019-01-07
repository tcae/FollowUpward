#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""

import pandas as pd

agg_minutes = [1, 2] # minutes = T
time_aggregations = {'1T': 4, '2T': 4} # , '4T':4
minute_data = pd.DataFrame() # required to use minute data in apply
vol_base_period = '1D'
sell_threshold = -2 # in per mille
buy_threshold = 10 # in per mille
transaction_fee = 1 # in per mille, i.e. 0,1%
best_n = 10

def check_tag(tg, ltg):
    return abs(tg) == abs(ltg)

def add_period_specific_labels(df: pd.DataFrame):
    "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

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
            if loss < sell_threshold: # reset win monitor because dip exceeded threshold
                win = 0.
                if lastlabel != "sell": # only one signal without further repeat
                    df.iat[lossix, lix] = lastlabel = "sell"
                    last_close = df.iat[lossix-1, cix]
                    df.iat[lossix, pix] = (df.iat[lossix, cix] - last_close) / last_close * 1000 # - transaction_fee
                lossix += 1
            if win > 0: # win monitoring is running
                win += delta
                if win < 0: # reset win monitor because it is below start price
                    win = 0.
        elif delta > 0:
            if win > 0: # win monitoring is running
                win += delta
            else: # first time bar of increase period
                winix = tix
                win = delta
            if win > buy_threshold: # reset win monitor because dip exceeded threshold
                if lastlabel != "buy": # only one signal without further repeat
                    df.iat[winix, lix] = lastlabel = "buy"
                    last_close = df.iat[winix-1, cix]
                    df.iat[winix, pix] = (df.iat[winix, cix] - last_close) / last_close * 1000 # - transaction_fee
                winix += 1
            if loss < 0: # loss monitoring is running
                loss += delta
                if loss > 0:
                    loss = 0. # reset loss monitor as it recovered before before triggered sell threshold

#    print(df[['close', 'label', 'change']])




def derive_features(df: pd.DataFrame):
    "calc derived candle features in relation to price based on the provided time aggregated dataframe df"
    # price changes in 1/1000
    df['height'] = (df['high'] - df['low']) / df['close'] * 1000
    df.loc[df['close'] > df['open'], 'top'] = (df['high'] - df['close']) / df['close'] * 1000
    df.loc[df['close'] <= df['open'], 'top'] = (df['high'] - df['open']) / df['close'] * 1000
    df.loc[df['close'] > df['open'], 'bottom'] = (df['open'] - df['low']) / df['close'] * 1000
    df.loc[df['close'] <= df['open'], 'bottom'] = (df['close'] - df['low']) / df['close'] * 1000

def time_aggregation(minute_df: pd.DataFrame):
    """in: dataframe of minute data of a currency pair;
       out: dict of dataframes of aggregations with features and targets"""
    aggregations = dict()
    time_aggs = list(time_aggregations.keys())
    for time_agg in time_aggs:
        if time_agg is '1T':
            df = minute_df
            df['volume_change'] = (df['volume']  - df.volume.rolling(vol_base_period).median()) / df.volume.rolling(vol_base_period).median() * 100 # in %
        else:
            df = pd.DataFrame()
            df['close'] = minute_df.close.resample(time_agg, label='right', closed='right').last()
            df['high'] = minute_df.high.resample(time_agg, label='right', closed='right').max()
            df['low'] = minute_df.low.resample(time_agg, label='right', closed='right').min()
            df['open'] = minute_df.open.resample(time_agg, label='right', closed='right').first()
            df['volume_change'] = minute_df.volume_change.resample(time_agg, label='right', closed='right').mean()
        derive_features(df)
        add_period_specific_labels(df)
        aggregations[time_agg] = df
    aggregations['CPC'] = add_asset_summary_labels(aggregations)
    return aggregations

def next_pairs_level(pairs: pd.DataFrame, level):
    "build pairs of pairs"
    paired = False
    first = pairs.loc[pairs.lvl < level] # childs can be on different levels!
    if not first.empty:
        for fp in first.index:
            second = first.loc[(first.bts > first.at[fp, 'sts'])] # & (((first.status == 'init') & (first.at[fp, 'status'] == 'init')) != True)] # if both are best == True then pair already exists
            for sp in second.index:
                if first.loc[(first.child1 == fp) & (first.child2 == sp)].empty: # otherwise pair already exists
                    paired = True
                    pairs = pairs.append(dict([('bts', first.at[fp, 'bts']), ('sts', second.at[sp, 'sts']), ('lvl', level), ('perf', first.at[fp, 'perf'] + second.at[sp, 'perf']), ('child1', fp), ('child2', sp)]), ignore_index=True)
        if paired:
            pairs = next_pairs_level(pairs, level + 1)
    return pairs

def calc_performance(row):
    return (minute_data.at[row.sts, 'close'] - minute_data.at[row.bts, 'close']) / minute_data.at[row.bts, 'close'] * 1000 - 2 * transaction_fee

def mark_childs(pairs, bix):
    pairs.at[bix, 'best'] = True
    if pairs.at[bix, 'lvl'] > 0:
        mark_childs(pairs, pairs[bix, 'child1'])
        mark_childs(pairs, pairs[bix, 'child2'])


def add_asset_summary_labels(aggregations: dict):
    "target = achieved if improvement > 1% without intermediate loss of more than 0.2%"

    global minute_data
    minute_data = aggregations['1T']
    max_period = '1T'
    for p in iter(aggregations.keys()):
        if aggregations[max_period].index[0].freq < aggregations[p].index[0].freq:
            max_period = p
    start_ts = minute_data.index[0]
    sell_ixs = aggregations[max_period].loc[aggregations[max_period].label == 'sell']
    pairs = pd.DataFrame()
    for end_ts in sell_ixs.index:
        for p in iter(aggregations.keys()):
            buy_sigs = aggregations[p].loc[(aggregations[p].label == 'buy') & (aggregations[p].index >= start_ts) & (aggregations[p].index < end_ts)]
            for bs in buy_sigs.index:
                sell_sigs = aggregations[p].loc[(aggregations[p].label == 'sell') & (aggregations[p].index <= end_ts) & (aggregations[p].index > bs)]
                for s in sell_sigs.index:
                    pairs = pairs.append(dict([('bts', bs), ('sts', s), ('lvl', 0)]), ignore_index=True)
        pairs['perf'] = pairs.apply(calc_performance, axis=1)

        # 1st level of pairs created
        pairs['best'] = False
        pairs = next_pairs_level(pairs, 1) # recursively create all pair levels

        # now select only those pairs that are part of the best n paths and continue to work with those
        best_perf = pairs.nlargest(max(best_n, len(pairs.index)), 'perf')
        st = best_perf.nsmallest(1, 'sts')
        start_ts = st.at[st.index[0], 'sts']
        for bix in best_perf.index:
            mark_childs(pairs, bix)
        pairs = pairs.loc[pairs.best] # reduce pairs to best_n path pairs
    best_perf = pairs.nlargest(1, 'perf')
    pairs['best'] = False
    mark_childs(pairs, best_perf.index[0])
    pairs = pairs.loc[pairs.best & (pairs.lvl == 0)] # reduce pairs to best_n path pairs on level 0
    check_result_consistency(aggregations, pairs)
    cpc_labels = pd.DataFrame(minute_data, columns=['close'])
    cpc_labels['label'] = '-'
    for bix in pairs.index:
        if cpc_labels.at[pairs.at[bix, 'bts'], 'label'] != '-':
            print('error buy: inconsistency due to unexpected value {0} instead of hold at timestamp'.format(cpc_labels.at[pairs.at[bix, 'bts'], 'label']))
            print(pairs.at[bix, 'bts'])
        else:
            cpc_labels.at[pairs.at[bix, 'bts'], 'label'] = 'buy'

        if cpc_labels.at[pairs.at[bix, 'sts'], 'label'] != '-':
            print('error sell: inconsistency due to unexpected value {} instead of hold at timestamp'.format(cpc_labels.at[pairs.at[bix, 'sts'], 'label']))
            print(pairs.at[bix, 'sts'])
        else:
            cpc_labels.at[pairs.at[bix, 'sts'], 'label'] = 'sell'
    return cpc_labels

def check_result_consistency(aggregations, pairs):
    "consistency checks"
    perf = time_aggregations.copy()
    for perf_elem in iter(perf.keys()):
        perf[perf_elem] = 0
    buy_list = sell_list = list()
    best_perf = 0.
    tdf = aggregations['1T']
    pairs.sort_values(by=['lvl', 'bts'])
    laststs = pd.Timestamp(2000) # earlier than any real data
    for p in pairs.index:
        bts = pairs.at[p, 'bts']
        sts = pairs.at[p, 'sts']
        check_perf = (tdf.at[sts, 'close'] - tdf.at[bts, 'close']) / tdf.at[bts, 'close'] * 1000 - 2 * transaction_fee
        if check_perf > best_perf:
            best_perf = check_perf

        if bts >= sts:
            print(f"error: intra pair sequence {bts} >= {sts} incorrect")

        if pairs.at[p, 'lvl'] == 0:
            if p != pairs.index[0]:
                if bts <= laststs:
                    print("error: level 0 pairs sequence between pairs incorrect")
            laststs = sts

            if bts in buy_list:
                print("error: double buy in pairs")
            else:
                buy_list.append(bts)
            if sts in sell_list:
                print("error: double sell in pairs")
            else:
                sell_list.append(sts)
        elif pairs.at(p, 'lvl') > 0:
            if (pairs.loc[(pairs.at[p, 'bts'] == pairs.at[pairs.at[p, 'child1'], 'bts'])].empty) or (pairs.loc[(pairs.at[p, 'sts'] == pairs.at[pairs.at[p, 'child2'], 'sts'])].empty):
                print("error: can't find consistent childs")
        else:
            print("error: unexpected level in pairs")
    for agg in iter(aggregations.keys()):
        tdf = aggregations[agg]
        lastclose = 0.
        sigs = tdf.loc[(tdf.label == 'buy') | (tdf.label == 'sell')]
        missed_sell_start = 0
        missed_buy_end = -1
        for sig in sigs.index:
            if sigs.at[sig, 'label'] == 'buy':
                if lastclose == 0.:
                    lastclose = sigs.at[sig, 'close']
                    missed_buy_end = 1
                else:
                    missed_buy_end += 1 # buy is following buy
            elif sigs.at[sig, 'label'] == 'sell':
                if lastclose > 0.:
                    perf[agg] = (sigs.at[sig, 'close'] - lastclose) / lastclose * 1000 - 2 * transaction_fee
                    lastclose = 0.
                    missed_buy_end = 0
                else:
                    if missed_buy_end < 0: # no buy signal yet seen
                        missed_sell_start += 1
                    else:
                        pass # sell is following sell
            else:
                print("error: unexpected signal - neither buy nor sell")
        if missed_buy_end > 0:
            print('info: missed {} buy signals at the end'.format(missed_buy_end))
        if missed_sell_start > 0:
            print('info: missed {} sell signals at the start'.format(missed_sell_start))

#        bsigs = tdf.loc[(tdf.label == 'buy')]
#        check = bsigs.loc[buy_list]
#        if (len(bsigs.index) - len(check.index)) > missed_buy_end :
#            print("error: missing {} buy signals in pairs".format(len(bsigs.index) - len(check.index) - missed_buy_end))

#        ssigs = tdf.loc[(tdf.label == 'sell')]
#        check = ssigs.loc[sell_list]
#        if (len(ssigs.index) - len(check.index)) > missed_sell_start :
#            print("error: missing {} sell signals in pairs".format(len(ssigs.index) - len(check.index) - missed_sell_start))

    print('performances')
    print(f'best: {best_perf}')
    for agg in iter(aggregations.keys()):
        print(f'{agg}: {perf[agg]}')
        if perf[agg] > best_perf:
            print(f'error: single time aggrgation {agg} performance {perf[agg]} exceeds global best performance {best_perf}')


def pair_aggregation(currencies):
    "transform dict of currency dataframes to dict of currency dicts with all time aggregations"
    for pair in currencies:
        cur = currencies[pair] # take 1T currency data
        currencies[pair] = time_aggregation(cur) # exchange by all required time aggregations
    return currencies



def test_features_labels():
    "tests creation of features and labels with artificial data"
    df_len = 21
    df = pd.DataFrame(index=pd.date_range('2018-12-28 01:10:00', periods=df_len, freq='T'))
    cl = 100.
    cl_delta = 1.1 / 5
    df['open'] = 0.
    df['high'] = 0.
    df['low'] = 0.
    df['close'] = 0.
    df['volume'] = 10.

    for tf in range(0, df_len):
        df.iloc[tf] = [cl- 1., cl + 0.5, cl - 2., cl, 10.]
        if tf <= 4: #raise above 1% to trigger buy
            cl += cl_delta
        elif tf <= 5: # fall -0.2% to trigger sell but only on minute basis
            cl -= cl_delta
            df.iloc[tf, 4] = 20.
        elif tf <= 9: # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta
        elif tf <= 13: # raise above 1% with dip above -0.2% to not raise a trigger
            cl -= cl_delta / 4
        elif tf <= 30: # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta

    currencies = dict()
    currencies['tst_usdt'] = df
    return currencies


#if __name__ == "__main__":
#    import sys
#    fib(int(sys.argv[1]))
aggregate_currencies = pair_aggregation(test_features_labels())
