#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:13:46 2019

@author: tc
"""
import pandas as pd
import targets_features as fl


def generate_minute_data():
    "tests creation of features and targets with artificial data"
    df_len = 21
    df = pd.DataFrame(index=pd.date_range('2018-12-28 01:10:00', periods=2*df_len+1, freq='T'))
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

    cl -= 2
    for tf in range(df_len, 2 * df_len):
        df.iloc[tf] = [cl- 1., cl + 0.5, cl - 2., cl, 10.]
        if tf <= 4+df_len: #raise above 1% to trigger buy
            cl += cl_delta
        elif tf <= 5+df_len: # fall -0.2% to trigger sell but only on minute basis
            cl -= cl_delta
            df.iloc[tf, 4] = 20.
        elif tf <= 9+df_len: # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta
        elif tf <= 13+df_len: # raise above 1% with dip above -0.2% to not raise a trigger
            cl -= cl_delta / 4
        elif tf <= 30+df_len: # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta
    cl -= 1.2
    df.iloc[2 * df_len] = [cl- 1., cl + 0.5, cl - 2., cl, 10.]

    return df

# content of test_tmpdir.py
#def test_needsfiles(tmpdir):
#    print(tmpdir)
#    assert 0

def test_fl():
    "regression test performance returns of targets and features based on artificial input"
    print("tests started")
    agg = {'CPC': 0, 1: 4, 2: 4}
    df = generate_minute_data()
    cp = fl.TargetsFeatures(minute_dataframe=df, aggregation=agg)
    if cp.missed_buy_end > 0:
        print('info: missed {} buy signals at the end'.format(cp.missed_buy_end))
    if cp.missed_sell_start > 0:
        print('info: missed {} sell signals at the start'.format(cp.missed_sell_start))
    t1 = cp.tf_aggs[1].loc[:,['delta', 'target', 'perf']]
    t2 = cp.tf_aggs[2].loc[:,['delta', 'target', 'perf']]
    tv1 = cp.tf_vectors[1]
    tv2 = cp.tf_vectors[2]
    tvcpc = cp.tf_vectors['CPC']
    cpc = cp.tf_aggs['CPC']
    test = cp.performance
    print(test)
#    print(cp.cpc_performance)
    assert test[1] == -0.23782522317685562
    assert test[2] == 6.2407515870716175
    assert test['CPC'] == 25.304288700980557
    print("tests finished")

test_fl()
