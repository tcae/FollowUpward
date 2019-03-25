#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 21:13:46 2019

@author: tc
"""
# import os
import pandas as pd
import targets_features as t_f


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

    for tix in range(0, df_len):
        df.iloc[tix] = [cl - 1., cl + 0.5, cl - 2., cl, 10.]
        if tix <= 4:  # raise above 1% to trigger buy
            cl += cl_delta
        elif tix <= 5:  # fall -0.2% to trigger sell but only on minute basis
            cl -= cl_delta
            df.iloc[tix, 4] = 20.
        elif tix <= 9:  # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta
        elif tix <= 13:  # raise above 1% with dip above -0.2% to not raise a trigger
            cl -= cl_delta / 4
        elif tix <= 30:  # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta

    cl -= 2
    for tix in range(df_len, 2 * df_len):
        df.iloc[tix] = [cl - 1., cl + 0.5, cl - 2., cl, 10.]
        if tix <= 4+df_len:  # raise above 1% to trigger buy
            cl += cl_delta
        elif tix <= 5+df_len:  # fall -0.2% to trigger sell but only on minute basis
            cl -= cl_delta
            df.iloc[tix, 4] = 20.
        elif tix <= 9+df_len:  # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta
        elif tix <= 13+df_len:  # raise above 1% with dip above -0.2% to not raise a trigger
            cl -= cl_delta / 4
        elif tix <= 30+df_len:  # raise above 1% with dip above -0.2% to not raise a trigger
            cl += cl_delta
    cl -= 1.2
    df.iloc[2 * df_len] = [cl - 1., cl + 0.5, cl - 2., cl, 10.]

    return df

# content of test_tmpdir.py
# def test_needsfiles(tmpdir):
#    print(tmpdir)
#    assert 0


def test_fl():
    "regression test performance returns of targets and features based on artificial input"
    pair = 'tst_usdt'
    print("tests started")
    agg = {'CPC': 0, 1: 4, 2: 4}
    df = generate_minute_data()
    cp = t_f.TargetsFeatures(aggregation=agg, cur_pair=pair)
    cp.calc_features_and_targets(df)
    cp.calc_performances()
    test = cp.performance
    print(test)
#    print(cp.cpc_performance)
    assert test[1] == -0.23782522317685562
    assert test[2] == 6.2407515870716175
    assert test['CPC'] == 25.304288700980557

#    fname = t_f.DATA_PATH + '/' + pair + t_f.PICKLE_EXT
    fname = t_f.DATA_PATH + '/' + pair + t_f.MSG_EXT
#    fname = t_f.DATA_PATH + '/' + pair + t_f.JSON_EXT
    cp.tf_vectors.save(fname)
    ncp_tfv = t_f.TfVectors(filename=fname)
    print(ncp_tfv.data_version)
    print(ncp_tfv.aggregations)
    print(ncp_tfv.pair_name())
    print("tests finished")


test_fl()
