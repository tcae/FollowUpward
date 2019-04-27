#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 19:04:56 2019

@author: tc

%%bash
catalyst ingest-exchange -x binance -i xrp_usdt -f minute
catalyst ingest-exchange -x binance -i btc_usdt -f minute

"""

# import os
from datetime import datetime
import pytz
import pandas as pd

from catalyst.utils.run_algo import run_algorithm
from catalyst.protocol import BarData
from catalyst.api import symbol
import targets_features as t_f


def tdelta(first: str, last: str) -> int:
    "returns date/time difference in minutes"
    my_min = pd.Timedelta(pd.Timestamp(last) - pd.to_datetime(first))
    min_res = my_min.days*24*60 + int(my_min.seconds/60)
    return min_res

USDT_SUFFIX = 'usdt'
BTC_SUFFIX = 'btc'
CUR_CAND = ['xrp_', 'eth_', 'bnb_', 'eos_', 'ltc_', 'neo_', 'trx_']
DATA_KEYS = ['open', 'high', 'low', 'close', 'volume']  # , 'price'
# classifier_input = dict()


def initialize(context):
    "catalyst init; context.handle_count is used to handle first handle_data special"
    context.handle_count = 0
    print("init")

def load_pair(data, pair):
    sy = symbol(pair)
    startdate = symbol(pair).start_date
    enddate = symbol(pair).end_minute
    min_count = tdelta(startdate, enddate)
    c_data = data.history(sy, DATA_KEYS, min_count, '1T')
    print("{}: reading {} first: {} last: {} minutes: {}".format(
        datetime.now().strftime(t_f.DT_FORMAT), pair, startdate.strftime(t_f.DT_FORMAT),
        enddate.strftime(t_f.DT_FORMAT), min_count))
    assert not c_data.empty, "{datetime.now().strftime(t_f.DT_FORMAT)}: empty dataframe from Catalyst"
    return c_data

def check_diff(cur_btc_usdt, cur_usdt):
    """ cur_usdt should be a subset of cur_btc_usdt. This function checks the average deviation.
    """
    print("c-u first: {}  last: {}".format(
        cur_usdt.index[0].strftime(t_f.DT_FORMAT),
        cur_usdt.index[len(cur_usdt)-1].strftime(t_f.DT_FORMAT)))
    print("c-b-u first: {}  last: {}".format(
        cur_btc_usdt.index[0].strftime(t_f.DT_FORMAT),
        cur_btc_usdt.index[len(cur_btc_usdt)-1].strftime(t_f.DT_FORMAT)))
    diff = cur_btc_usdt[cur_btc_usdt.index.isin(cur_usdt.index)]
    for key in DATA_KEYS:
        if key != 'volume':
            diff[key] = (cur_btc_usdt[key] - cur_usdt[key]) / cur_usdt[key]
        else:
            diff[key] = cur_btc_usdt[key] - cur_usdt[key]
        diff_average = diff[key].sum() / len(cur_usdt)
        if key != 'volume':
            print(f"check_diff {key}: {diff_average:%}")
        else:
            print(f"check_diff {key}: {diff_average}")


def handle_data(context, data: BarData):
    "called every minute by Catalyst framework"

    if context.handle_count < 1:
        btcusdt = load_pair(data, 'btc_usdt')
        t_f.save_asset_dataframe(btcusdt, t_f.DATA_PATH, 'btc_usdt')

        for pair in CUR_CAND:
            cb_pair = pair + BTC_SUFFIX
            cbtc = load_pair(data, cb_pair)
            cbtcusdt = pd.DataFrame(btcusdt)
            cbtcusdt = cbtcusdt[cbtcusdt.index.isin(cbtc.index)]
            for key in DATA_KEYS:
                if key != 'volume':
                    cbtcusdt[key] = cbtc[key] * btcusdt[key]
            cbtcusdt['volume'] = cbtc.volume
            cbu_pair = pair + BTC_SUFFIX + '_' + USDT_SUFFIX

            cu_pair = pair + USDT_SUFFIX
            cusdt = load_pair(data, cu_pair)
            t_f.save_asset_dataframe(cusdt, t_f.DATA_PATH, cu_pair)

            cbtcusdt.loc[cusdt.index,:] = cusdt[:]  # take values of cusdt where available
            # check_diff(cbtcusdt, cusdt)
            t_f.save_asset_dataframe(cbtcusdt, t_f.DATA_PATH, cbu_pair)

    if False:  # context.handle_count < 1:
        btcusdt = load_pair(data, 'btc_usdt')
        tf = t_f.TargetsFeatures(cur_pair='btc_usdt')
        tf.calc_features_and_targets(btcusdt)
        tf.calc_performances()
        test = tf.performance
        for p in test:
            print(f"performance potential for aggregation {p}: {test[p]:%}")
        tf.tf_vectors.save(t_f.DATA_PATH + '/btc_usdt.msg')

        for pair in CUR_CAND:
            cb_pair = pair + BTC_SUFFIX
            cbtc = load_pair(data, cb_pair)
            cbtcusdt = pd.DataFrame(btcusdt)
            cbtcusdt = cbtcusdt[cbtcusdt.index.isin(cbtc.index)]
            for key in DATA_KEYS:
                if key != 'volume':
                    cbtcusdt[key] = cbtc[key] * btcusdt[key]
            cbtcusdt['volume'] = cbtc.volume
            cbu_pair = pair + BTC_SUFFIX + '_' + USDT_SUFFIX
            tf = t_f.TargetsFeatures(cur_pair=cbu_pair)
            tf.calc_features_and_targets(cbtcusdt)
            tf.calc_performances()
            test = tf.performance
            for p in test:
                print(f"performance potential for aggregation {p}: {test[p]:%}")
            tf.tf_vectors.save(t_f.DATA_PATH + '/' + cbu_pair + '.msg')

            cu_pair = pair + USDT_SUFFIX
            cusdt = load_pair(data, cu_pair)
            tf = t_f.TargetsFeatures(cur_pair=cu_pair)
            tf.calc_features_and_targets(cusdt)
            tf.calc_performances()
            test = tf.performance
            for p in test:
                print(f"performance potential for aggregation {p}: {test[p]:%}")
            tf.tf_vectors.save(t_f.DATA_PATH + '/' + cu_pair + '.msg')

            cbtcusdt.loc[cusdt.index,:] = cusdt[:]  # take values of cusdt where available
            # check_diff(cbtcusdt, cusdt)

            print(f"{datetime.now().strftime(t_f.DT_FORMAT)}: processing {pair} is ready")

    context.handle_count += 1


#def analyze(context=None, results=None):
#    "catalyst framework aftermath callback"
#    pass

daily_perf = run_algorithm(initialize=initialize,
                           handle_data=handle_data,
                           # analyze=analyze,
                           start=datetime(2019, 4, 1, 0, 0, 0, 0, pytz.utc),
                           end=datetime(2019, 4, 1, 0, 0, 0, 0, pytz.utc),
                           exchange_name='binance',
                           data_frequency='minute',
                           quote_currency='usdt',
                           capital_base=10000)
print(f"daily performance: {daily_perf}")
