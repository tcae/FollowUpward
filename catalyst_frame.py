#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 19:04:56 2019

@author: tc

%%bash
catalyst ingest-exchange -x binance -i xrp_usdt -f minute
catalyst ingest-exchange -x binance -i btc_usdt -f minute

"""

#import os
from datetime import datetime
import pytz
import pandas as pd

from catalyst.utils.run_algo import run_algorithm
from catalyst.protocol import BarData
from catalyst.api import symbol
import targets_features as t_f


def tdelta(first: str, last: str)-> int:
    "returns date/time difference in minutes"
    my_min = pd.Timedelta(pd.Timestamp(last) - pd.to_datetime(first))
    min_res = my_min.days*24*60 + int(my_min.seconds/60)
    return min_res

CUR_CAND = ['xrp_usdt', 'btc_usdt', 'eth_usdt', 'bnb_usdt', 'eos_usdt', 'ltc_usdt', 'neo_usdt', 'trx_usdt']
DATA_KEYS = ['open', 'high', 'low', 'close', 'volume'] # , 'price'
# classifier_input = dict()


def initialize(context):
    "catalyst init; context.handle_count is used to handle first handle_data special"
    context.handle_count = 0
    print("init")


def handle_data(context, data: BarData):
    "called every minute by Catalyst framework"

    if context.handle_count < 1:
        for pair in CUR_CAND:
            sy = symbol(pair)
            print(f"{datetime.now()}: reading {pair}")
            startdate = symbol(pair).start_date
            enddate = symbol(pair).end_minute
            min_count = tdelta(startdate, enddate)
            c_data = data.history(sy, DATA_KEYS, min_count, '1T')
            assert not c_data.empty, "{datetime.now()}: empty dataframe from Catalyst"
            tf = t_f.TargetsFeatures(c_data, cur_pair=pair)
            test = tf.performance
            print(test)
            tf.tf_vectors.save(t_f.DATA_PATH + '/' + pair + '.msg')
            print(f"{datetime.now()}: processing {pair} is ready")

    context.handle_count += 1


#def analyze(context=None, results=None):
#    "catalyst framework aftermath callback"
#    pass

daily_perf = run_algorithm(initialize=initialize,
                           handle_data=handle_data,
                           # analyze=analyze,
                           start=datetime(2019, 2, 28, 0, 0, 0, 0, pytz.utc),
                           end=datetime(2019, 2, 28, 0, 0, 0, 0, pytz.utc),
                           exchange_name='binance',
                           data_frequency='minute',
                           quote_currency='usdt',
                           capital_base=10000)
print(f"daily performance: {daily_perf}")
