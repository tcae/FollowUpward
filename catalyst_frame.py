#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 19:04:56 2019

@author: tc

%%bash
catalyst ingest-exchange -x binance -i xrp_usdt -f minute
catalyst ingest-exchange -x binance -i btc_usdt -f minute

"""

import pytz
import pandas as pd
from datetime import datetime

from catalyst.utils.run_algo import run_algorithm
from catalyst.protocol import BarData
from catalyst.api import symbol
from targets_features import TargetsFeatures

import os
import pickle

def tdelta(first:str, last:str)-> int:
    my_min = pd.Timedelta(pd.Timestamp(last) - pd.to_datetime(first))
    min_res = my_min.days*24*60 + int(my_min.seconds/60)
    return min_res

CUR_CAND = ['xrp_usdt', 'btc_usdt']
DATA_KEYS = ['open', 'high', 'low', 'close', 'volume'] # , 'price'
classifier_input = dict()


def initialize(context):
    context.handle_count = 0
    print("init")


def handle_data(context, data: BarData):

    if (context.handle_count < 1):
        filename = os.getcwd() + '/'
        for pair in CUR_CAND:
            sy = symbol(pair)
            print(f"{datetime.now()}: reading {pair}")
            start = symbol(pair).start_date
            end = symbol(pair).end_minute
            min_count = tdelta(start, end)
            c_data = data.history(sy, DATA_KEYS, min_count, '1T')
            assert not c_data.empty, "{datetime.now()}: empty dataframe from Catalyst"
            print(f"{datetime.now()}: processing {pair}")
            tf = TargetsFeatures(c_data)
            classifier_input[pair] = tf.tf_vectors
            for ta in tf.tf_vectors:
                assert not tf.tf_vectors[ta].empty, "empty dataframe from TargetsFeatures"
            print(f"{datetime.now()}: processing {pair} is ready")
            filename = filename + f'{pair}-'
        filename = filename + 'classifier_input.pydata'
        df_f = open(filename, 'wb')
        pickle.dump(classifier_input, df_f)
        df_f.close()
        print("{datetime.now()}: data frames written to " + filename)

    context.handle_count += 1
    return None


def analyze(context=None, results=None):
    pass

start = datetime(2019, 1, 21, 0, 0, 0, 0, pytz.utc)
# end = datetime(2018, 9, 24, 0, 0, 0, 0, pytz.utc)
end = datetime(2019, 1, 21, 0, 0, 0, 0, pytz.utc)
results = run_algorithm(initialize=initialize,
                        handle_data=handle_data,
                        analyze=analyze,
                        start=start,
                        end=end,
                        exchange_name='binance',
                        data_frequency='minute',
                        quote_currency ='usdt',
                        capital_base=10000 )