#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:40:09 2019

@author: tc

"""


import os
import pandas as pd
import pickle

class CurrencyPairs2Classifier:
    """converts historic catalyst data of currency pairs into classifier target and feature vectors
    and stores them for classifier training and evaluation
    """

cur_cand = ['xrp_usdt', 'btc_usdt']
data_keys = ['open', 'high', 'low', 'close', 'volume'] # , 'price'


def __init__(self, catalyst_currency_pairs: DataFrame):
    """converts historic catalyst data of currency pairs into classifier target and feature vectors
    and stores them for classifier training and evaluation
    """
    test = currencies = dict()
    filename = os.getcwd() + '/df-test.pydata'

    for pair in catalyst_currency_pairs:
        current = data.history(symbol(pair), data_keys, 239*24*60, '1T')
        currencies[pair] = current
        print(current.head())
        print(current.tail())

    print("got catalyst history data")
    df_f = open(filename, 'wb')
    pickle.dump(currencies, df_f)
    df_f.close()
    print("data frame is written")
#    df_f = open(filename, 'rb')
#    test = pickle.load(df_f)
#    df_f.close()
#    print(test)
    return None

def feature_normalize(filename: str):
    currencies = dict()
    df_f = open(filename, 'rb')
    currencies = pickle.load(df_f)
    df_f.close()
#    combined_curr = combine_catalyst_data(currencies)
    aggregate_currencies = pair_aggregation(currencies)
    return None
