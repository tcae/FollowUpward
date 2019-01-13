#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 18:57:44 2019

@author: tc
"""

class currency_pairs:
    cp_labels =['xrp_usdt'] # tst_usdt
    cp_dict = dict()
    
    def pair_aggregation(self):
        "transform dict of currency dataframes to dict of currency dicts with all time aggregations"
        self.features_labels = self.minute_data.copy()
        for pair in self.minute_data:
            self.features_labels[pair] = self.time_aggregation(pair) # exchange by all required time aggregations
            self.features_labels[pair][self.cpc_label_key] = self.add_asset_summary_labels(self.features_labels[pair])
        return self.features_labels

    def __init__(self, minute_filename=None):
        if minute_filename == None:
            self.cp_labels = ['tst_usdt']
        else:
            pass
