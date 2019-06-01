#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:27:59 2019

@author: tc
"""

import talos as ta

import matplotlib.pyplot as plt


r = ta.Reporting('xrp_eos_talos_test_001.csv')

# returns the results dataframe
r.data

# returns the highest value for 'val_acc'
print(f"high: {r.high()}")

# returns the number of rounds it took to find best model
print(f"rounds2high: {r.rounds2high()}")
print(r.correlate())
r.plot_corr()

# draws a histogram for 'val_acc'
r.plot_hist()

#high The highest result for a given metric
#
#rounds The number of rounds in the experiment
#
#rounds2high The number of rounds it took to get highest result
#
#low The lowest result for a given metric
#
#correlate A dataframe with Spearman correlation against a given metric
#
#plot_line A round-by-round line graph for a given metric
#
#plot_hist A histogram for a given metric where each observation is a permutation
#
#plot_corr A correlation heatmap where a single metric is compared against hyperparameters
#
#table A sortable dataframe with a given metric and hyperparameters
#
#best_params A dictionary of parameters from the best model
#
