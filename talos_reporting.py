#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:27:59 2019

@author: tc
"""

import talos as ta

import matplotlib.pyplot as plt


r = Reporting('experiment_log.csv')

# returns the results dataframe
r.data

# returns the highest value for 'val_fmeasure'
r.high('val_fmeasure')

# returns the number of rounds it took to find best model
r.rounds2high()

# draws a histogram for 'val_acc'
r.plot_hist()