#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:54:13 2019

@author: tc
"""
# <codecell> rolling cell
import numpy as np
import pandas as pd

def rolling1():
    print("rolling1")
    a = np.arange(5)
    print(a)
    df = pd.DataFrame(a, columns=['a'],index = pd.date_range('2012-10-08 18:15:05', periods=5, freq='T'))
    print(df)
    c = df.rolling(3).apply(np.mean).dropna()
    print(c)

def rolling2():
    print("rolling2")
    a = pd.DataFrame(np.arange(5), columns=['a'],\
                     index = pd.date_range('2012-10-08 18:15:05', periods=5, freq='T'))
    print(a)
    b = a.rolling(3).mean().dropna()
    print(b)
    c = a.rolling(3).apply(np.mean).dropna()
    print(c)

def rolling3():
    print("rolling2")
    a = pd.DataFrame(np.arange(6), columns=['a'],\
                     index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
    print(a)
    b = a.rolling(3,center=True).mean().dropna()
    print(b)
    b = a.rolling(3,center=False).mean().dropna()
    print(b)
    b = a.rolling(3,closed='right').mean().dropna()
    print(b)
    b = a.rolling(3,closed='left').mean().dropna()
    print(b)
    b = a.rolling(3,closed='both').mean().dropna()
    print(b)
    b = a.rolling(3,closed='neither').mean().dropna()
    print(b)

def rolling4():
    print("rolling4")
    a = pd.DataFrame(np.arange(6), columns=['created'],\
                     index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
    a['mean'] = a.created.rolling(3).mean().dropna()
#    a['mean center'] = a.created.rolling(3,center=True).mean().dropna()
#    a['mean right'] = a.created.rolling(3,closed='right').mean().dropna()
#    a['mean left'] = a.created.rolling(3,closed='left').mean().dropna()
#    a['mean both'] = a.created.rolling(3,closed='both').mean().dropna()
#    a['mean neither'] = a.created.rolling(3,closed='neither').mean().dropna()
    a['shifted'] = a.created.shift(2)
    a['min'] = a.created.rolling(3).min()
    a['max'] = a.created.rolling(3).max()
    a['tshift'] = a.created.tshift(2)
    print(a)
    b = a.created.shift(2)
    print("shift")
    print(b)
    b = a.created.tshift(2)
    print("tshift")
    print(b)

def rolling5():
    print("rolling4")
    a = pd.DataFrame(np.arange(6), columns=['created'],\
                     index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
    a['mean'] = a.created.rolling('3T').mean().dropna()
#    a['shifted'] = a.created.shift('3T') # doesn't work with strings
    a['shifted'] = a.created.shift(2)
    a['min'] = a.created.rolling('3T').min()
    a['max'] = a.created.rolling('3T').max()
#    a['tshift'] = a.created.tshift('3T') # doesn't work with strings
    a['tshift'] = a.created.tshift(2)
    print(a)
    b = a.created.shift(2)
    print("shift")
    print(b)
    b = a.created.tshift(2)
    print("tshift")
    print(b)


print(f"panda version {pd.__version__}")
#rolling1()
#rolling2()
#rolling3()
#rolling4()
rolling5()

# <codecell> re
import re

def re_test():
    p = re.compile(r"(\d+)([THDW])")
    m = p.match("25T")
    nbr = int(m.group(1))
    s = m.group(2)
    print(f"found: {nbr} {s}")

    print(int(re.compile(r"(\d+)([T])").match("22T").group(1)))

re_test()

# <codecell> append
import pandas as pd
import numpy as np

def check_append():
    df= pd.DataFrame()
    #print(df)
    #print(df.dtypes)
    # ignore_index=True may reset the index with an append, e.g. when index has gaps
    df = df.append({'key': 'abc', 'hugo': int(2)}, ignore_index=True)
    df = df.append({'key': 'abc', 'hugo': 3, 'erich': 4.}, ignore_index=True)
    df = df.append({'key': 'abc', 'hugo': 5, 'erich': 4.}, ignore_index=True)
    df[3] = {'key': 'abc', 'hugo': 5, 'erich': 4.}
    print(df)

def check_iloc():
    df = pd.DataFrame(columns=["a","b","c","d"])
    df.set_index(['a', 'b'], inplace = True)

    df.loc[('3','4'),['c','d']] = [4,5]

    df.loc[('4','4'),['c','d']] = [3,1]
    print(df)

def check_iloc2():
    df = pd.DataFrame(columns=["a","b"])
    df = df.append({'a': 'hugo', 'b': 511}, ignore_index=True)
    df = df.append({'a': 'erich', 'b': 512}, ignore_index=True)
    df = df.append({'a': 'karl', 'b': 513}, ignore_index=True)
    print(df)
    df = df.iloc[[2,1]]
    print(df)
    df = df.append({'a': 'mimi', 'b': 514}, ignore_index=True)
    print(df)
    print("index was reset")

def check_loc3():
    df = pd.DataFrame(columns=["a","b"])
    df = df.append({'a': 'hugo', 'b': 511}, ignore_index=True)
    df = df.append({'a': 'erich', 'b': 512}, ignore_index=True)
    df = df.append({'a': 'karl', 'b': 513}, ignore_index=True)
    print(df)
    df = df.iloc[[2,1]]
    print(df)
    df.loc[4 ,['a', 'b']] = ['mimi', 514]
    print(df)
    print("index was reset")

def check_loc4():
    df = pd.DataFrame(np.arange(6).reshape(3,2), columns=['A','B'])
    print(df)
    df.loc[:,'C'] = df.loc[:,'A']
    print(df)
    df.loc[3] = 5
    print(df)

#check_append()
check_loc3()
