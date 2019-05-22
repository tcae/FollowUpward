#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 09:54:13 2019

@author: tc
"""
# <codecell> iterator
class CtrlSet:

    def __init__(self, history_sets):
        self.hs = history_sets
        self.bases_iter = None
        self.base = None
        self.tfv = None

    def __iter__(self):
        self.bases_iter = iter(self.hs.baselist)
        return self

    def __next__(self):  # only python 2 uses next() without underscore
        try:
            if self.hs.ctrl[self.set_type].empty:
                print(f"no {self.set_type} set in HistorySets")
                raise StopIteration()
            while True:
                self.base = self.bases_iter.__next__()
        except StopIteration:
            raise StopIteration()


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

def rolling6():
    print("rolling4")
    a = pd.DataFrame(np.arange(6), columns=['created'],\
                     index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
    a['s1'] = a.created.shift(1)
    a['s2'] = a.created.shift(2)
    a['s3'] = a.created.shift(3)
    a['t1'] = a.created.tshift(1)
    a['t2'] = a.created.tshift(2)
    a['t3'] = a.created.tshift(3)
    print(a)
    print(np.datetime64('2012-10-08 18:15:05'))
    print(True in a.index.isin([np.datetime64('2012-10-08 18:15:05')]))
    [print(my_feature) for my_feature in a]
    print( set([my_feature for my_feature in a]))
#    print( set([my_feature for my_feature in a.index]))
    print("--")
    print(dict(a))
    print("===")
    [print(my_feature) for my_feature in a]
#    [print(dict(a[my_feature])) for my_feature in a]
#    print(dict(a).items())

print(f"panda version {pd.__version__}")
#rolling1()
#rolling2()
#rolling3()
#rolling4()
rolling6()

# <codecell> filter row subset df

a = pd.DataFrame(np.array([[1,2,3], [11,12,13], [21,22,23], [31,32,33]]),\
                 index = pd.date_range('2012-10-08 18:15:05', periods=4, freq='T'),
                 columns=['one', 'two', 'three'])
print(a)
b = pd.DataFrame(np.array([0,1,0,1]),\
                 index = pd.date_range('2012-10-08 18:15:05', periods=4, freq='T'),
                 columns=['x'])
print(b)
c=a[a.one==11]
print(c)
e=b[b.x==1]
print(e)
#d=a.drop(b[b.filter==1].index)
#print(d)
f=a[b.x == 1] # returns KeyError: False if column name is filter (filter is also a dataframe method!)
print(f)
g=pd.DataFrame(a.one)
print(g) # inlcudes full index
#h=a[f.index] # results in KeyError as it trys to find timestamps as column name
h=a[a.index.isin(f.index)]
print(h)

# <codecell> split df

a = pd.DataFrame(np.arange(6), columns=['created'],\
                 index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
a['s1'] = a.created.shift(1)
a['s2'] = a.created.shift(2)
a['s3'] = a.created.shift(3)
print(a)
cl = list(a.columns)
cl.remove('s1')
print(cl)
#b = pd.DataFrame(a, columns=cl)
b = a[cl]
print(b)

# <codecell> dict construct
import numpy as np
import pandas as pd

a = pd.DataFrame(np.arange(6), columns=['created'],\
                 index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
a['s1'] = a.created.shift(1)
a['s2'] = a.created.shift(2)
a['s3'] = a.created.shift(3)
features = {key:np.array(value) for key,value in dict(a).items()}
print(features)
#for key,value in dict(a).items():
#    print(f"key: {key}   - value: {np.array(value)}")
#    features = {key:np.array(value)}
#print(features)
print(np.array(a.columns))

key_list = [key for key in features]
print(key_list)
print('s1' in a)
print('sx' in a)

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

# <codecell> type check

def type_check():
    test1 = '123456';  # Only digit in this string
    print(type(test1))
    test2 = 123456;  # Only digit in this string
    print(type(test2))
    print(isinstance(test2, int))
    print(isinstance(test2, str))


# <codecell> append
import pandas as pd

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

def check_loc():
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
    ndf = df[(df.a == 'mimi')|(df.a == 'erich')]
    print(ndf)
    print(df)
    k = pd.DataFrame(df)
    k['b'] = ndf['b'] + df['b']
    print(k)

def check_loc4():
    df = pd.DataFrame(np.arange(6).reshape(3,2), columns=['A','B'])
    print(df)
    df.loc[:,'C'] = df.loc[:,'A']
    print(df)
    df.loc[3] = 5
    print(df)

#check_append()
check_loc3()
#print(str.isdecimal())

# <codecell> isin
import numpy as np
import pandas as pd

adf = pd.DataFrame(columns=["a","b", "c"],\
                 index = pd.date_range('2012-10-08 18:15:05', periods=3, freq='T'))
adf.iloc[0] = {'a': 'hugo1', 'b': 511, 'c': "x"}
adf.iloc[1] = {'a': 'hugo2', 'b': 512, 'c': "w"}
adf.iloc[2] = {'a': 'karl1', 'b': 512, 'c': "y"}
#df.iloc[3] = {'a': 'karl2', 'b': 513, 'c': "z"}

df = pd.DataFrame(columns=["a","b", "c"],\
                 index = pd.date_range('2012-10-08 18:15:05', periods=4, freq='T'))
df.iloc[0] = {'a': 'hugo3', 'b': 511, 'c': "x"}
df.iloc[1] = {'a': 'hugo4', 'b': 512, 'c': "w"}
df.iloc[2] = {'a': 'karl3', 'b': 512, 'c': "y"}
df.iloc[3] = {'a': 'karl4', 'b': 513, 'c': "z"}

bdf = df[df.index.isin(adf.index)]
print(bdf)
cdf = adf.index.difference(df.index)
print(len(cdf))

# <codecell> take over
import numpy as np
import pandas as pd

df = pd.DataFrame(columns=["a","b", "c"],\
                 index = pd.date_range('2012-10-08 18:15:05', periods=4, freq='T'))
df.iloc[0] = {'a': 'hugo', 'b': 511, 'c': "x"}
df.iloc[1] = {'a': 'hugo', 'b': 512, 'c': "w"}
df.iloc[2] = {'a': 'karl', 'b': 512, 'c': "y"}
df.iloc[3] = {'a': 'karl', 'b': 513, 'c': "z"}
df['i'] = df.index
print(df)
df = df.set_index('a')
print(df)
print(df.loc[df.b == 512])
print(df.loc[df.b == 512]['c'])

df['c'] = df.loc[df.b == 512, 'c']
print(df)
df = df.reset_index()
print(df)
df = df.set_index('i')
print(df)


# <codecell> dict
import numpy as np
import pandas as pd

def timedelta_minutes(first:str, last:str):
    my_min = pd.Timedelta(pd.Timestamp(last) - pd.to_datetime(first))
    min_res = my_min.days*24*60 + int(my_min.seconds/60)
    return min_res

def check_dict():
    test_d = {1:10, 2:20, 'cpc':'text'}
    for d in test_d:
        print(d)
        if isinstance(d, int):
            print(d+1)
        print(test_d[d])
    for k in range(0):
        print('hallo')

check_dict()
# my_time = pd.Timestamp('2009-01-01 05:45:00') - pd.to_datetime('2008-01-01 04:23:00')
my_time = pd.Timestamp('2009-01-02') - pd.to_datetime('2009-01-01')

min = pd.Timedelta(my_time,'m')
# min = np.timedelta64(my_time, 'm')
print(f"{min.days} {int(min.seconds/60)}")
print(f"{min.days*24*60 + int(min.seconds/60)}")
print(timedelta_minutes('2009-01-01', '2009-01-02'))
print((np.datetime64('2009-01-02', 'm') - np.datetime64('2009-01-01', 'm')) > np.timedelta64(2, 'm'))
print(np.datetime64('2009-01-02', 'm') )
print(np.datetime64('2009-01-02', 'm') - np.timedelta64(2, 'm'))

# <codecell> numpy

import numpy as np

class MyTest:
    def __init__(self, x):
        self.x = 3
        self.y = self.x *2

# x = np.array([2,3,1,0])
x = np.zeros((2,3), dtype=MyTest)
x[1,2] = MyTest(2)
print(x)

# <codecell> datetime config

import pandas as pd
import targets_features as t_f

fname = '/Users/tc/tf_models/crypto' + '/' + 'sample_set_split.config'
try:
    cdf = pd.read_csv(fname, skipinitialspace=True, \
                      converters={'set_start': np.datetime64})
    print(f"sample set split loaded from {fname}")
    cdf.sort_values(by=['set_start'], inplace=True)
    cdf = cdf.reset_index(drop=True)
    cdf['set_stop'] = cdf.set_start.shift(-1)
    cdf['diff'] = cdf['set_stop'] - cdf['set_start']
    print(cdf)
    print(cdf['diff'].sum())
    print(cdf.dtypes)
except IOError:
    print(f"pd.read_csv({fname}) IO error")

# <codecell> check_diff
import numpy as np
import pandas as pd
DATA_KEYS = ['a']

def check_diff(cur_btc_usdt, cur_usdt):
    """ cur_usdt should be a subset of cur_btc_usdt. This function checks the average deviation.
    """
    print(f"c-u first: {cur_usdt.index[0]}  last: {cur_usdt.index[len(cur_usdt)-1]}")
    print(f"c-b-u first: {cur_btc_usdt.index[0]}  last: {cur_btc_usdt.index[len(cur_btc_usdt)-1]}")
    # diff = cur_btc_usdt[cur_btc_usdt.index.isin(cur_usdt.index)]
    diff = dict.fromkeys(DATA_KEYS, 0)
    for tic in cur_usdt.index:
        for key in DATA_KEYS:
            print(f"{tic} {key} {diff[key]} {cur_btc_usdt.loc[tic, key]} {cur_usdt.loc[tic, key]}")
            if key != 'volume':
                diff[key] += (cur_btc_usdt.loc[tic, key] - cur_usdt.loc[tic, key]) / cur_usdt.loc[tic, key]
            else:
                diff[key] += cur_btc_usdt.loc[tic, key] - cur_usdt.loc[tic, key]
    for key in DATA_KEYS:
        diff_average = diff[key] / len(cur_usdt)
        print(diff[key])
        if key != 'volume':
            print(f"check_diff {key}: {diff_average:%}")
        else:
            print(f"check_diff {key}: {diff_average}")

def check_diff2(cur_btc_usdt, cur_usdt):
    """ cur_usdt should be a subset of cur_btc_usdt. This function checks the average deviation.
    """
    print(f"c-u first: {cur_usdt.index[0]}  last: {cur_usdt.index[len(cur_usdt)-1]}")
    print(f"c-b-u first: {cur_btc_usdt.index[0]}  last: {cur_btc_usdt.index[len(cur_btc_usdt)-1]}")
    diff = cur_btc_usdt[cur_btc_usdt.index.isin(cur_usdt.index)]
    for key in DATA_KEYS:
        if key != 'volume':
            diff[key] = (cur_btc_usdt[key] - cur_usdt[key]) / cur_usdt[key]
        else:
            diff[key] = cur_btc_usdt[key] - cur_usdt[key]
        diff_average = diff[key].sum() / len(cur_usdt)
        # print(diff[key])
        if key != 'volume':
            print(f"check_diff {key}: {diff_average:%}")
        else:
            print(f"check_diff {key}: {diff_average}")

adf = pd.DataFrame(np.arange(2,6), columns=['a'],\
                 index = pd.date_range('2012-10-08 18:15:05', periods=4, freq='T'))
print(adf)
bdf = pd.DataFrame(np.arange(1,7), columns=['a'],\
                 index = pd.date_range('2012-10-08 18:15:05', periods=6, freq='T'))
# check_diff2(bdf, adf)
print(bdf)
bdf.loc[adf.index,adf.columns] = adf[:] # replace only values of subset adf in bdf
print(bdf)
s = set()
for ts in adf.index:
    s.add(ts)
print(s)
l = list(s)
print(l)
l.sort()
print(l)
for ts in l:
    print(ts)

# <codecell> check_diff
from datetime import datetime

dt = datetime.now()
print(dt.strftime('%Y-%m-%d_%H:%M'))
print(datetime.now().strftime('%Y-%m-%d_%H:%M'))

# <codecell> check_diff

print (list(zip(['AB', 'CD'], *[iter('EF')]*2)))

# <codecell> timediff
import pandas as pd
from datetime import datetime, timedelta  # , timezone

minutes = 100
dtnow = datetime.utcnow()
last_tic = pd.Timestamp(year=2019, month=4, day=20, hour=21, minute=35)
pylast = last_tic.to_pydatetime()
dtlast = pylast.replace(tzinfo=None)
dfdiff = int((dtnow - dtlast) / timedelta(minutes=1))
print(f"now {dtnow} last_tic {last_tic} pylast {pylast} dtlast {dtlast} dfdiff {dfdiff}")
if dfdiff < minutes:
    tdminutes = dfdiff
print(pd.Timestamp(dtnow).week)
print(pd.Timestamp(dtnow).weekofyear)
