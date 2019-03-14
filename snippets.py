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
my_time = pd.Timestamp('2009-01-01') - pd.to_datetime('2008-01-01')

min = pd.Timedelta(my_time,'m')
# min = np.timedelta64(my_time, 'm')
print(f"{min.days} {int(min.seconds/60)}")
print(f"{min.days*24*60 + int(min.seconds/60)}")
print(timedelta_minutes('2008-01-01', '2009-01-01'))

