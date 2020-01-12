import pandas as pd
import env_config as env
from env_config import Env

VOL_BASE_PERIOD = "1D"
TARGET_KEY = 5

"""Receives a dict of currency pairs with associated minute candle data and
transforms it into a dict of currency pairs with associated dicts of
time_aggregations features. The time aggregation is the dict key with one
special key 'CPC' that provides the summary targets

Attributes
----------
time_aggregations:
    dict with required time aggregation keys and associated number
    of periods that shall be compiled in a corresponding feature vector
minute_data:
    currency pair (as keys) dict of input minute data as corresponding
    pandas DataFrame

To Do
=====
buy - sell signals:
    now reduced to first signal
    rolling shall allow a richer buy - sell signalling that are simply mapped on a
    common timeline where the optimization problem is addressed

>>> read dataframe from file
>>> concatenate features to full feature vector and write it as file


abbreviatons and terms
======================
time aggregation - time period for which features are derived, e.g. open, close, high, low.
In this context different time aggregations are used to low pass filter high frequent
volatility.

aggregations:
    dict with required time aggregation keys and associated number
    of periods that shall be compiled in a corresponding feature vector
target_key:
    has to be a key of aggregations. Targets are only calculated for that target_key


"""


def calc_features(minute_data):
    tf_aggs = __calc_aggregation(minute_data, Env.time_aggs)
    vec = __expand_feature_vectors(tf_aggs, TARGET_KEY)
    return vec


def __smallest_dict_key(thisdict):
    smallest_key = 5000
    for k in thisdict:
        if isinstance(k, int):
            if k < smallest_key:
                smallest_key = k
    assert smallest_key != 5000, "no int in dict keys"
    return smallest_key


def __derive_features(df):
    """derived features in relation to price based on the provided
    time aggregated dataframe df with the exception of the derived feature 'delta'
    """
    # price deltas in 1/1000
    df["height"] = (df["high"] - df["low"]) / df["close"] * 1000
    df.loc[df["close"] > df["open"], "top"] = (df["high"] - df["close"]) / df["close"] * 1000
    df.loc[df["close"] <= df["open"], "top"] = (df["high"] - df["open"]) / df["close"] * 1000
    df.loc[df["close"] > df["open"], "bottom"] = (df["open"] - df["low"]) / df["close"] * 1000
    df.loc[df["close"] <= df["open"], "bottom"] = (df["close"] - df["low"]) / df["close"] * 1000


def __calc_aggregation(minute_df, time_aggregations):
    """Time aggregation through rolling aggregation with the consequence that new data is
    generated every minute and even long time aggregations reflect all minute bumps in their
    features

    in:
        dataframe of minute data of a currency pair
        with the columns: open, high, low, close, volume
    out:
        dict of dataframes of aggregations with features
    """
    tf_aggs = dict()  # feature aggregations
    mdf = minute_df  # .copy()
    df = pd.DataFrame(minute_df)  # .copy()
    df["vol"] = (mdf["volume"] - mdf.volume.rolling(VOL_BASE_PERIOD).mean()) \
        / mdf.volume.rolling(VOL_BASE_PERIOD).mean()
    df = df.fillna(value={"vol": 0.000001})
    maxmin = 0
    for time_agg in time_aggregations:
        if isinstance(time_agg, int):
            if (time_agg * time_aggregations[time_agg]) > maxmin:
                maxmin = time_agg * time_aggregations[time_agg]
    if maxmin > len(df.index):
        raise env.MissingHistoryData("History data has {} samples but should have >= {}".format(
                len(df.index), maxmin))
    for time_agg in time_aggregations:
        # print(f"{datetime.now()}: time_aggregation {time_agg}")
        if time_agg > 1:
            df = pd.DataFrame()
            df["open"] = mdf.open.shift(time_agg-1)
            df["high"] = mdf.high.rolling(time_agg).max()
            df["low"] = mdf.low.rolling(time_agg).min()
            df["close"] = mdf.close
            df["vol"] = mdf.vol.rolling(time_agg).mean()
        df["delta"] = (mdf.close - mdf.close.shift(time_agg)) / mdf.close.shift(time_agg)
        tf_aggs[time_agg] = df
        __derive_features(df)
    return tf_aggs


def __expand_feature_vectors(tf_aggs, target_key):
    """Builds a feature vector for just the target_key with
    1 minute DHTBV and D*V feature sequences and the remaining D sequences of
    n time steps (tics) as configured in time_aggregations in T units.
    The most important step in expand_feature_vectors is
    1) the concatenation of feature vectors per sample to provide a history
    for the classifier
    2) discarding the original currency values that are not used
    as features (except 'close')

    Result:
        A DataFrame with feature vectors as rows. The column name indicates
        the type of feature, i.e. either 'close' or 'D|H|T|B|V|DV' in case of
        1 minute aggregation or just 'D' for all other aggregations with aggregation+'T_'
        as column prefix.
    """
    df = pd.DataFrame(tf_aggs[target_key], columns=["close"])
    skey = __smallest_dict_key(tf_aggs)
    for ta in tf_aggs:
        for tics in range(Env.time_aggs[ta]):
            ctitle = str(ta) + "T_" + str(tics) + "_"
            offset = tics*ta
            # now add feature columns according to aggregation
            df[ctitle + "D"] = tf_aggs[ta].delta.shift(offset)
            if ta == skey:  # full set only for smallest aggregation (minute data)
                df[ctitle + "H"] = tf_aggs[ta].height.shift(offset)
                df[ctitle + "T"] = tf_aggs[ta].top.shift(offset)
                df[ctitle + "B"] = tf_aggs[ta].bottom.shift(offset)
                df[ctitle + "V"] = tf_aggs[ta].vol.shift(offset)
                df[ctitle + "DV"] = tf_aggs[ta].vol.shift(offset) *\
                    tf_aggs[ta].delta.shift(offset)
    df = df.dropna()
    if df.empty:
        raise env.MissingHistoryData("empty dataframe from expand_feature_vectors")
    return df