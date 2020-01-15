#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
from datetime import datetime  # , timedelta
import numpy as np
import pandas as pd
from sklearn.utils import Bunch

import env_config as env
from env_config import Env
import crypto_targets as ct
import cached_crypto_data as ccd


class NoSubsetWarning(Exception):
    pass


def targets_to_features(tfv_ta_df, target_df):
    """Extracts a sample subset with targets and features of a specific time aggregation
    based on given targets. target_df and tfv_ta_df both have to share the same index basis.
    The index of target_df shall be a subset of tfv_ta_df.
    """
    df = tfv_ta_df[tfv_ta_df.index.isin(target_df.index)]
    # check compatibility of target_df.sym with
    d = len(target_df.index.difference(tfv_ta_df.index))
    c = len(df)
    b = len(target_df)
    p = len(tfv_ta_df)
    if d > 0:
        raise NoSubsetWarning(f"subset({b}) with {c}/{d} rows that are/are not in superset({p})")
    return df


def report_setsize(setname, df):
    print(f"{str_setsize(df)} on {setname}")


def str_setsize(df):
    hc = len(df[df.target == ct.TARGETS[ct.HOLD]])
    sc = len(df[df.target == ct.TARGETS[ct.SELL]])
    bc = len(df[df.target == ct.TARGETS[ct.BUY]])
    tc = hc + sc + bc
    return f"buy {bc} sell {sc} hold {hc} total {tc}"


def to_scikitlearn(df, np_data=None, descr=None):
    """Load and return the crypto dataset (classification).
    """

    fn_list = list(df.keys())
    fn_list.remove("target")
    fn_list.remove("close")
    if np_data is None:
        # data = df[fn_list].to_numpy(dtype=float) # incompatible with pandas 0.19.2
        data = df[fn_list].values
    else:
        data = np_data
        # target = df["target"].to_numpy(dtype=float) # incompatible with pandas 0.19.2
        # tics = df.index.to_numpy(dtype=np.datetime64) # incompatible with pandas 0.19.2
    target = df["target"].values  # compatible with pandas 0.19.2
    close = df["close"].values  # compatible with pandas 0.19.2
    tics = df.index.values  # compatible with pandas 0.19.2
    feature_names = np.array(fn_list)
    target_names = np.array(ct.TARGETS.keys())
    if descr is None:
        descr = "missing description"

    return Bunch(data=data, target=target, close=close,
                 target_names=target_names,
                 tics=tics,
                 descr=descr,
                 feature_names=feature_names)


class TargetsFeatures:
    """ Holds the source ohlcv data as pandas DataFrame 'minute_data',
        the feature vectors as DataFrame rows of 'vec' and
        the target trade signals in column 'target' in both DataFrames.

    """

    def __init__(self, base, minute_dataframe=None, path=None):
        """Receives the key attributes for feature generation

        if minute_dataframe is not None then it is used
        otherwise if path is not None the base will be loaded via the given path

        """
        assert(env.config_ok())
        self.__base = base
        self.__quote = Env.quote
        self.__feature_type = "Finvalid"
        self.path = path
        assert (minute_dataframe is not None) or (path is not None)
        self.minute_data = minute_dataframe  # is DataFrame with ohlvc and target columns
        if self.minute_data is not None:
            self.crypto_targets()
        elif path is not None:
            try:
                self.minute_data = ccd.load_asset_dataframe(self.__base, path=path)
            except env.MissingHistoryData:
                raise
            else:
                self.crypto_targets()
        report_setsize(self.__base, self.minute_data)
        self.vec = None  # is a  DataFrame with features columns and 'target', 'close' columns

    def calc_features(self, minute_data):
        print("ERROR: no features implemented")
        return None

    def crypto_targets(self):
        """Assigns minute_dataframe to attribute *minute_data*.
        If *minute_data* has no 'target' column then targets are calculated and added to
        *minute_data*.

        minute_dataframe shall have
        the columns: open, high, low, close, volume and timestamps as index
        """

        if self.minute_data is None:
            raise env.MissingHistoryData("{}–{} symbol without minute data ({})".format(
                                        self.__base, self.__quote, self.vec))
        if self.minute_data.empty is None:
            self.minute_data = None
            raise env.MissingHistoryData("{}–{} symbol with empty minute data".format(
                                     self.__base, self.__quote))
        if "target" not in self.minute_data:
            ct.crypto_trade_targets(self.minute_data)  # add aggregation targets

    def load_cache_ok(self):
        """ Will load cached base data if present.

            Returns True if cache can be loaded and False if not.
        """
        if self.path is None:
            return False
        df = None
        sym = env.sym_of_base(self.__base)
        fname = Env.cache_path + sym + "_" + self.__feature_type + "_DataFrame.h5"
        try:
            # df = pd.read_msgpack(fname)
            df = pd.read_hdf(fname, sym)
            print("{}: loaded {}({}) {} tics ({} - {})".format(
                datetime.now().strftime(Env.dt_format), self.__feature_type, env.sym_of_base(self.__base),
                len(df), df.index[0].strftime(Env.dt_format),
                df.index[len(df)-1].strftime(Env.dt_format)))
            if not self.minute_data.index.isin(df).all:
                to_be_calculated = self.minute_data.index(df.index)
                # ! TODO split into subset of consecutive elements, calc features and targets for missing parts
                # ! TODO concat loaded and calculated sets
                print("WARNING: need to calculate features as loaded features are only subset of main set")
                return False
        except IOError:
            return False
        return (df is not None)

    def save_cache(self):
        """ Will save features and targets of self.base in Env.cache_path.
        """
        if self.path is None:
            return
        if self.vec is None:
            print("WARNING: Unexpected call to save features in cache with missing features")
            return
        print("{}: writing {}({}) {} tics ({} - {})".format(
            datetime.now().strftime(Env.dt_format), self.__feature_type, env.sym_of_base(self.__base),
            len(df), df.index[0].strftime(Env.dt_format),
            df.index[len(df)-1].strftime(Env.dt_format)))
        sym = env.sym_of_base(self.__base)
        fname = Env.cache_path + sym + "_" + self.__feature_type + "_DataFrame.h5"
        df.to_hdf(fname, sym, mode="w")

    def calc_features_and_targets(self):
        """Assigns minute_dataframe to attribute *minute_data*.
        Calculates features and assigns them to attribute *vec*.
        If minute_dataframe is None
        then an earlier assigned *minute_data* is used to recalculate features.
        If *minute_data* has no 'target' column then targets are calculated and added to
        *minute_data*.

        Releasing feature data by assigning *vec* None and recalculating those later is a valid
        use case to free up memory temporary.

        minute_dataframe shall have
        the columns: open, high, low, close, volume and timestamps as index
        """

        if not self.load_cache_ok():
            self.crypto_targets()
            self.vec = self.calc_features(self.minute_data)
            if self.vec is not None:
                if "target" in self.minute_data:
                    self.vec.loc[:, "target"] = self.minute_data.loc[:, "target"]
                if self.path is not None:
                    self.save_cache()
            else:
                print(f"ERROR: feature calculation failed")
        return self.vec

    def __append_minute_df_with_targets(self, minute_df):
        """ unused?
        """
        self.vec = None
        if "target" not in minute_df:
            raise ValueError("append_minute_df_with_targets: missing target column")
        if self.minute_data is None:
            self.minute_data = minute_df
        else:
            self.minute_data = pd.concat([self.minute_data, minute_df], sort=False)


def target_labels(target_id):
    return ct.TARGET_NAMES[target_id]


if __name__ == "__main__":
    env.test_mode()
    base = "xrp"
    df = ccd.load_asset_dataframe(base, Env.data_path)
    df = df.loc[(df.index >= pd.Timestamp("2019-01-01 15:59:00+00:00")) &
                (df.index <= pd.Timestamp("2019-01-31 15:58:00+00:00"))]
    tf = TargetsFeatures(base, df)
    tf.minute_data["label"] = tf.minute_data["target"].apply(lambda x: ct.TARGET_NAMES[x])
    tf.minute_data["label2"] = tf.minute_data["target2"].apply(lambda x: ct.TARGET_NAMES[x])
    print(tf.minute_data.head(2))
    print(tf.minute_data.tail(2))
