#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
# import numpy as np
import pandas as pd
from sklearn.utils import Bunch
import numpy as np
import env_config as env
from env_config import Env
import crypto_targets as ct
import cached_crypto_data as ccd
import aggregated_features as caf
import condensed_features as cof


PICKLE_EXT = ".pydata"  # pickle file extension
JSON_EXT = ".json"  # msgpack file extension
MSG_EXT = ".msg"  # msgpack file extension

VOL_BASE_PERIOD = "1D"
TARGET_KEY = 5
CONDENSED_FEATURES = True


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
    hc = len(df[df.target == ct.TARGETS[ct.HOLD]])
    sc = len(df[df.target == ct.TARGETS[ct.SELL]])
    bc = len(df[df.target == ct.TARGETS[ct.BUY]])
    tc = hc + sc + bc
    print(f"buy {bc} sell {sc} hold {hc} total {tc} on {setname}")


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

    def __load_ohlcv_and_targets(self):
        """ OBSOLETE
            Loads the historic data temporarily, calculates the features and targets
            and releases the original data afterwards
        """
        try:
            df = ccd.load_asset_dataframe(self.__base, path=Env.data_path)
        except env.MissingHistoryData:
            raise
        else:
            self.calc_features_and_targets(df)
            report_setsize(self.__base, self.vec)

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

        self.crypto_targets()
        if CONDENSED_FEATURES:
            self.vec = cof.calc_features(self.minute_data)
        else:
            self.vec = caf.calc_features(self.minute_data)
        if "target" in self.minute_data:
            self.vec.loc[:, "target"] = self.minute_data.loc[:, "target"]
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
