#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 21:43:26 2019

@author: tc
"""
import logging
# from datetime import datetime  # , timedelta
# import numpy as np
import pandas as pd

import env_config as env
from env_config import Env
import crypto_targets as ct
import cached_crypto_data as ccd

logger = logging.getLogger(__name__)


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
        self.vec = None
        self.base = base
        self.quote = Env.quote
        self.path = path
        self.minute_data = minute_dataframe  # is DataFrame with ohlvc and target columns
        if (self.minute_data is None) and (path is not None):
            try:
                self.minute_data = ccd.load_asset_dataframe(self.base, path=path)
            except env.MissingHistoryData:
                raise

    @staticmethod
    def history():
        "history_minutes_without_features"
        return 0

    @staticmethod
    def feature_count():
        "returns the number of features for one sample"
        return 0

    @staticmethod
    def feature_str():
        "returns a string that represent the features class as mnemonic"
        return "AbstractTargetsFeatures"

    def calc_features(self, minute_data):
        logger.error("no features implemented")
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
                                        self.base, self.quote, self.vec))
        if self.minute_data.empty:
            self.minute_data = None
            raise env.MissingHistoryData("{}–{} symbol with empty minute data".format(
                                     self.base, self.quote))
        if "target" not in self.minute_data:
            self.minute_data.loc[:, "target"] = ct.crypto_trade_targets(self.minute_data)  # add aggregation targets

    def index_ok(self):
        """ Will check index start for consistency between minute_data and vec.
            index is also considered ok when vec is shorter than minute_data.
            vec supplementations have to be checked separately.
        """
        assert self.minute_data is not None
        assert self.vec is not None
        ok = True
        len_md = len(self.minute_data)
        len_vec = len(self.vec)
        if len_md == len_vec:  # minute_data was already reduced to vec length
            if self.minute_data.index[0] != self.vec.index[0]:
                logger.warning("vec start is {} but was expected {}".format(
                    self.vec.index[0], self.minute_data.index[0]))
                ok = False
            if self.minute_data.index[len_md-1] != self.vec.index[len_vec-1]:
                logger.warning("unexpected last tic of minute data {} versus vec {}".format(
                    self.minute_data.index[len_md-1], self.vec.index[len_vec-1]))
                ok = False
        else:   # orignal minute data length including history that is not in vec
            hmwf = self.history()
            vec_start = self.minute_data.index[0] + pd.Timedelta(hmwf, unit='T')
            if vec_start != self.vec.index[0]:
                logger.warning("vec start is {} but was expected {}".format(self.vec.index[0], vec_start))
                ok = False
        return ok

    def supplement_vec(self):
        """ Will check index start for consistency between minute_data and vec.
            index is also considered ok when vec is shorter than minute_data.
            vec supplementations have to be checked separately.
        """
        assert self.minute_data is not None
        assert self.vec is not None
        if self.minute_data.index[-1] != self.vec.index[-1]:
            # orignal minute data length including history that is not in vec
            hmwf = self.history() + 1  # recalc last due to potentially reloaded ohlcv
            ohlcv_start = self.vec.index[-1] - pd.Timedelta(hmwf, unit='T')
            assert ohlcv_start >= self.minute_data.index[0]
            ohlcv_df_rem = self.minute_data.loc[self.minute_data.index >= ohlcv_start]
            vec_df_rem = self.calc_features(ohlcv_df_rem)
            self.vec = self.vec.drop([self.vec.index[-1]])
            vec_df_rem = vec_df_rem.loc[vec_df_rem.index > self.vec.index[-1]]
            if "target" in self.minute_data:
                assert (vec_df_rem.index[0] - self.vec.index[-1]) == pd.Timedelta(1, unit='T')
                self.vec = pd.concat([self.vec, vec_df_rem], sort=False)
                assert self.vec.index[-1] == self.minute_data.index[-1]
                # copy target to vec
                self.vec = self.vec.drop(columns="target")
                target_md = self.minute_data.loc[self.minute_data.index >= self.vec.index[0]]
                self.vec = pd.concat([self.vec, target_md.target.rename("target")], axis=1, sort=False)
            else:
                if "target" in self.vec:
                    # supplement targets that are in vec but not in minute_data
                    ohlcv_start = self.vec.index[-1] - pd.Timedelta(ct.trade_signal_history(), unit='T')
                    assert ohlcv_start >= self.minute_data.index[0]
                    ohlcv_df_rem = self.minute_data.loc[self.minute_data.index >= ohlcv_start]
                    ohlcv_df_rem["target"] = ct.crypto_trade_targets(ohlcv_df_rem)
                    ohlcv_df_rem = ohlcv_df_rem.loc[ohlcv_df_rem.index > self.vec.index[-1]]
                    assert vec_df_rem.index[-1] == ohlcv_df_rem.index[-1]
                    assert vec_df_rem.index[0] == ohlcv_df_rem.index[0]
                    assert len(vec_df_rem.index) == len(ohlcv_df_rem.index)
                    assert (ohlcv_df_rem.index[0] - self.vec.index[-1]) == pd.Timedelta(1, unit='T')
                    vec_df_rem = pd.concat([vec_df_rem, ohlcv_df_rem.target.rename("target")], axis=1, sort=False)
                    assert len(vec_df_rem.index) == len(ohlcv_df_rem.index)

                    assert (vec_df_rem.index[0] - self.vec.index[-1]) == pd.Timedelta(1, unit='T')
                    self.vec = pd.concat([self.vec, vec_df_rem], sort=False)
                    assert self.vec.index[-1] == self.minute_data.index[-1]
                else:
                    # calculate all targets from scratch
                    self.minute_data["target"] = ct.crypto_trade_targets(self.minute_data)
                    if "target" in self.minute_data:
                        smd = self.minute_data.loc[self.minute_data.index >= self.vec.index[0]]
                        self.vec["target"] = smd["target"]
            # env.check_df(self.vec)

    def load_cache_ok(self):
        """ Will load cached base data if present
            Returns True if cache can be loaded and False if not.
        """
        if self.path is None:
            return False
        sym = env.sym_of_base(self.base)
        fname = self.path + sym + "_" + self.feature_str() + "_DataFrame.h5"
        try:
            self.vec = pd.read_hdf(fname, sym)  # targets and features
            logger.debug("loaded {}({}) {} tics ({} - {})".format(
                self.feature_str(), env.sym_of_base(self.base),
                len(self.vec), self.vec.index[0].strftime(Env.dt_format),
                self.vec.index[len(self.vec)-1].strftime(Env.dt_format)))
            if self.index_ok():
                self.supplement_vec()
                assert "target" in self.vec
                assert len(self.vec) > 0
                self.minute_data = pd.concat([self.minute_data, self.vec.target], axis=1, sort=False)
            else:
                self.vec = None
        except IOError:
            return False
        return (self.vec is not None)

    def save_cache(self):
        """ Will save features and targets of self.base in Env.cache_path.
        """
        if self.path is None:
            return
        if self.index_ok():
            logger.debug("writing {}({}) {} tics ({} - {})".format(
                self.feature_str(), env.sym_of_base(self.base),
                len(self.vec), self.vec.index[0].strftime(Env.dt_format),
                self.vec.index[len(self.vec)-1].strftime(Env.dt_format)))
            sym = env.sym_of_base(self.base)
            fname = self.path + sym + "_" + self.feature_str() + "_DataFrame.h5"
            self.vec.to_hdf(fname, sym, mode="w")
        else:
            logger.error(f"feature cache save of {self.base} failed due to index check")

    def enforce_target_recalculation(self):
        """ repair action to use new target algorithm
        """
        if "target" in self.vec:
            self.vec = self.vec.drop(columns="target")
        if "target" in self.minute_data:
            self.minute_data = self.minute_data.drop(columns="target")
            logger.info("enforce_target_recalculation")
            self.crypto_targets()  # recalculate all targets from scratch
            smd = self.minute_data.loc[self.minute_data.index >= self.vec.index[0]]
            self.vec["target"] = smd["target"]

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

        if self.vec is None:
            if not self.load_cache_ok():
                self.crypto_targets()
                self.vec = self.calc_features(self.minute_data)
                if self.vec is not None:
                    if "target" in self.minute_data:
                        smd = self.minute_data.loc[self.minute_data.index >= self.vec.index[0]]
                        self.vec["target"] = smd["target"]
                    # if self.path is not None:
                    #     self.save_cache()
                else:
                    logger.error("feature calculation failed")
        # if self.minute_data is not None:  # disabled to not loose history data
        #     self.minute_data = self.minute_data[self.minute_data.index >= self.vec.index[0]]

        # self.enforce_target_recalculation()  # ! temporary repair action
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
    logger.info("probably launched wrong python file")
