import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)  # before importing tensorflow.

# import env_config as env
# from env_config import Env

# import os
import pandas as pd
# import numpy as np
# import timeit
# import itertools
# import math
# import pickle

# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import Process

# Import datasets, classifiers and performance metrics
# from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.metrics as km
# import keras
# import keras.metrics as km
# import tensorflow.compat.v1 as tf1

# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

import crypto_targets as ct
import cached_crypto_data as ccd
# import condensed_features as cof
# import aggregated_features as agf
# import classify_keras as ck
import adaptation_data as ad
# import classifier_predictor as class_pred

"""
NUM_PARALLEL_EXEC_UNITS = 6
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
    inter_op_parallelism_threads=2,
    allow_soft_placement=True,
    device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.compat.v1.Session(config=config)
# keras.backend.set_session(session)
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
"""
logger = logging.getLogger(__name__)


class PredictionData(ccd.CryptoData):

    def __init__(self, estimator_obj):
        super().__init__()
        self.predictor = estimator_obj
        self.missing_file_warning = False
        self.path = self.predictor.path_with_epoch()
        self.btl = [0.5, 0.6]
        self.stl = [0.6, 0.7]
        self.fee_factor = ct.FEE
        self.base_dfl = None  # dict of lists with prediction data frames
        self.total = None  # data frame with total performance and count
        self.note = None  # data frame with intermediate results
        self.confusion = None  # data frame with confusion matrix target vs actual

    def history(self):
        "no history minutes are requires to calculate prediction data"
        return 0

    def keys(self):
        "returns the list of element keys"
        return self.predictor.targets.target_dict().keys()

    def mnemonic(self):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        mem = self.predictor.mnemonic_with_epoch() + "_predictions"
        return mem

    def predict_batch(self, base: str, features_df):
        """ Predicts all samples and returns the result as numpy array.
            set_type specific evaluations can be done using the saved prediction data.
        """
        if (features_df is None) or features_df.empty:
            return None

        # logger.info(
        #     "before scaling {} first {} last {}\n{}".format(
        #         base, features_df.index[0], features_df.index[-1],
        #         features_df.describe(percentiles=[], include='all')))
        if base in self.predictor.scaler:
            fdf_scaled = self.predictor.scaler[base].transform(features_df.values)
            # fdf = pd.DataFrame(data=fdf_scaled, index=features_df.index, columns=features_df.columns)
            # logger.info(
            #     "after scaling {} first {} last {}\n{}".format(
            #         base, fdf.index[0], fdf.index[-1],
            #         fdf.describe(percentiles=[], include='all')))
            pred = self.predictor.kerasmodel.predict_on_batch(fdf_scaled)
        else:
            logger.warning(f"no scaler for {base}")
            pred = self.predictor.kerasmodel.predict_on_batch(features_df.values)
        # pdf = pd.DataFrame(data=pred, index=fdf.index, columns=self.keys())
        return pred

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        ohlcv = self.predictor.ohlcv.get_data(base, first, last)
        fdf = self.predictor.features.get_data(base, first, last)
        tdf = self.predictor.targets.get_data(base, first, last)
        if (tdf is None) or tdf.empty:
            return None
        pred = self.predict_batch(base, fdf)
        pdf = pd.DataFrame(data=pred, index=fdf.index, columns=self.keys())
        pdf = pd.concat([ohlcv.close, tdf.target, pdf], axis=1, join="inner")
        return self.check_timerange(pdf, first, last)

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp):
        return self.new_data(base, first, last)

    def process_hold_signal(self, row, bt, st):
        self.confusion.loc[row["target"], (bt, st, ct.TARGETS[ct.HOLD])] += 1
        return row

    def process_sell_signal(self, row):
        for bt in self.btl:
            for st in self.stl:
                if (row[ct.SELL] > st):
                    if (self.note[bt, st, "bprice"] > 0):
                        # close open position
                        row[bt, st, "btic"] = self.note[bt, st, "btic"]
                        row[bt, st, "bprice"] = self.note[bt, st, "bprice"]
                        row[bt, st, "perf"] = (
                            row["close"] * (1 - self.fee_factor) -
                            self.note[bt, st, "bprice"] * (1 + self.fee_factor)) / self.note[bt, st, "bprice"]
                        self.note[bt, st, "bprice"] = 0  # close position
                        self.total[bt, st, "perf"] += row[bt, st, "perf"]
                        self.total[bt, st, "count"] += 1
                    self.confusion.loc[row["target"], (bt, st, ct.TARGETS[ct.SELL])] += 1
                else:
                    row = self.process_hold_signal(row, bt, st)
        return row

    def process_buy_signal(self, row):
        for bt in self.btl:
            for st in self.stl:
                if row[ct.BUY] > bt:
                    if self.note[bt, st, "bprice"] <= 0:
                        self.note.loc[bt, st, "bprice"] = row["close"]
                        self.note.loc[bt, st, "btic"] = row["tic"]
                    self.confusion.loc[row["target"], (bt, st, ct.TARGETS[ct.BUY])] += 1
                else:
                    row = self.process_hold_signal(row, bt, st)
        return row

    def calc_perf_row(self, row):
        if row[ct.SELL] >= row[ct.HOLD]:
            if row[ct.SELL] >= row[ct.BUY]:
                row = self.process_sell_signal(row)
            else:
                row = self.process_buy_signal(row)  # because hold <= sell < buy
        else:
            if row[ct.BUY] >= row[ct.HOLD]:
                row = self.process_buy_signal(row)  # because sell < hold <= buy
            else:
                for bt in self.btl:
                    for st in self.stl:
                        row = self.process_hold_signal(row, bt, st)
        return row

    def calc_performance(self, pdf):
        mix1 = pd.MultiIndex.from_product([self.btl, self.stl, ["btic", "bprice", "perf"]])
        rdf = pd.DataFrame(index=pdf.index, columns=mix1)
        df = pd.concat([pdf, rdf], axis=1, join="inner")
        df.index.rename("tic", inplace=True)
        df = df.reset_index()  # to get access to tic in apply
        df.apply(self.calc_perf_row, axis=1, result_type="broadcast")
        df = df.set_index("tic")
        return df

    def prepare_calc_performance(self):
        mix2a = pd.MultiIndex.from_product([self.btl, self.stl, ["perf", "count"]])
        self.total = pd.Series(index=mix2a)
        mix2b = pd.MultiIndex.from_product([self.btl, self.stl, ["btic", "bprice"]])
        self.note = pd.Series(index=mix2b)
        idx = pd.IndexSlice
        self.note.loc[idx[:, :, "bprice"]] = 0
        self.total.loc[idx[:, :, "perf"]] = 0
        self.total.loc[idx[:, :, "count"]] = 0
        mix3 = pd.MultiIndex.from_product([self.btl, self.stl, ct.TARGETS.values()])
        self.confusion = pd.DataFrame(data=0, index=ct.TARGETS.values(), columns=mix3, dtype=int)
        self.confusion.index.rename("target", inplace=True)
        self.confusion.columns.rename(["bt", "st", "actual"], inplace=True)

    def find_best(self):
        """ find best result in self.total and return result
        """
        best_st = 0
        best_bt = 0
        best_perf = -1.1
        for bt in self.btl:
            for st in self.stl:
                if self.total[bt, st, "perf"] > best_perf:
                    best_perf = self.total[bt, st, "perf"]
                    best_st = st
                    best_bt = bt
        return (best_perf, self.total[best_bt, best_st, "count"], best_bt, best_st)

    def assess_perf(self, bases: list, set_type, epoch):
        """Evaluates the performance on the given set and prints the confusion and
        performance matrix.

        Returns a tuple of (best-performance-factor, at-buy-probability-threshold,
        at-sell-probability-threshold, with-number-of-transactions)
        """
        self.prepare_calc_performance()
        self.base_dfl = dict()  # base dict of lists with prediction data frames
        for base in bases:
            self.base_dfl[base] = list()
            ohlcv_list = ad.SplitSets.split_sets(set_type, self.predictor.ohlcv.load_data(base))
            features_list = ad.SplitSets.split_sets(set_type, self.predictor.features.load_data(base))
            targets_list = ad.SplitSets.split_sets(set_type, self.predictor.targets.load_data(base))
            assert len(ohlcv_list) == len(features_list)
            assert len(ohlcv_list) == len(targets_list)
            for ix in range(len(ohlcv_list)):
                odf = ohlcv_list[ix]
                fdf = features_list[ix]
                tdf = targets_list[ix]
                if (fdf is None) or fdf.empty or (tdf is None) or tdf.empty:
                    logger.warning(f"empty data for {base} between {odf.index[0]} and {odf.index[-1]}")
                    continue
                [fdf, tdf] = ccd.common_timerange([fdf, tdf])

                if self.predictor.scaler is not None:
                    fdf_scaled = self.predictor.scaler.transform(fdf.values)
                    pred = self.predictor.kerasmodel.predict_on_batch(fdf_scaled)
                else:
                    logger.error("missing scaler")
                    pred = self.predictor.kerasmodel.predict_on_batch(fdf.values)
                if pred is None:
                    logger.warning(f"no prediction data for {base} between {odf.index[0]} and {odf.index[-1]}")
                    continue
                pdf = pd.DataFrame(data=pred, index=fdf.index, columns=self.predictor.targets.target_dict().keys())
                pdf.loc[pdf.index[-1], ct.SELL] = 1  # force sell at end of data range
                if pdf.empty:
                    logger.warning(f"empty prediction data for {base} between {odf.index[0]} and {odf.index[-1]}")
                    continue
                pdf = pd.concat([odf.close, tdf.target, pdf], axis=1, join="inner")
                self.base_dfl[base].append(self.calc_performance(pdf))
        logger.info(f"\n performance results \n{self.total}\n")
        logger.info(f"\n precision results \n{self.confusion}\n")
        return self.find_best()
