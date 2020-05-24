""" tensorflow2 keras regressor
"""
import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)  # before importing tensorflow.

import env_config as env
from env_config import Env

# import os
import pandas as pd
import numpy as np
import timeit
# import itertools
# import math
# import pickle

# Import datasets, classifiers and performance metrics
# from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import tensorflow.keras as keras
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
import condensed_features as cof
import adaptation_data as ad
# import performance_data as perfdat
import prediction_data as preddat
# import perf_matrix as perfmat
import classifier_predictor as cp

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
# """
# tf.config.set_visible_devices([], 'GPU')
# tf.config.experimental.set_visible_devices([], 'GPU')

logger = logging.getLogger(__name__)


class PerfMatrix:
    """Evaluates the performance across a range of buy/sell thresholds
    """

    def __init__(self, epoch, set_type="unknown set type"):

        self.epoch = epoch
        self.set_type = set_type
        self.BUY = ct.TARGETS[ct.BUY]
        self.SELL = ct.TARGETS[ct.SELL]
        self.HOLD = ct.TARGETS[ct.HOLD]
        self.track = [self.PERF, self.COUNT, self.BUY_IX, self.BUY_PRICE] = [ix for ix in range(4)]
        self.btl = [0.002, 0.005, 0.01, 0.015]  # buy thresholds
        self.stl = [-0.001, -0.005, -0.01]  # sell thresholds
        self.perf = np.zeros((len(self.btl), len(self.stl), len(self.track)))  # overall performace array
        self.base_perf = dict()  # performance array for each base
        self.conf = np.zeros((len(ct.TARGETS), len(ct.TARGETS)))  # (target, actual) confusion matrix

    def _close_open_transactions_np(self, base, close_df):
        """ only corrects the performance calculation but not the confusion matrix
        """
        close_arr = close_df["close"].values
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                buy_price = self.perf[bt, st, self.BUY_PRICE]
                if buy_price > 0:
                    if self.perf[bt, st, self.BUY_IX] == (len(close_df) - 1):
                        # last transaction was opened by last sample --> revert transaction
                        self.perf[bt, st, self.BUY_PRICE] = 0
                    else:
                        # last transaction was opened before last sample --> close with sell
                        # add here the transaction tracking
                        transaction_perf = \
                            (close_arr[-1] * (1 - ct.FEE) - buy_price * (1 + ct.FEE)) / buy_price * (1 + ct.FEE)
                        self.perf[bt, st, self.PERF] += transaction_perf
                        self.perf[bt, st, self.COUNT] += 1
                        self.perf[bt, st, self.BUY_PRICE] = 0
                        self.base_perf[base][bt, st, self.PERF] += transaction_perf
                        self.base_perf[base][bt, st, self.COUNT] += 1

    def assess_prediction_np(self, base: str, pred_np, close_df):  # noqa: C901  silence McCabe warning
        """ Assesses the performance with different buy/sell thresholds. This method can be called multiple times,
            which accumulates the performance results in the class attributes 'perf' and 'conf'

            - pred_np is a tensor with numpy data returning the predictions per class
              with class index == ct.TARGETS[BUY|SELL|HOLD]
            - close_df is a data frame with a 'close' column containing close prices
        """
        pred_cnt = len(pred_np)
        close_arr = close_df["close"].values
        if base not in self.base_perf:
            self.base_perf[base] = np.zeros((len(self.btl), len(self.stl), len(self.track)))
        for sample in range(pred_cnt):
            pred = pred_np[sample]
            for st in range(len(self.stl)):
                if pred <= self.stl[st]:  # SELL candidate
                    for bt in range(len(self.btl)):
                        buy_price = self.perf[bt, st, self.BUY_PRICE]
                        if buy_price > 0:  # there is an open transaction to close
                            # add here the transaction tracking shall be inserted
                            transaction_perf = \
                                (close_arr[sample] * (1 - ct.FEE) - buy_price * (1 + ct.FEE)) \
                                / buy_price * (1 + ct.FEE)
                            self.perf[bt, st, self.PERF] += transaction_perf
                            self.perf[bt, st, self.COUNT] += 1
                            self.perf[bt, st, self.BUY_PRICE] = 0
                            self.base_perf[base][bt, st, self.PERF] += transaction_perf
                            self.base_perf[base][bt, st, self.COUNT] += 1
            for bt in range(len(self.btl)):
                if pred >= self.btl[bt]:  # BUY candidate
                    for st in range(len(self.stl)):
                        if self.perf[bt, st, self.BUY_PRICE] == 0:  # no alrady open transaction
                            # first buy of a possible sequence of multiple buys before sell
                            self.perf[bt, st, self.BUY_PRICE] = close_arr[sample]
                            self.perf[bt, st, self.BUY_IX] = sample
        self._close_open_transactions_np(base, close_df)

    def best_np(self, perf_np):
        bp = -999999.9
        bc = bbt = bst = 0
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                if perf_np[bt, st, self.PERF] > bp:
                    (bp, bc, bbt, bst) = (perf_np[bt, st, self.PERF],
                                          perf_np[bt, st, self.COUNT], self.btl[bt], self.stl[st])
        return (bp, bbt, bst, int(bc))

    def _prep_perf_mat(self, perf_np):
        perf_mat = pd.DataFrame(index=self.btl, columns=pd.MultiIndex.from_product([self.stl, ["perf", "count"]]))
        perf_mat.index.rename("bt", inplace=True)
        perf_mat.columns.rename(["st", "kpi"], inplace=True)
        for bt in range(len(self.btl)):
            for st in range(len(self.stl)):
                perf_mat.loc[self.btl[bt], (self.stl[st], "perf")] = "{:>5.0%}".format(perf_np[bt, st, self.PERF])
                perf_mat.loc[self.btl[bt], (self.stl[st], "count")] = int(perf_np[bt, st, self.COUNT])
        return perf_mat

    def report_assessment_np(self):
        (bp, bbt, bst, bc) = self.best_np(self.perf)
        logger. info(
            f"best {self.set_type} performance overall: {bp:>5.0%} at bt/st {bbt}/{bst} with {bc} transactions")
        # logger.info(f"performance matrix np: \n{self.perf}")
        logger.info(f"{self.set_type} performance matrix overall: \n{self._prep_perf_mat(self.perf)}")
        for base in self.base_perf:
            (bp, bbt, bst, bc) = self.best_np(self.base_perf[base])
            logger. info(
                f"best {self.set_type} performance of {base}: {bp:>5.0%} at bt/st {bbt}/{bst} with {bc} transactions")
            logger.info(f"{self.set_type} performance matrix of {base}: \n{self._prep_perf_mat(self.base_perf[base])}")


class EpochPerformance(keras.callbacks.Callback):
    def __init__(self, regressor, patience_mistake_focus=6, patience_stop=12):
        # logger.debug("EpochPerformance: __init__")
        self.regressor = regressor
        self.missing_improvements = 0
        self.best_perf = -100.0
        self.patience_mistake_focus = patience_mistake_focus
        self.patience_stop = patience_stop
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        # logger.debug(f"EpochPerformance: on_epoch_begin: {epoch}, logs: {logs}")
        self.regressor.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        logger.info("epoch end ==> epoch: {} regressor config: {}".format(
            epoch, self.regressor.mnemonic_with_epoch()))

        logger.debug(f"on_epoch_end {ad.TRAIN}")
        (best, buy_thrsld, sell_thrsld, transactions) = \
            self.regressor.assess_performance(self.regressor.bases, ad.TRAIN, epoch)
        logger.info(f"{ad.TRAIN} perf: {best:5.0%}  count: {transactions} at {buy_thrsld}/{sell_thrsld}")
        logger.debug(f"on_epoch_end {ad.VAL}")
        (best, buy_thrsld, sell_thrsld, transactions) = \
            self.regressor.assess_performance(self.regressor.bases, ad.VAL, epoch)
        logger.info(f"{ad.VAL} perf: {best:5.0%}  count: {transactions} at {buy_thrsld}/{sell_thrsld}")

        logger.debug(f"on_epoch_end assessment done")
        if best > self.best_perf:
            self.best_perf = best
            self.missing_improvements = 0
            self.regressor.save()
        else:
            self.missing_improvements += 1
        if self.missing_improvements >= self.patience_stop:
            self.regressor.kerasmodel.stop_training = True
            logger.info("Stop training due to missing val_perf improvement since {} epochs".format(
                  self.missing_improvements))
        logger.info(f"on_epoch_end logs:{logs}")


class Regressor(cp.Predictor):
    """Provides methods to adapt single currency performance classifiers
    """

    def __init__(self, bases: list(), ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        super().__init__(bases, ohlcv, features, targets)

    def get_training_tf(self):
        # logger.debug("creating training batches")
        td = ad.TrainingData(self.bases, self.scaler, self.features, self.targets)
        (fdf, tdf) = td.create_training_datasets()
        dataset = tf.data.Dataset.from_tensor_slices((fdf.values, tdf.values))
        dataset = dataset.batch(td.training_batch_size())
        return fdf.values.shape[1], dataset

    def get_tf_data_set_type(self, set_type: str):
        td = ad.TrainingData(self.bases, self.scaler, self.features, self.targets)
        (fdf, tdf) = td.create_type_datasets(set_type)
        dataset = tf.data.Dataset.from_tensor_slices((fdf.values, tdf.values))
        dataset = dataset.batch(td.training_batch_size())
        return fdf.values.shape[1], dataset

    def assess_performance(self, bases: list, set_type, epoch=0):
        """Evaluates the performance on the given set and prints the confusion and
        performance matrix.

        Returns a tuple of (best-performance-factor, at-buy-probability-threshold,
        at-sell-probability-threshold, with-number-of-transactions)
        """
        logger.debug(f"assess_performance {set_type}")
        start_time = timeit.default_timer()
        pm = PerfMatrix(epoch, set_type)
        pred = preddat.PredictionData(self)
        for base, odf, fdf, _ in ad.AssessmentGenerator(set_type, self.bases, self.ohlcv, self.features, self.targets):
            pred_np = pred.predict_batch(base, fdf)
            pred_np = keras.backend.eval(pred_np).flatten()
            # pred_df = pd.DataFrame(data={"pred": pred_np}, index=fdf.index)
            # ccd.dfdescribe(f"predictions {base}", pred_df)
            pm.assess_prediction_np(base, pred_np, odf)

        pm.report_assessment_np()
        tdiff = (timeit.default_timer() - start_time) / 60
        logger.info(f"performance assessment set type {set_type} time: {tdiff:.1f} min")
        return pm.best_np(pm.perf)

    def adapt_keras(self):

        # def MLP2(x, y, x_val, y_val, params):
        def MLP2(params):
            keras.backend.clear_session()  # done by switch in talos Scan command
            logger.info(f"MLP params: {params}")
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(
                params["l1_neurons"], input_dim=feature_elems,
                kernel_initializer=params["kernel_initializer"], activation=params["activation"]))
            model.add(keras.layers.Dropout(params["dropout"]))
            model.add(keras.layers.Dense(
                int(params["l2_neurons"]),
                kernel_initializer=params["kernel_initializer"], activation=params["activation"]))
            if params["use_l3"]:
                model.add(keras.layers.Dropout(params["dropout"]))
                model.add(keras.layers.Dense(
                    int(params["l3_neurons"]),
                    kernel_initializer=params["kernel_initializer"],
                    activation=params["activation"]))
            model.add(keras.layers.Dense(1))

            model.compile(optimizer=params["optimizer"],
                          loss=params["losses"],
                          metrics=["mae", "mse"])
            self.kerasmodel = model
            self.mnemonic = None  # reset mnemonic with new parameters
            if len(self.bases) == 1:
                # if only one base then base specific classifiers that should be part of their name
                self.params["base"] = self.bases[0]
            self.params["l1"] = params["l1_neurons"]
            self.params["do"] = params["dropout"]
            self.params["l2"] = params["l2_neurons"]
            if params["use_l3"]:
                self.params["l3"] = params["l3_neurons"]
            else:
                self.params["l3"] = "no"
            self.params["opt"] = params["optimizer"]
            self.params["loss"] = params["losses"]
            out = None

            env.Tee.set_path(self.path_without_epoch(), log_prefix="TrainEval")
            callbacks = [
                EpochPerformance(self, patience_mistake_focus=5, patience_stop=10),
                # Interrupt training if `val_loss` stops improving for over 2 epochs
                # keras.callbacks.EarlyStopping(patience=10),
                # keras.callbacks.ModelCheckpoint(tensorfile, verbose=1),
                keras.callbacks.TensorBoard(log_dir=self.path_without_epoch())]
            # training_gen = ad.TrainingGenerator(self.scaler, self.features, self.targets, shuffle=False)
            # validation_gen = ad.BaseGenerator(ad.VAL, self.scaler, self.features, self.targets)

            model.summary(print_fn=logger.debug)
            out = self.kerasmodel.fit(
                    x=train_dataset,
                    # steps_per_epoch=len(train_dataset),
                    epochs=50,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=eval_dataset,
                    # validation_steps=len(validation_gen),
                    # class_weight=None,
                    # max_queue_size=10,
                    workers=6,
                    use_multiprocessing=True,
                    shuffle=False  # True leads to frequent access of different trainings files
                    )
            assert out is not None
            env.Tee.reset_path()

            return out, model

        start_time = timeit.default_timer()
        self.adapt_scaler_training()
        (feature_elems, train_dataset) = self.get_training_tf()
        (feature_elems, eval_dataset) = self.get_tf_data_set_type(ad.VAL)

        feature_elems = len(self.features.keys())
        params = {
                "l1_neurons": [int(1*feature_elems)],  # , int(1.2*feature_elems)],
                "l2_neurons": [int(0.5*feature_elems)],  # , int(0.8*feature_elems)],
                "epochs": [50],
                "use_l3": [False],  # , True],
                "l3_neurons": [int(0.3*feature_elems)],
                "kernel_initializer": ["he_uniform"],
                "dropout": [0.2],  # , 0.45, 0.8],
                "optimizer": ["Adam"],
                "losses": ["mse"],
                "activation": ["relu"]}

        logger.debug(params)
        cp.next_MLP_iteration(params, dict(), MLP2, 1)

        tdiff = (timeit.default_timer() - start_time) / 60
        logger.info(f"{env.timestr()} MLP adaptation time: {tdiff:.0f} min")


if __name__ == "__main__":
    # env.test_mode()
    start_time = timeit.default_timer()
    ohlcv = ccd.Ohlcv()
    targets = ct.Gain10up5low30min(ohlcv)
    # targets = ct.T10up5low30min(ohlcv)
    try:

        features = cof.F3cond14(ohlcv)
        for base in Env.bases:
            regressor = Regressor([base], ohlcv, features, targets)
            if True:
                regressor.adapt_keras()
            else:
                # regressor = convert_cpc_classifier()
                # regressor.load_classifier_file_collection(
                #     "MLP2_epoch=15_talos_iter=3_l1=14_do=0.2_l2=16_l3=no_opt=Adam__F2cond20__T10up5low30min")
                # ("MLP2_talos_iter-3_l1-14_do-0.2_l2-16_l3-no_opt-Adam__F2cond20__T10up5low30min", epoch=15)
                regressor.load("MLP2_l1-9_do-0.2_l2-7_l3-no_opt-Adam__F3cond14__T10up5low30min", epoch=1)
                env.Tee.set_path(regressor.path_without_epoch(), log_prefix="TrainEval")
                # regressor.save()
                # MLP2_epoch=0_talos_iter=0_l1=16_do=0.2_h=19_no=l3_opt=optAdam__F2cond20__T10up5low30min_0")
                # perf = PerformanceData(PredictionData(regressor))
                # for base in [base]:
                #     features_df = features.load_data(base)
                #     print(features_df.index[0], features_df.index[-1])
                #     perf_df = perf.get_data(base, features_df.index[0], features_df.index[-1], use_cache=False)
                #     perf.save_data(base, perf_df)

                print("return new: ", regressor.assess_performance([base], ad.VAL))
                # print("return new: ", regressor.assess_performance(["xrp"], ad.VAL))
                env.Tee.reset_path()

    except KeyboardInterrupt:
        logger.info("keyboard interrupt")
    tdiff = (timeit.default_timer() - start_time) / 60
    logger.info(f"total script time: {tdiff:.0f} min")
