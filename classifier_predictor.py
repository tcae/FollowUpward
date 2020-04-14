""" tensorflow2 keras classifier
"""
import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)  # before importing tensorflow.

import env_config as env
from env_config import Env

import os
# import pandas as pd
# import numpy as np
import timeit
# import itertools
# import math
import pickle

# Import datasets, classifiers and performance metrics
from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.metrics as km
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
# import performance_data as perfdat
import prediction_data as preddat
import perf_matrix as perfmat

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
logger.debug(f"Tensorflow version: {tf.version.VERSION}")
logger.debug(f"Keras version: {keras.__version__}")
logger.debug(__doc__)


class Predictor:

    def __init__(self, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        self.ohlcv = ohlcv
        self.targets = targets
        self.features = features
        self.params = {}
        self.scaler = dict()  # key == base str, value == scaler object
        self.kerasmodel = None
        self.path = Env.model_path
        self.mnemonic = None
        self.epoch = 0

    def mnemonic_without_epoch(self):
        "returns a string that represent the estimator class as mnemonic, e.g. to use it in file names"
        if self.mnemonic is None:
            pmem = [str(p) + "-" + str(self.params[p]) for p in self.params]
            self.mnemonic = "MLP2_" + "_".join(pmem) + "__" + self.features.mnemonic() + "__" + self.targets.mnemonic()
        return self.mnemonic

    def mnemonic_with_epoch(self):
        "returns a string that represent the estimator class as mnemonic including the predictor adaptation epoch"
        mem = self.mnemonic_without_epoch() + "_" + f"epoch{self.epoch}"
        return mem

    def path_without_epoch(self):
        """ Returns a path where several files that are related to a classifier are stored,
            e.g. keras model, scaler, parameters, prediction data.
            With that there is one folder to find everything related and also just one folder to
            delete when the classifier is obsolete.
        """
        path = self.path + self.mnemonic_without_epoch()
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except OSError:
                logger.error(f"Creation of the directory {path} failed")
                return
        return path + "/"

    def path_with_epoch(self):
        """ Returns a dictionary of full qualified filenames to store or load a classifier.
        """
        path_epoch = f"{self.path_without_epoch()}epoch{self.epoch}"
        if not os.path.isdir(path_epoch):
            try:
                os.mkdir(path_epoch)
            except OSError:
                logger.error(f"Creation of the directory {path_epoch} failed")
                return
        return path_epoch + "/"


class EpochPerformance(keras.callbacks.Callback):
    def __init__(self, classifier, patience_mistake_focus=6, patience_stop=12):
        # logger.debug("EpochPerformance: __init__")
        self.classifier = classifier
        self.missing_improvements = 0
        self.best_perf = -100.0
        self.patience_mistake_focus = patience_mistake_focus
        self.patience_stop = patience_stop
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        # logger.debug(f"EpochPerformance: on_epoch_begin: {epoch}, logs: {logs}")
        self.classifier.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        logger.info("epoch end ==> epoch: {} classifier config: {}".format(
            epoch, self.classifier.mnemonic_with_epoch()))

        logger.debug(f"on_epoch_end {ad.TRAIN}")
        (best, buy_thrsld, sell_thrsld, transactions) = self.classifier.assess_performance(Env.bases, ad.TRAIN, epoch)
        logger.info(f"{ad.TRAIN} perf: {best:5.0%}  count: {transactions} at {buy_thrsld}/{sell_thrsld}")
        logger.debug(f"on_epoch_end {ad.VAL}")
        (best, buy_thrsld, sell_thrsld, transactions) = self.classifier.assess_performance(Env.bases, ad.VAL, epoch)
        logger.info(f"{ad.VAL} perf: {best:5.0%}  count: {transactions} at {buy_thrsld}/{sell_thrsld}")

        logger.debug(f"on_epoch_end assessment done")
        if best > self.best_perf:
            self.best_perf = best
            self.missing_improvements = 0
            self.classifier.save()
        else:
            self.missing_improvements += 1
        if self.missing_improvements >= self.patience_stop:
            self.classifier.kerasmodel.stop_training = True
            logger.info("Stop training due to missing val_perf improvement since {} epochs".format(
                  self.missing_improvements))
        logger.info(f"on_epoch_end logs:{logs}")


def next_MLP_iteration(params: dict, p_inst: dict, model, level):
    activate = True
    for para in params:
        if para not in p_inst:
            activate = False
            for val in params[para]:
                p_inst[para] = val
                next_MLP_iteration(params, p_inst, model, level + 1)
            p_inst.popitem()
            break
    if activate:
        logger.debug("next MLP2: ", p_inst)
        model(p_inst)


class Classifier(Predictor):
    """Provides methods to adapt single currency performance classifiers
    """

    def __init__(self, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        super().__init__(ohlcv, features, targets)

    def filename(self):
        """ Returns a dictionary of full qualified filenames to store or load a classifier.
        """
        fname = dict()
        path_epoch = self.path_with_epoch()
        fname["scaler"] = path_epoch + self.mnemonic_with_epoch() + "_scaler.pydata"
        fname["params"] = path_epoch + self.mnemonic_with_epoch() + "_params.pydata"
        fname["keras"] = path_epoch + self.mnemonic_with_epoch() + "_keras.h5"
        return fname

    def load(self, classifier_file: str, epoch: int):
        """ loads a classifier and the corresoonding scaler.
            'classifier_file' shall be provided without path and suffix
            because the path is requested as model_path from Env andhg
            scaler and classifier are stored in different files with
            different suffix.
            Returns 'None'
        """
        self.epoch = epoch
        self.mnemonic = classifier_file
        fname = self.filename()
        logger.debug(f"load classifier: {classifier_file}, epoch: {epoch}")
        try:
            with open(fname["scaler"], "rb") as df_f:
                self.scaler = pickle.load(df_f)
                df_f.close()
                logger.debug(f"scaler loaded from {fname['scaler']}")
        except IOError:
            logger.error(f"IO-error when loading scaler from {fname['scaler']}")

        try:
            with open(fname["params"], "rb") as df_f:
                self.params = pickle.load(df_f)
                df_f.close()
                logger.debug(f"{len(self.params)} params loaded from {fname['params']}")
        except IOError:
            logger.error(f"IO-error when loading params from {fname['params']}")

        try:
            with open(fname["keras"], "rb") as df_f:
                df_f.close()

                self.kerasmodel = keras.models.load_model(fname["keras"], custom_objects=None, compile=True)
                logger.info(f"classifier loaded from {fname['keras']}")
        except IOError:
            logger.error(f"IO-error when loading classifier from {fname['keras']}")

    def save(self):
        """Saves classifier with scaler and params in seperate files in a classifier specific folder.
        """
        if self.kerasmodel is not None:
            # Save entire tensorflow keras model to a HDF5 file
            fname = self.filename()
            self.kerasmodel.save(fname["keras"])
            # keras.models.save_model(classifier, fname["keras"], overwrite=True, include_optimizer=True)
            # , save_format="tf", signatures=None)

            if self.scaler is not None:
                df_f = open(fname["scaler"], "wb")
                pickle.dump(self.scaler, df_f, pickle.HIGHEST_PROTOCOL)
                df_f.close()

            df_f = open(fname["params"], "wb")
            pickle.dump(self.params, df_f, pickle.HIGHEST_PROTOCOL)
            df_f.close()

            logger.info(f"classifier saved in {self.path_with_epoch()}")
        else:
            logger.warning(f"missing classifier - cannot save it")

    def assess_performance(self, bases: list, set_type, epoch=0):
        """Evaluates the performance on the given set and prints the confusion and
        performance matrix.

        Returns a tuple of (best-performance-factor, at-buy-probability-threshold,
        at-sell-probability-threshold, with-number-of-transactions)
        """
        logger.debug(f"assess_performance {set_type}")
        start_time = timeit.default_timer()
        pm = perfmat.PerfMatrix(epoch, set_type)
        pred = preddat.PredictionData(self)
        for base, odf, fdf, tdf in ad.AssessmentGenerator(set_type, self.ohlcv, self.features, self.targets):
            pred_np = pred.predict_batch(base, fdf)
            pm.assess_prediction_np(base, pred_np, odf, tdf)

        pm.report_assessment_np()
        tdiff = (timeit.default_timer() - start_time) / 60
        logger.info(f"performance assessment set type {set_type} time: {tdiff:.1f} min")
        return pm.best_np()

    def adapt_scaler(self, base, features_np):
        """ With just one common scaler for all bases, this turned out to be suboptimal.
            Hence, a dict of scalers is maintained per base. This is not a problem for ML
            but a base never seen before in real trading needs to get their individual scaler
            first before it can be used.
        """
        if base not in self.scaler:
            self.scaler[base] = \
                preprocessing.RobustScaler(
                    with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0), copy=True)
        self.scaler[base].fit(features_np)

    def adapt_scaler_training(self):
        """ With just one common scaler for all bases, this turned out to be suboptimal.
            Hence, a dict of scalers is maintained per base. This is not a problem for ML
            but a base never seen before in real trading needs to get their individual scaler
            first before it can be used.
        """
        logger.info(f"adapting scaler")
        for base in Env.bases:
            _, fdf, _ = ad.SplitSets.set_type_data(base, ad.TRAIN, None, self.features, None)
            if (fdf is None) or fdf.empty:
                logger.warning(f"missing {base} features")
            features_np = fdf.values
            self.adapt_scaler(base, features_np)
        logger.info(f"scaler adapted")

        start_time = timeit.default_timer()
        logger.debug("creating training batches")
        td = ad.TrainingData(self.scaler, self.features, self.targets)
        td.create_training_datasets()
        tdiff = (timeit.default_timer() - start_time)
        meta = td.load_meta()
        cnt = meta["samples"]
        logger.info(f"training set creation time: {(tdiff / 60):.0f} min = {(tdiff / cnt):.0f}s/sample")

        return features_np.shape[1]

    def adapt_keras(self):

        # def MLP2(x, y, x_val, y_val, params):
        def MLP2(params):
            keras.backend.clear_session()  # done by switch in talos Scan command
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
            model.add(keras.layers.Dense(3, activation=params["last_activation"]))

            model.compile(optimizer=params["optimizer"],
                          loss="categorical_crossentropy",
                          metrics=["accuracy", km.Precision()])
            self.kerasmodel = model
            self.params["l1"] = params["l1_neurons"]
            self.params["do"] = params["dropout"]
            self.params["l2"] = params["l2_neurons"]
            if params["use_l3"]:
                self.params["l3"] = params["l3_neurons"]
            else:
                self.params["l3"] = "no"
            self.params["opt"] = params["optimizer"]
            out = None

            env.Tee.set_path(self.path_without_epoch(), log_prefix="TrainEval")
            callbacks = [
                EpochPerformance(self, patience_mistake_focus=5, patience_stop=10),
                # Interrupt training if `val_loss` stops improving for over 2 epochs
                # keras.callbacks.EarlyStopping(patience=10),
                # keras.callbacks.ModelCheckpoint(tensorfile, verbose=1),
                keras.callbacks.TensorBoard(log_dir=self.path_without_epoch())]
            training_gen = ad.TrainingGenerator(self.scaler, self.features, self.targets, shuffle=False)
            validation_gen = ad.BaseGenerator(ad.VAL, self.scaler, self.features, self.targets)

            model.summary(print_fn=logger.debug)
            out = self.kerasmodel.fit(
                    x=training_gen,
                    steps_per_epoch=len(training_gen),
                    epochs=50,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=validation_gen,
                    validation_steps=len(validation_gen),
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
        feature_elems = self.adapt_scaler_training()

        fc = len(self.features.keys())
        tc = len(self.targets.target_dict())
        assert tc == 3
        params = {
                "l1_neurons": [max(3*tc, int(0.7*fc)), max(3*tc, int(1.2*fc))],
                "l2_neurons": [max(2*tc, int(0.5*fc)), max(2*tc, int(0.8*fc))],
                "epochs": [50],
                "use_l3": [False, True],
                "l3_neurons": [max(1*tc, int(0.3*fc))],
                "kernel_initializer": ["he_uniform"],
                "dropout": [0.2, 0.45, 0.8],
                "optimizer": ["Adam"],
                "losses": ["categorical_crossentropy"],
                "activation": ["relu"],
                "last_activation": ["softmax"]}

        logger.debug(params)
        next_MLP_iteration(params, dict(), MLP2, 1)

        tdiff = (timeit.default_timer() - start_time) / 60
        logger.info(f"{env.timestr()} MLP adaptation time: {tdiff:.0f} min")
