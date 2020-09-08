""" tensorflow2 keras classifier
"""
import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)  # before importing tensorflow.

import env_config as env
from env_config import Env

import os
import pandas as pd
import numpy as np
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
import condensed_features as cof
import aggregated_features as agf
# import classify_keras as ck
import adaptation_data as ad
# import performance_data as perfdat
import prediction_data as preddat
import perf_matrix as perfmat
# import update_crypto_history as uch

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
tf.config.set_visible_devices([], 'GPU')
# tf.config.experimental.set_visible_devices([], 'GPU')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

logger = logging.getLogger(__name__)
logger.debug(f"Tensorflow version: {tf.version.VERSION}")
logger.debug(f"Keras version: {keras.__version__}")
logger.debug(__doc__)


class Predictor:

    def __init__(self, bases: list(), ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        self.ohlcv = ohlcv
        self.targets = targets
        self.features = features
        self.params = {}
        self.scaler = dict()  # key == base str, value == scaler object
        self.kerasmodel = None
        self.path = Env.model_path
        self.mnemonic = None
        self.epoch = 0
        self.bases = bases
        self.timestamp = env.timestr()

    def mnemonic_without_epoch(self):
        "returns a string that represent the estimator class as mnemonic, e.g. to use it in file names"
        if self.mnemonic is None:
            pmem = [str(p) + "-" + str(self.params[p]) for p in self.params]
            self.mnemonic = \
                "MLP2_" + self.timestamp + "__" + "_".join(pmem) + "__" + \
                self.features.mnemonic() + "__" + self.targets.mnemonic()
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
        for base in self.bases:
            _, fdf, _ = ad.SplitSets.set_type_data(base, ad.TRAIN, None, self.features, None)
            if (fdf is None) or fdf.empty:
                logger.warning(f"missing {base} features")
            features_np = fdf.values
            self.adapt_scaler(base, features_np)
        logger.info(f"scaler adapted")


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
        epoch += 1
        logger.debug(f"EpochPerformance: on_epoch_begin: {epoch}, logs: {logs}")
        self.classifier.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        logger.info("start on_epoch_end ==> epoch: {} classifier config: {}".format(
            epoch, self.classifier.mnemonic_with_epoch()))

        logger.debug(f"on_epoch_end {ad.TRAIN}")
        (best, buy_thrsld, sell_thrsld, transactions) = \
            self.classifier.assess_performance(self.classifier.bases, ad.TRAIN, epoch)
        logger.info(f"{ad.TRAIN} perf: {best:5.0%}  count: {transactions} at {buy_thrsld}/{sell_thrsld}")
        logger.debug(f"on_epoch_end {ad.VAL}")
        (best, buy_thrsld, sell_thrsld, transactions) = \
            self.classifier.assess_performance(self.classifier.bases, ad.VAL, epoch)
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
        logger.info(f"end on_epoch_end ==> epoch: {epoch} logs:{logs}")


def next_MLP_iteration(params: dict, p_inst: dict, model, level):
    """ receives a ML hyperparameter dictionary of lists with corresponding hyperparameter values,
        that are iterated through all combinations once to adapt the provided model with them.
    """
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

    def __init__(self, bases: list(), ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        super().__init__(bases, ohlcv, features, targets)

    def get_tf_data(self, fdf_tdf_list):
        # BUFFER_SIZE = 10000
        BATCH_SIZE = ad.TrainingData.training_batch_size()

        tf_data_list = list()
        maxset = 0
        for (feature_df, target_df) in fdf_tdf_list:
            maxset = max(maxset, len(feature_df))
            tnp_cat = keras.utils.to_categorical(
                target_df.values, num_classes=len(ct.TARGETS))
            ds = tf.data.Dataset.from_tensor_slices((feature_df.values, tnp_cat))
            # ds = ds.shuffle(BUFFER_SIZE).repeat()
            ds = ds.repeat()
            tf_data_list.append(ds)
        llen = len(fdf_tdf_list)
        wl = [1/llen for _ in range(llen)]
        resampled_ds = tf.data.experimental.sample_from_datasets(tf_data_list, weights=wl)
        resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
        resampled_steps_per_epoch = int(np.ceil(llen * maxset / BATCH_SIZE))
        return (resampled_steps_per_epoch, resampled_ds)

    def get_tf_data_set_type(self, set_type: str):
        td = ad.TrainingData(self.bases, self.scaler, self.features, self.targets)
        # fdf_tdf_list = [td.create_type_datasets(set_type, lbl) for lbl in [ct.BUY, ct.HOLD, ct.SELL]]
        fdf_tdf_list = [td.create_type_datasets(set_type, None)]
        return self.get_tf_data(fdf_tdf_list)

    def get_training_tf(self):
        # logger.debug("creating training batches")
        td = ad.TrainingData(self.bases, self.scaler, self.features, self.targets)
        fdf_tdf_list = [td.create_training_datasets(lbl) for lbl in [ct.BUY, ct.HOLD, ct.SELL]]
        # samples are already shuffled - no need to shuffle by tf.data
        return self.get_tf_data(fdf_tdf_list)

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
        for base, odf, fdf, tdf in ad.AssessmentGenerator(set_type, self.bases,
                                                          self.ohlcv, self.features, self.targets):
            pred_np = pred.predict_batch(base, fdf)
            pm.assess_prediction_np(base, pred_np, odf["close"].values, tdf["target"].values, odf.index.values)

        pm.report_assessment_np()
        tdiff = (timeit.default_timer() - start_time) / 60
        logger.info(f"performance assessment set type {set_type} time: {tdiff:.1f} min")
        return pm.best_np()

    def adapt_keras(self):

        # def MLP2(x, y, x_val, y_val, params):
        def MLP2(params):
            keras.backend.clear_session()  # done by switch in talos Scan command
            logger.info(f"MLP params: {params}")
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(
                params["l1_neurons"], input_dim=fc,
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
                    steps_per_epoch=train_steps,
                    epochs=50,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=eval_dataset,
                    validation_steps=eval_steps,
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
        (train_steps, train_dataset) = self.get_training_tf()
        (eval_steps, eval_dataset) = self.get_tf_data_set_type(ad.VAL)

        fc = len(self.features.keys())
        tc = len(self.targets.target_dict())
        assert tc == 3
        params = {
                "l1_neurons": [max(3*tc, int(1*fc))],  # , max(3*tc, int(1.2*fc))
                "l2_neurons": [max(2*tc, int(0.5*fc))],  # , max(2*tc, int(0.8*fc))
                "epochs": [50],
                "use_l3": [True],  # False,
                "l3_neurons": [max(1*tc, int(0.3*fc))],
                "kernel_initializer": ["he_uniform"],
                "dropout": [0.45],  # 0.2 performs worse than 0.45
                "optimizer": ["Adam"],
                "losses": ["categorical_crossentropy"],
                "activation": ["relu"],
                "last_activation": ["softmax"]}

        logger.debug(params)
        next_MLP_iteration(params, dict(), MLP2, 1)

        tdiff = (timeit.default_timer() - start_time) / 60
        logger.info(f"{env.timestr()} MLP adaptation time: {tdiff:.0f} min")


class ClassifierSet():

    def __init__(self, bases, ohlcv, features, targets):
        super().__init__()
        self.features = features
        classifier = Classifier(bases, ohlcv, self.features, targets)
        classifier.load(
            "MLP2_2020-04-24_06h35m__base-trx_l1-14_do-0.2_l2-7_l3-no_opt-Adam__F3cond14__Target10up5low30min",
            epoch=1)
        classifier.adapt_scaler_training()
        self.baseclass = {base: classifier for base in bases}

    def predict_probs(self, base, first, last):
        if base not in self.baseclass:
            logger.warning(f"no classifier found for base {base}")
            return None
        clfr = self.baseclass[base]
        pred = preddat.PredictionData(clfr)
        fdf = self.features.get_data(base, first, last)
        if (fdf is None) or fdf.empty:
            logger.warning(f"no features for {base} between {first} and {last}")
            return None
        pred_np = pred.predict_batch(base, fdf)
        pdf = pd.DataFrame(data=pred_np, index=fdf.index, columns=clfr.targets.target_dict().keys())
        return pdf

    def predict_signals(self, base, first, last, buy_threshold=0, sell_threshold=0):
        if base not in self.baseclass:
            logger.warning(f"no classifier found for base {base}")
            return None
        clfr = self.baseclass[base]
        pred = preddat.PredictionData(clfr)
        fdf = self.features.get_data(base, first, last)
        if (fdf is None) or fdf.empty:
            logger.warning(f"no features for {base} between {first} and {last}")
            return None
        pred_np = pred.predict_batch(base, fdf)
        signals = np.zeros((len(fdf), 1), dtype=np.int32)  # by default all is 0 == HOLD
        BUY = ct.TARGETS[ct.BUY]
        SELL = ct.TARGETS[ct.SELL]
        HOLD = ct.TARGETS[ct.HOLD]
        signals[(pred_np[:, BUY] > pred_np[:, HOLD]) &
                (pred_np[:, BUY] > pred_np[:, SELL]) &
                (pred_np[:, BUY] > buy_threshold)] = BUY
        signals[(pred_np[:, SELL] > pred_np[:, HOLD]) &
                (pred_np[:, SELL] > pred_np[:, BUY]) &
                (pred_np[:, SELL] > sell_threshold)] = SELL

        pdf = pd.DataFrame(data=signals, index=fdf.index, columns=["signal"])
        # ccd.dfdescribe("predict_signals", pdf)
        return pdf

    def predict_signals_test(self):
        tdat = {"close": [100,  90, 100,  99,  90,  91, 120, 121, 130, 129, 110, 111, 100,  99, 110, 120, 125.0],
                "target": [00,   0,   1,   0,   0,   2,   1,   0,   0,   0,   2,   1,   0,   0,  1,    1,   1],
                ct.HOLD: [0.0, 0.3, 0.4, 0.7, 0.3, 0.4, 0.4, 0.7, 0.3, 0.5, 0.2, 0.4, 0.2, 0.6, 0.1, 0.0, 0.0],
                ct.BUY:  [0.0, 0.1, 0.6, 0.1, 0.0, 0.0, 0.6, 0.0, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 0.8],
                ct.SELL: [0.0, 0.6, 0.0, 0.0, 0.7, 0.2, 0.0, 0.2, 0.0, 0.0, 0.8, 0.3, 0.4, 0.2, 0.1, 0.1, 0.2],
                "times": [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17]}
        df = pd.DataFrame(
            data=tdat,
            index=pd.date_range('2012-10-08 18:15:05', periods=17, freq='T'))
        pred_np = df.loc[:, [ct.HOLD, ct.BUY, ct.SELL]].values
        print(f"inital \n {df.head(18)}")
        print(f"pred_np \n {pred_np}")
        signals = np.zeros((len(df), 1), dtype=np.int32)  # by default all is 0 == HOLD
        BUY = ct.TARGETS[ct.BUY]
        SELL = ct.TARGETS[ct.SELL]
        HOLD = ct.TARGETS[ct.HOLD]
        signals[(pred_np[BUY] > pred_np[HOLD]) &
                (pred_np[BUY] > pred_np[SELL]) &
                (pred_np[BUY] > 0)] = BUY
        signals[(pred_np[SELL] > pred_np[HOLD]) &
                (pred_np[SELL] > pred_np[BUY]) &
                (pred_np[SELL] > 0)] = SELL


def all_data_objs(ohlcv):
    """ prevents 'import but unused' plint warnings
    """
    return [ohlcv, cof.F4CondAgg(ohlcv), ct.Targets(ohlcv)]
    return [ohlcv, cof.F3cond14(ohlcv), agf.F1agg110(ohlcv), ct.Targets(ohlcv)]


if __name__ == "__main__":
    # env.test_mode()
    bases = Env.training_bases
    # bases = ["btc"]
    ohlcv = ccd.Ohlcv()
    features = cof.F4CondAgg(ohlcv)
    # features = cof.F3cond14(ohlcv)
    # features = agf.AggregatedFeatures(ohlcv)
    targets = ct.TargetGrad1h1pct(ohlcv)  # old TargetGrad30m1pct
    # targets = ct.Target10up5low30min(ohlcv)
    # targets = ct.Target5up0low30minregr(ohlcv, features)
    if False:
        cs = ClassifierSet(bases, ohlcv, features, targets)
        cs.predict_signals_test()
    else:
        start_time = timeit.default_timer()
        try:
            classifier = Classifier(bases, ohlcv, features, targets)
            if True:
                classifier.adapt_keras()
            else:
                # classifier = convert_cpc_classifier()
                # classifier.load_classifier_file_collection(
                #     "MLP2_epoch=15_talos_iter=3_l1=14_do=0.2_l2=16_l3=no_opt=Adam__F2cond20__T10up5low30min")
                # ("MLP2_talos_iter-3_l1-14_do-0.2_l2-16_l3-no_opt-Adam__F2cond20__T10up5low30min", epoch=15)
                classifier.load(
                    "MLP2_2020-04-24_06h35m__base-trx_l1-14_do-0.2_l2-7_l3-no_opt-Adam__F3cond14__Target10up5low30min",
                    epoch=1)
                classifier.adapt_scaler_training()
                env.Tee.set_path(classifier.path_without_epoch(), log_prefix="TrainEval")
                # classifier.save()
                # MLP2_epoch=0_talos_iter=0_l1=16_do=0.2_h=19_no=l3_opt=optAdam__F2cond20__T10up5low30min_0")
                # perf = PerformanceData(PredictionData(classifier))
                # for base in [base]:
                #     features_df = features.load_data(base)
                #     print(features_df.index[0], features_df.index[-1])
                #     perf_df = perf.get_data(base, features_df.index[0], features_df.index[-1], use_cache=False)
                #     perf.save_data(base, perf_df)

                print("return new: ", classifier.assess_performance(bases, ad.VAL, epoch=1))
                # print("return new: ", classifier.assess_performance(["xrp"], ad.VAL))
                env.Tee.reset_path()

        except KeyboardInterrupt:
            logger.info("keyboard interrupt")
        tdiff = (timeit.default_timer() - start_time) / 60
        logger.info(f"total script time: {tdiff:.0f} min")
