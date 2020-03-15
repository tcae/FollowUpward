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

import env_config as env
from env_config import Env
import crypto_targets as ct
import cached_crypto_data as ccd
import condensed_features as cof
import aggregated_features as agf
# import classify_keras as ck
import adaptation_data as ad

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

print(f"Tensorflow version: {tf.version.VERSION}")
print(f"Keras version: {keras.__version__}")
print(__doc__)


class Predictor:

    def __init__(self, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        self.ohlcv = ohlcv
        self.targets = targets
        self.features = features
        self.params = {}
        self.scaler = None
        self.kerasmodel = None
        self.training_gen = ad.TrainingGenerator(self.scaler, features, targets, shuffle=False)
        self.validation_gen = ad.ValidationGenerator(ad.VAL, self.scaler, features, targets)
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
                print(f"Creation of the directory {path} failed")
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
                print(f"Creation of the directory {path_epoch} failed")
                return
        return path_epoch + "/"


class PredictionData(ccd.CryptoData):

    def __init__(self, estimator_obj: Predictor):
        super().__init__()
        self.predictor = estimator_obj
        self.missing_file_warning = False
        self.path = self.predictor.path_with_epoch()

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

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=False):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        fdf = self.predictor.features.get_data(base, first, last, use_cache=True)
        tdf = self.predictor.targets.get_data(base, first, last, use_cache=True)
        if (fdf is None) or fdf.empty or (tdf is None) or tdf.empty:
            return None
        [fdf, tdf] = ccd.common_timerange([fdf, tdf])
        # ohlcv_df = pd.Series(ohlcv_df.close, name="close")

        if self.predictor.scaler is not None:
            fdf_scaled = self.predictor.scaler.transform(fdf.values)
        pred = self.predictor.kerasmodel.predict_on_batch(fdf_scaled)
        pdf = pd.DataFrame(data=pred, index=fdf.index, columns=self.keys())
        return self.check_timerange(pdf, first, last)

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=False):
        return self.new_data(base, first, last, use_cache)


class PerformanceData(ccd.CryptoData):

    def __init__(self, prediction_obj: PredictionData):
        super().__init__()
        self.predictions = prediction_obj
        self.path = self.predictions.predictor.path_with_epoch()

    def history(self):
        "no history minutes are requires to calculate prediction data"
        return 0

    def keyiter(self):
        for bt in np.linspace(0.3, 0.9, num=7):  # bt = lower bound buy signal bucket
            for st in np.linspace(0.4, 1, num=7):  # st = lower bound sell signal bucket
                yield (f"bt{bt:1.1f}/st{st:1.1f}", (round(bt, 1), round(st, 1)))

    def keys(self):
        "returns the list of element keys"
        keyl = list()
        [keyl.append((bt, st)) for (lbl, (bt, st)) in self.keyiter()]
        return keyl

    def indexlist(self):
        "returns the list of element keys"
        ixl = list()
        [ixl.append(ix) for (_, ix) in self.keyiter()]
        return ixl

    def mnemonic(self):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        mem = self.predictions.predictor.mnemonic_with_epoch() + "_performances"
        return mem

    def performance_calculation(self, in_df, bt: float, st: float, fee_factor: float, verbose=False):
        """ Expect a DataFrame with columns 'close', 'buy', 'sell' and a Timerange index.
            Returns a DataFrame with a % performance column 'perf' in %,
            buy_tic, buy_price, buy prediction, buy_threshold, sell_tic, sell_price, sell prediction, sell_threshold.
            The returned DataFrame has an int range index not compatible with in_df.
        """
        in_df.index.rename("tic", inplace=True)  # debug candy
        df = in_df.reset_index().copy()  # preserve tics as data but use numbers as index
        df.index.rename("ix", inplace=True)
        df.at[df.index[-1], "buy"] = 0  # no buy at last tic
        df.at[df.index[-1], "sell"] = 1  # forced sell at last tic
        df["sell_ix"] = pd.Series(dtype=pd.Int64Dtype())
        df["sell_tic"] = pd.Series(dtype='datetime64[ns, UTC]')
        df["sell_price"] = pd.Series(dtype=float)
        df["sell_trhld"] = st
        df["buy_trhld"] = bt
        df = df.loc[(df.buy >= bt) | (df.sell >= st)]
        # ccd.show_verbose(in_df, verbose)
        df = df.reset_index()  # preserve ix line numbers as unique reference of sell lines
        if len(df) > 1:
            # ccd.show_verbose(df, verbose)
            df["sell_ix"] = pd.Series(dtype=pd.Int64Dtype())
            df["sell_tic"] = pd.Series(dtype='datetime64[ns, UTC]')
            df.loc[(df.sell >= st), "sell_price"] = df.close
            df.loc[(df.sell >= st), "sell_ix"] = df["ix"]
            df.loc[(df.sell < st), "sell"] = np.nan  # debug candy
            df.loc[(df.sell >= st), "sell_tic"] = df["tic"]  # debug candy
            df.loc[:, "sell_ix"] = df.sell_ix.fillna(method='backfill')
            df.loc[:, "sell_price"] = df.sell_price.fillna(method='backfill')
            df.loc[:, "sell"] = df.sell.fillna(method='backfill')  # debug candy
            df.loc[:, "sell_tic"] = df.sell_tic.fillna(method='backfill')  # debug candy
            df = df.rename(columns={"close": "buy_price", "tic": "buy_tic", "ix": "buy_ix"})
            df = df.loc[(df.buy >= bt)]
            df = df.drop_duplicates("sell_ix", keep="first")
            df["perf"] = (df.sell_price * (1 - fee_factor) - df.buy_price * (1 + fee_factor)) / df.buy_price * 100
            df = df.reset_index(drop=True)  # eye candy
            # ccd.show_verbose(df, verbose)
        else:
            df = df.rename(columns={"close": "buy_price", "tic": "buy_tic", "ix": "buy_ix"})
            df = df.drop(df.index[-1])
        df = df.set_index("buy_tic")
        return df

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        ohlcv_df = self.predictions.predictor.ohlcv.get_data(base, first, last, use_cache=True)
        pred_df = self.predictions.get_data(base, first, last, use_cache=False)  # enforce recalculation
        if (ohlcv_df is None) or ohlcv_df.empty or (pred_df is None) or pred_df.empty:
            return None
        cdf = pd.concat([ohlcv_df.close, pred_df.buy, pred_df.sell], axis=1, join="inner")  # combi df
        # mix = pd.MultiIndex.from_tuples(self.keys(), names=["bt", "st"])
        # perf_df = pd.DataFrame(data=float(0.0), index=cdf.index, columns=mix)
        rdf_list = list()
        for (lbl, (bt, st)) in self.keyiter():  # (bt, st) = signal thresholds
            rdf = self.performance_calculation(cdf, bt, st, ct.FEE, verbose=False)
            # perf_df.loc[[tic for tic in rdf.index], (round(rdf.buy_trhld, 1), round(rdf.sell_trhld, 1))] = rdf.perf
            rdf_list.append(rdf)
        rdf = pd.concat(rdf_list, join="outer", axis=0, keys=self.keys(), sort=True)
        return rdf

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=False):
        return self.new_data(base, first, last, use_cache)


class EpochPerformance(keras.callbacks.Callback):
    def __init__(self, classifier, patience_mistake_focus=6, patience_stop=12):
        # print("EpochPerformance: __init__")
        self.classifier = classifier
        self.missing_improvements = 0
        self.best_perf = 0
        self.patience_mistake_focus = patience_mistake_focus
        self.patience_stop = patience_stop
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        # print(f"EpochPerformance: on_epoch_begin: {epoch}, logs: {logs}")
        self.classifier.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        print("{} epoch end ==> epoch: {} classifier config: {}".format(
                env.timestr(), epoch, self.classifier.mnemonic_with_epoch()))

        (best, transactions) = self.classifier.assess_performance(Env.bases, ad.TRAIN, epoch)
        (best, transactions) = self.classifier.assess_performance(Env.bases, ad.VAL, epoch)
        if best > self.best_perf:
            self.best_perf = best
            self.missing_improvements = 0
            classifier.save()
        else:
            self.missing_improvements += 1
        if self.missing_improvements >= self.patience_stop:
            self.classifier.kerasmodel.stop_training = True
            print("Stop training due to missing val_perf improvement since {} epochs".format(
                  self.missing_improvements))
        print(f"on_epoch_end logs:{logs}")


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

    def load_classifier_file_collection(self, classifier_file: str):
        """ loads a classifier and the corresoonding scaler.
            'classifier_file' shall be provided without path and suffix
            because the path is requested as model_path from Env and
            scaler and classifier are stored in different files with
            different suffix.

            This method is maintained for backward compatibility and is deprecated.
        """
        fname = str("{}{}{}".format(self.path, classifier_file, ".scaler_pydata"))
        try:
            with open(fname, "rb") as df_f:
                self.scaler = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"scaler loaded from {fname}")
        except IOError:
            print(f"IO-error when loading scaler from {fname}")

        fname = str("{}{}{}".format(self.path, classifier_file, ".params_pydata"))
        try:
            with open(fname, "rb") as df_f:
                self.params = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"{len(self.params)} params loaded from {fname}")
        except IOError:
            print(f"IO-error when loading params from {fname}")

        fname = str("{}{}{}".format(self.path, classifier_file, ".keras_h5"))
        try:
            with open(fname, "rb") as df_f:
                df_f.close()

                self.kerasmodel = keras.models.load_model(fname, custom_objects=None, compile=True)
                print(f"mpl2 classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")
        self.epoch = self.params["epoch"]
        del self.params["epoch"]
        print(self.params)

    def load(self, classifier_file: str, epoch: int):
        """ loads a classifier and the corresoonding scaler.
            'classifier_file' shall be provided without path and suffix
            because the path is requested as model_path from Env and
            scaler and classifier are stored in different files with
            different suffix.
            Returns 'None'
        """
        self.epoch = epoch
        self.mnemonic = classifier_file
        fname = self.filename()
        try:
            with open(fname["scaler"], "rb") as df_f:
                self.scaler = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"scaler loaded from {fname['scaler']}")
        except IOError:
            print(f"IO-error when loading scaler from {fname['scaler']}")

        try:
            with open(fname["params"], "rb") as df_f:
                self.params = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"{len(self.params)} params loaded from {fname['params']}")
        except IOError:
            print(f"IO-error when loading params from {fname['params']}")

        try:
            with open(fname["keras"], "rb") as df_f:
                df_f.close()

                self.kerasmodel = keras.models.load_model(fname["keras"], custom_objects=None, compile=True)
                print(f"classifier loaded from {fname['keras']}")
        except IOError:
            print(f"IO-error when loading classifier from {fname['keras']}")

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

            print(f"{env.timestr()} classifier saved in {self.path_with_epoch()}")
        else:
            print(f"{env.timestr()} WARNING: missing classifier - cannot save it")

    def assess_performance(self, bases: list, set_type, epoch=0):
        """ Evaluates the performance on the given set and prints the confusion and
            performance matrix.

            Returns a tuple of (best-performance, at-buy-probability-threshold,
            at-sell-probability-threshold, with-number-of-transactions)
        """
        perf = PerformanceData(PredictionData(self))
        df_list = list()
        for base in bases:
            perf_df = perf.set_type_data(base, set_type)
            if (perf_df is not None) and (not perf_df.empty):
                df_list.append(perf_df)
        if len(df_list) > 1:
            set_type_df = pd.concat(df_list, join="outer", axis=0, keys=bases)
        elif len(df_list) == 1:
            set_type_df = df_list[0]
        else:
            print(f"WARNING assess_performance: no {set_type} dataset available")
            return (0, 0)
        mix = pd.MultiIndex.from_tuples(perf.keys(), names=["bt", "st"])
        total = pd.DataFrame(index=mix, columns=["perf", "count"], dtype=int)
        idx = pd.IndexSlice  # requird for multilevel slicing
        for (lbl, (bt, st)) in perf.keyiter():  # (bt, st) = signal thresholds
            total.loc[(bt, st), "perf"] = set_type_df.loc[idx[:, bt, st], "perf"].sum().round(0)
            total.loc[(bt, st), "count"] = set_type_df.loc[idx[:, bt, st], "perf"].count().round(0)
        max_ix = total.perf.idxmax()
        res = (total.at[max_ix, "perf"],
               total.at[max_ix, "count"])
        total = total.unstack(level=0)
        print(total)
        return res

    def adapt_keras(self):

        # def MLP2(x, y, x_val, y_val, params):
        def MLP2(params):
            keras.backend.clear_session()  # done by switch in talos Scan command
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(
                params["l1_neurons"], input_dim=samples.shape[1],
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

            env.Tee.set_path(self.path_without_epoch())
            callbacks = [
                EpochPerformance(self, patience_mistake_focus=5, patience_stop=10),
                # Interrupt training if `val_loss` stops improving for over 2 epochs
                # keras.callbacks.EarlyStopping(patience=10),
                # keras.callbacks.ModelCheckpoint(tensorfile, verbose=1),
                keras.callbacks.TensorBoard(log_dir=self.path_without_epoch())]

            # print(model.summary())
            out = self.kerasmodel.fit(
                    x=self.training_gen,
                    # steps_per_epoch=len(self.training_gen),
                    epochs=50,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=self.validation_gen,
                    # validation_steps=len(self.validation_gen),
                    # class_weight=None,
                    # max_queue_size=10,
                    workers=6,
                    use_multiprocessing=True,
                    shuffle=False  # True leads to frequent access of different trainings files
                    )
            assert out is not None
            env.Tee.reset_path()

            return out, model

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
                print("next MLP2: ", p_inst)
                model(p_inst)

        start_time = timeit.default_timer()
        print(f"{env.timestr()} adapting scaler")
        scaler = preprocessing.StandardScaler(copy=False)
        for samples, targets in self.validation_gen:
            scaler.partial_fit(samples)
        self.scaler = scaler
        print(f"{env.timestr()} scaler adapted")

        fc = len(self.features.keys())
        tc = len(self.targets.target_dict())
        assert tc == 3
        params = {
                "l1_neurons": [max(3*tc, int(0.7*fc))],  # , max(3*tc, int(1.2*fc))],
                "l2_neurons": [max(2*tc, int(0.5*fc))],  # , max(2*tc, int(0.8*fc))],
                "epochs": [50],
                "use_l3": [False],  # [False, True]
                "l3_neurons": [max(1*tc, int(0.3*fc))],
                "kernel_initializer": ["he_uniform"],
                "dropout": [0.2],  # [0.2, 0.45, 0.8]
                "optimizer": ["Adam"],
                "losses": ["categorical_crossentropy"],
                "activation": ["relu"],
                "last_activation": ["softmax"]}

        # print(params)
        next_MLP_iteration(params, dict(), MLP2, 1)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{env.timestr()} MLP adaptation time: {tdiff:.0f} min")


# def convert_cpc_classifier():
#     load_classifier = "MLP_l1-16_do-0.2_h-19_no-l3_optAdam_F2cond24_0"
#     cpc = ck.Cpc(load_classifier, None)
#     classifier.kerasmodel = cpc.classifier
#     classifier.scaler = cpc.scaler
#     classifier.params["l1"] = 16
#     classifier.params["do"] = 0.2
#     classifier.params["h"] = 19
#     classifier.params["no"] = "l3"
#     classifier.params["opt"] = "optAdam"
#     classifier.epoch = 0
#     classifier.save()


if __name__ == "__main__":
    # env.test_mode()
    tee = env.Tee()
    start_time = timeit.default_timer()
    ohlcv = ccd.Ohlcv()
    targets = ct.T10up5low30min(ohlcv)
    if True:
        features = cof.F2cond20(ohlcv)
    else:
        features = agf.AggregatedFeatures(ohlcv)
    classifier = Classifier(ohlcv, features, targets)
    if True:
        classifier.adapt_keras()
    else:
        # convert_cpc_classifier()
        # classifier.load_classifier_file_collection(
        #     "MLP2_epoch=15_talos_iter=3_l1=14_do=0.2_l2=16_l3=no_opt=Adam__F2cond20__T10up5low30min")
        classifier.load("MLP2_talos_iter-3_l1-14_do-0.2_l2-16_l3-no_opt-Adam__F2cond20__T10up5low30min", epoch=15)
        env.Tee.set_path(classifier.path_without_epoch())
        # classifier.save()
        # MLP2_epoch=0_talos_iter=0_l1=16_do=0.2_h=19_no=l3_opt=optAdam__F2cond20__T10up5low30min_0")
        # perf = PerformanceData(PredictionData(classifier))
        # for base in Env.bases:
        #     features_df = features.load_data(base)
        #     print(features_df.index[0], features_df.index[-1])
        #     perf_df = perf.get_data(base, features_df.index[0], features_df.index[-1], use_cache=False)
        #     perf.save_data(base, perf_df)

        print("return new: ", classifier.assess_performance(Env.bases, ad.VAL))
        # print("return new: ", classifier.assess_performance(["xrp"], ad.VAL))
        env.Tee.reset_path()
    tdiff = (timeit.default_timer() - start_time) / 60
    print(f"{env.timestr()} total script time: {tdiff:.0f} min")
    tee.close()
