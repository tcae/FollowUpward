import os
import pandas as pd
import numpy as np
import timeit
# import itertools
# import math
import pickle
# import h5py

# Import datasets, classifiers and performance metrics
from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.metrics as km
# import keras
# import keras.metrics as km
# import tensorflow.compat.v1 as tf1
import talos as ta

# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

import env_config as env
from env_config import Env
import crypto_targets as ct
# import crypto_features as cf
# import crypto_history_sets as chs
import cached_crypto_data as ccd
import condensed_features as cof
import aggregated_features as agf
import classify_keras as ck
import adaptation_data as ad


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
        self.predictor = None
        self.training_gen = ad.TrainingGenerator(self.scaler, features, targets, shuffle=False)
        self.validation_gen = ad.ValidationGenerator(ad.VAL, self.scaler, features, targets)
        self.path = Env.model_path

    def mnemonic(self):
        "returns a string that represent the estimator class as mnemonic, e.g. to use it in file names"
        pmem = [str(p) + "-" + str(self.params[p]) for p in self.params]
        mem = "MLP2_" + "_".join(pmem) + "__" + self.features.mnemonic() + "__" + self.targets.mnemonic()
        return mem

    def path(self):
        """ Returns a path where several files that are related to a classifier are stored,
            e.g. keras model, scaler, parameters, prediction data.
            With that there is one folder to find everything related and also just one folder to
            delete when the classifier is obsolete.
        """
        return self.path + self.mnemonic()


class PredictionData(ccd.CryptoData):

    def __init__(self, estimator_obj: Predictor):
        super().__init__()
        self.est = estimator_obj
        self.missing_file_warning = False
        self.path = self.est.path()

    def history(self):
        "no history minutes are requires to calculate prediction data"
        return 0

    def keys(self):
        "returns the list of element keys"
        return self.est.targets.target_dict().keys()

    def mnemonic(self):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        mem = "predictions_" + self.est.mnemonic()
        return mem

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        fdf = self.est.features.get_data(base, first, last)
        tdf = self.est.targets.get_data(base, first, last)
        if (fdf is None) or fdf.empty or (tdf is None) or tdf.empty:
            return None
        [fdf, tdf] = ccd.common_timerange([fdf, tdf])
        # odf = pd.Series(odf.close, name="close")

        if self.est.scaler is not None:
            fdf_scaled = self.est.scaler.transform(fdf.values)
        pred = self.est.classifier.predict_on_batch(fdf_scaled)
        pdf = pd.DataFrame(data=pred, index=fdf.index, columns=self.keys())
        return self.check_timerange(pdf, first, last)


class PerformanceData(ccd.CryptoData):

    def __init__(self, prediction_obj: PredictionData):
        super().__init__()
        self.pred = prediction_obj
        self.path = self.pred.path

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
        [keyl.append(lbl) for (lbl, _) in self.keyiter()]
        return keyl

    def indexlist(self):
        "returns the list of element keys"
        ixl = list()
        [ixl.append(ix) for (_, ix) in self.keyiter()]
        return ixl

    def mnemonic(self):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        mem = "performance_" + self.pred.est.mnemonic()
        return mem

    def performance_calculation(self, in_df, bt: float, st: float, fee_factor: float, verbose=False):
        """ Expect a DataFrame with columns 'close', 'buy', 'sell' and a Timerange index.
            Returns a DataFrame with a % performance column 'perf%', buy_tic and sell_tic.
            The returned DataFrame has an int range index not compatible with in_df.
        """
        in_df.index.rename("tic", inplace=True)  # debug candy
        df = in_df.reset_index().copy()  # preserve tics as data but use one numbers as index
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
            ccd.show_verbose(df, verbose)
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
            ccd.show_verbose(df, verbose)
            df = df.drop_duplicates("sell_ix", keep="first")
            df["perf"] = (df.sell_price * (1 - fee_factor) - df.buy_price * (1 + fee_factor)) / df.buy_price * 100
            df = df.reset_index(drop=True)  # eye candy
            ccd.show_verbose(df, verbose)
        else:
            df = df.rename(columns={"close": "buy_price", "tic": "buy_tic", "ix": "buy_ix"})
            df = df.drop(df.index[-1])
        return df

    def new_data_backup(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        odf = self.pred.est.ohlcv.get_data(base, first, last)
        pred_df = self.pred.get_data(base, first, last, use_cache)  # enforce recalculation
        if (odf is None) or odf.empty or (pred_df is None) or pred_df.empty:
            return None
        pred_df = pred_df[["buy", "sell"]]
        cdf = pd.concat([odf.close, pred_df.buy, pred_df.sell], axis=1, join="inner")  # combi df
        perf_df = pd.DataFrame(columns=self.keys(), index=cdf.index)
        for (lbl, (bt, st)) in self.keyiter():  # (bt, st) = signal thresholds
            rdf = self.performance_calculation(cdf, bt, st, ct.FEE, verbose=False)
            perf_df.loc[:, lbl] = 0
            if not rdf.empty:
                perf_df.loc[[rdf.sell_tic.iat[stix] for stix in range(len(rdf))], lbl] = rdf["perf"].values
        return self.check_timerange(perf_df, first, last)

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        odf = self.pred.est.ohlcv.get_data(base, first, last)
        pred_df = self.pred.get_data(base, first, last, use_cache)  # enforce recalculation
        if (odf is None) or odf.empty or (pred_df is None) or pred_df.empty:
            return None
        cdf = pd.concat([odf.close, pred_df.buy, pred_df.sell], axis=1, join="inner")  # combi df
        df_list = list()
        for (lbl, (bt, st)) in self.keyiter():  # (bt, st) = signal thresholds
            rdf = self.performance_calculation(cdf, bt, st, ct.FEE, verbose=False)
            df_list.append(rdf)
        perf_df = pd.concat(df_list, join="outer", axis=0, keys=self.keys())
        return perf_df

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        return self.new_data(base, first, last, use_cache)


class EpochPerformance(keras.callbacks.Callback):
    def __init__(self, classifier, patience_mistake_focus=6, patience_stop=12):
        self.classifier = classifier
        self.missing_improvements = 0
        self.best_perf = 0
        self.patience_mistake_focus = patience_mistake_focus
        self.patience_stop = patience_stop
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.classifier.params["epoch"] = epoch

    def on_epoch_end(self, epoch, logs=None):
        print("{} epoch end ==> talos iteration: {} epoch: {} classifier config: {}".format(
                env.timestr(), self.classifier.params["talos_iter"], epoch, self.classifier.mnemonic()))
        (best, bpt, spt, transactions) = self.classifier.performance_assessment(ad.TRAIN, epoch)
        (best, bpt, spt, transactions) = self.classifier.performance_assessment(ad.VAL, epoch)
        if best > self.best_perf:
            self.best_perf = best
            self.missing_improvements = 0
            classifier.save()
        else:
            self.missing_improvements += 1
        if self.missing_improvements >= self.patience_stop:
            self.classifier.classifier.stop_training = True
            print("Stop training due to missing val_perf improvement since {} epochs".format(
                  self.missing_improvements))
        print(f"on_epoch_end logs:{logs}")


class Classifier(Predictor):
    """Provides methods to adapt single currency performance classifiers
    """

    def __init__(self, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        super().__init__(ohlcv, features, targets)
        self.params["epoch"] = 0
        self.params["talos_iter"] = 0

    def load_classifier_file_collection(self, classifier_file: str):
        """ loads a classifier and the corresoonding scaler.
            'classifier_file' shall be provided without path and suffix
            because the path is requested as model_path from Env and
            scaler and classifier are stored in different files with
            different suffix.
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

                self.predictor = keras.models.load_model(fname, custom_objects=None, compile=True)
                print(f"mpl2 classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

    def load_classifier_subdir(self, classifier_file: str):
        """ loads a classifier and the corresoonding scaler.
            'classifier_file' shall be provided without path and suffix
            because the path is requested as model_path from Env and
            scaler and classifier are stored in different files with
            different suffix.
        """
        pname = self.path()
        fname = pname + "/scaler.pydata"
        try:
            with open(fname, "rb") as df_f:
                self.scaler = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"scaler loaded from {fname}")
        except IOError:
            print(f"IO-error when loading scaler from {fname}")

        fname = pname + "/params.pydata"
        try:
            with open(fname, "rb") as df_f:
                self.params = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"{len(self.params)} params loaded from {fname}")
        except IOError:
            print(f"IO-error when loading params from {fname}")

        fname = pname + "/keras.h5"
        try:
            with open(fname, "rb") as df_f:
                df_f.close()

                self.predictor = keras.models.load_model(fname, custom_objects=None, compile=True)
                print(f"mpl3 classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

    def load(self, classifier_file: str):
        """ loads a classifier and the corresoonding scaler.
            'classifier_file' shall be provided without path and suffix
            because the path is requested as model_path from Env and
            scaler and classifier are stored in different files with
            different suffix.
            Returns 'None'
        """
        pname = self.path()
        if os.path.isdir(pname):
            self.load_classifier_subdir(classifier_file)
        else:
            self.load_classifier_file_collection(classifier_file)

    def save(self):
        """Saves classifier with scaler and params in seperate files.
        """
        if not os.path.isdir(self.path):
            try:
                os.mkdir(self.path)
                print(f"created {self.path}")
            except OSError:
                print(f"Creation of the directory {self.path} failed")
                return
            else:
                print(f"Successfully created the directory {self.path}")

        if self.predictor is not None:
            pname = self.path()
            if not os.path.isdir(pname):
                try:
                    os.mkdir(pname)
                except OSError:
                    print(f"Creation of the directory {self.path} failed")
                    return

            # Save entire tensorflow keras model to a HDF5 file
            fname = pname + "/keras.h5"
            self.predictor.save(fname)
            # keras.models.save_model(classifier, fname, overwrite=True, include_optimizer=True)
            # , save_format="tf", signatures=None)

            if self.scaler is not None:
                fname = pname + "/scaler.pydata"
                df_f = open(fname, "wb")
                pickle.dump(self.scaler, df_f, pickle.HIGHEST_PROTOCOL)
                df_f.close()

            fname = pname + "/params.pydata"
            df_f = open(fname, "wb")
            pickle.dump(self.params, df_f, pickle.HIGHEST_PROTOCOL)
            df_f.close()

            print(f"{env.timestr()} classifier saved in {pname}")
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
            df_list.append(perf_df)
        set_type_df = pd.concat(df_list, join="outer", axis=0, keys=bases)
        set_type_df = set_type_df.swaplevel(i=0, j=1, axis=0)
        print(set_type_df.describe(percentiles=[], include='all'))
        total = pd.DataFrame(columns=["perf", "count", "buy_trhld", "sell_trhld"], dtype=float)
        for kix, k in enumerate(perf.keys()):
            total.loc[kix, "perf"] = set_type_df.loc[k, "perf"].sum().round(2)
            total.loc[kix, "count"] = set_type_df.loc[k, "perf"].count().round(0)
            total.loc[kix, "buy_trhld"] = set_type_df.loc[k, "buy_trhld"].mean().round(1)
            total.loc[kix, "sell_trhld"] = set_type_df.loc[k, "sell_trhld"].mean().round(1)
            # total.loc[kix, "label"] = k
            if total.at[kix, "count"] < 10:
                print("check:", set_type_df.loc[k])
        # print(set_type_df.describe())
        # total = set_type_df.sum(numeric_only=True).to_frame(name="perf")
        # total.index.rename("ix", inplace=True)
        # total = total.reset_index()
        print(set_type_df, "\n", total)
        max_ix = total.perf.idxmax()
        res = (total.at[max_ix, "perf"],
               total.at[max_ix, "buy_trhld"],
               total.at[max_ix, "sell_trhld"],
               total.at[max_ix, "count"])
        return res

    def adapt_keras(self):

        def MLP1(x, y, x_val, y_val, params):
            keras.backend.clear_session()  # done by switch in talos Scan command
            self.params["talos_iter"] += 1
            print(f"talos iteration: {self.params['talos_iter']}")
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
            self.predictor = model
            self.params["l1"] = params["l1_neurons"]
            self.params["do"] = params["dropout"]
            self.params["l2"] = params["l2_neurons"]
            if params["use_l3"]:
                self.params["l3"] = params["l3_neurons"]
            else:
                self.params["l3"] = "no"
            self.params["opt"] = params["optimizer"]

            tensorboardpath = Env.tensorboardpath()
            tensorfile = "{}epoch{}.hdf5".format(tensorboardpath, "{epoch}")
            callbacks = [
                EpochPerformance(self, patience_mistake_focus=5, patience_stop=10),
                # Interrupt training if `val_loss` stops improving for over 2 epochs
                # keras.callbacks.EarlyStopping(patience=10),
                # keras.callbacks.ModelCheckpoint(tensorfile, verbose=1),
                keras.callbacks.TensorBoard(log_dir=tensorfile)]

            # print(model.summary())
            out = self.predictor.fit_generator(
                    self.training_gen,
                    steps_per_epoch=len(self.training_gen),
                    epochs=params["epochs"],
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=self.validation_gen,
                    validation_steps=len(self.validation_gen),
                    class_weight=None,
                    max_queue_size=10,
                    workers=6,
                    use_multiprocessing=True,
                    shuffle=True,
                    initial_epoch=0)
            assert out is not None

            return out, model

        start_time = timeit.default_timer()
        print(f"{env.timestr()} loading history sets")
        self.params["talos_iter"] = 0

        print(f"{env.timestr()} adapting scaler")
        scaler = preprocessing.StandardScaler(copy=False)
        for samples, targets in self.validation_gen:
            # print("scaler fit")
            scaler.partial_fit(samples)
        self.scaler = scaler
        print(f"{env.timestr()} scaler adapted")

        dummy_x = np.empty((1, samples.shape[1]))
        dummy_y = np.empty((1, targets.shape[1]))

        fc = len(self.features.keys())
        tc = len(self.targets.target_dict())
        assert tc == 3
        if False:
            params = {
                "l1_neurons": [60],  # [max(3*tc, int(0.7*fc)), max(3*tc, int(0.9*fc))],
                "l2_neurons": [48],  # 48 = 60 * 0.8   [max(2*tc, int(0.5*fc)), max(2*tc, int(0.8*fc))],
                "epochs": [50],
                "use_l3": [False],  # True
                "l3_neurons": [38],  # 38 = 60 * 0.8 * 0.8   [max(1*tc, int(0.3*fc))],
                "kernel_initializer": ["he_uniform"],
                "dropout": [0.2],  # , 0.45, switched off anyhow 0.6,
                "optimizer": ["Adam"],
                "losses": ["categorical_crossentropy"],
                "activation": ["relu"],
                "last_activation": ["softmax"]}
        else:
            params = {
                "l1_neurons": [max(3*tc, int(0.7*fc)), max(3*tc, int(1.2*fc))],
                "l2_neurons": [max(2*tc, int(0.5*fc)), max(2*tc, int(0.8*fc))],
                "epochs": [50],
                "use_l3": [False],  # [False, True]
                "l3_neurons": [max(1*tc, int(0.3*fc))],
                "kernel_initializer": ["he_uniform"],
                "dropout": [0.45, 0.8],  # [0.2, 0.45, 0.8]
                "optimizer": ["Adam"],
                "losses": ["categorical_crossentropy"],
                "activation": ["relu"],
                "last_activation": ["softmax"]}

        ta.Scan(x=dummy_x,  # real data comes from generator
                y=dummy_y,  # real data comes from generator
                model=MLP1,
                print_params=True,
                clear_session=True,
                params=params,
                # dataset_name="xrp-eos-bnb-btc-eth-neo-ltc-trx",
                # dataset_name="xrp_eos",
                # grid_downsample=1,
                experiment_name=f"talos_{env.timestr()}")

        # ta.Deploy(scan, "talos_lstm_x", metric="val_loss", asc=True)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{env.timestr()} MLP adaptation time: {tdiff:.0f} min")


def convert_cpc_classifier():
    load_classifier = "MLP_l1-16_do-0.2_h-19_no-l3_optAdam_F2cond24_0"
    cpc = ck.Cpc(load_classifier, None)
    classifier.predictor = cpc.classifier
    classifier.scaler = cpc.scaler
    classifier.params["l1"] = 16
    classifier.params["do"] = 0.2
    classifier.params["h"] = 19
    classifier.params["no"] = "l3"
    classifier.params["opt"] = "optAdam"
    classifier.params["opt"] = "optAdam"
    classifier.params["epoch"] = 0
    classifier.save()


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
    if False:
        classifier.adapt_keras()
    else:
        # convert_cpc_classifier()
        classifier.load("MLP2_epoch=15_talos_iter=3_l1=14_do=0.2_l2=16_l3=no_opt=Adam__F2cond20__T10up5low30min")
        # MLP2_epoch=0_talos_iter=0_l1=16_do=0.2_h=19_no=l3_opt=optAdam__F2cond20__T10up5low30min_0")
        print("return new: ", classifier.assess_performance(Env.bases, ad.VAL))
        # print("return new: ", classifier.assess_performance(["xrp"], ad.VAL))
    tdiff = (timeit.default_timer() - start_time) / 60
    print(f"{env.timestr()} total script time: {tdiff:.0f} min")
    tee.close()
