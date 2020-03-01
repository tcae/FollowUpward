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


class Estimator:

    def __init__(self, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        self.ohlcv = ohlcv
        self.targets = targets
        self.features = features
        self.params = {}
        self.scaler = None
        self.classifier = None
        self.estimator = None
        self.training_gen = ad.TrainingGenerator(self.scaler, features, targets, shuffle=False)
        self.validation_gen = ad.ValidationGenerator(ad.VAL, self.scaler, features, targets)

    def mnemonic(self):
        "returns a string that represent the estimator class as mnemonic, e.g. to use it in file names"
        pmem = [str(p) + "=" + str(self.params[p]) for p in self.params]
        mem = "MLP2_" + "_".join(pmem)
        return mem


class PredictionData(ccd.CryptoData):

    def __init__(self, estimator_obj: Estimator):
        self.est = estimator_obj

    def history(self):
        "no history minutes are requires to calculate prediction data"
        return 0

    def keys(self):
        "returns the list of element keys"
        return self.est.targets.target_dict().keys()

    def mnemonic(self, base: str):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        mem = base + "_" + self.est.mnemonic() + "__"
        mem += self.est.features.mnemonic() + "__" + self.est.targets.mnemonic()
        return mem

    def new_data(self, base, lastdatetime, minutes):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        fdf = self.est.features.get_data(base, lastdatetime, minutes)
        tdf = self.est.targets.get_data(base, lastdatetime, minutes)
        df = pd.concat([fdf, tdf], axis=1, join="inner", keys=["features", "targets"])
        for tk in self.est.targets.target_dict():
            df[("pred", tk)] = 0.0  # prepare for predictioins with same index
        fdf = df["features"]  # inner join ensures shared samples
        tdf = df["targets"]
        data, target = ad.prepare4keras(self.est.scaler, fdf, tdf, self.est.targets)
        pred = self.est.predict_on_batch(data)
        df["pred"] = pred
        return self.check_timerange(df, lastdatetime, minutes)

        # df_ages["age_by_decade"] = pd.cut(x=df_ages["age"], bins=[20, 29, 39, 49], labels=["20s", "30s", "40s"])
        # ! introduce quantiles in another code area beside pred


class EvalPerf:
    """Evaluates the performance of a buy/sell probability threshold
    """

    def __init__(self, buy_prob_threshold, sell_prob_threshold):
        """receives minimum thresholds to consider the transaction
        """
        self.bpt = buy_prob_threshold
        self.spt = sell_prob_threshold
        self.transactions = 0
        self.open_transaction = False
        self.performance = 0

    def add_trade_signal(self, prob, close_price, signal):
        if signal == ct.TARGETS[ct.BUY]:
            if not self.open_transaction:
                if prob >= self.bpt:
                    self.open_transaction = True
                    self.open_buy = close_price * (1 + ct.FEE)
                    self.highest = close_price
        elif signal == ct.TARGETS[ct.SELL]:
            if self.open_transaction:
                if prob >= self.spt:
                    self.open_transaction = False
                    gain = (close_price * (1 - ct.FEE) - self.open_buy) / self.open_buy
                    self.performance += gain
                    self.transactions += 1
        elif signal == ct.TARGETS[ct.HOLD]:
            pass

    def __str__(self):
        return "bpt: {:<5.2} spt: {:<5.2} perf: {:>5.0%} #trans.: {:>4}".format(
              self.bpt, self.spt, self.performance, self.transactions)


class PerfMatrix:
    """Evaluates the performance across a range of buy/sell thresholds
    """

    def __init__(self, epoch, set_type="unknown set type"):

        self.p_range = range(6, 10)
        self.perf = np.zeros((len(self.p_range), len(self.p_range)), dtype=EvalPerf)
        for bp in self.p_range:
            for sp in self.p_range:
                self.perf[bp - self.p_range[0], sp - self.p_range[0]] = \
                    EvalPerf(float((bp)/10), float((sp)/10))
        self.confusion = np.zeros((ct.TARGET_CLASS_COUNT, ct.TARGET_CLASS_COUNT), dtype=int)
        self.epoch = epoch
        self.set_type = set_type
        self.start_ts = timeit.default_timer()
        self.end_ts = None
        self.descr = list()  # list of descriptions that contribute to performance

    def pix(self, bp, sp):
        return self.perf[bp - self.p_range[0], sp - self.p_range[0]]

    def print_result_distribution(self):
        for bp in self.p_range:
            for sp in self.p_range:
                print(self.pix(bp, sp))

    def best(self):
        """Returns a tuple of (best-performance-factor, at-buy-probability-threshold,
        at-sell-probability-threshold, with-number-of-transactions)
        """
        best = float(self.perf[0, 0].performance)
        hbp = hsp = self.p_range[0]
        for bp in self.p_range:
            for sp in self.p_range:
                if best < self.pix(bp, sp).performance:
                    best = self.pix(bp, sp).performance
                    hbp = bp
                    hsp = sp
        bpt = float(hbp/10)
        spt = float(hsp/10)
        t = self.pix(hbp, hsp).transactions
        return (best, bpt, spt, t)

    def __str__(self):
        (best, bpt, spt, t) = self.best()
        return f"epoch {self.epoch}, best performance {best:6.0%}" + \
            f" with buy threshold {bpt:.1f} / sell threshold {spt:.1f} at {t} transactions"

    def add_signal(self, prob, close_price, signal, target):
        assert (prob >= 0) and (prob <= 1), \
                print(f"PerfMatrix add_signal: unexpected probability {prob}")
        if signal in ct.TARGETS.values():
            for bp in self.p_range:
                for sp in self.p_range:
                    self.pix(bp, sp).add_trade_signal(prob, close_price, signal)
            if target not in ct.TARGETS.values():
                print(f"PerfMatrix add_signal: unexpected target result {target}")
                return
            self.confusion[signal, target] += 1
        else:
            raise ValueError(f"PerfMatrix add_signal: unexpected class result {signal}")

    def assess_sample_prediction(self, pred, skl_close, skl_target, skl_tics, skl_descr):
        """Assess the highest probability of a class prediction
        to find the optimum buy/sell threshold. No consideration of time gaps or open transactions
        at the end of the sample sequence
        """
        pred_cnt = len(pred)
        # begin = env.timestr(skl_tics[0])
        for sample in range(pred_cnt):
            high_prob_cl = 0
            for cl in range(len(pred[0])):
                if pred[sample, high_prob_cl] < pred[sample, cl]:
                    high_prob_cl = cl
            self.add_signal(pred[sample, high_prob_cl], skl_close[sample],
                            high_prob_cl, skl_target[sample])
        self.descr.append(skl_descr)

    def assess_prediction(self, pred, skl_close, skl_target, skl_tics, skl_descr):
        """Assess the highest probability of a class prediction
        to find the optimum buy/sell threshold.

        handling of time gaps: in case of a time gap the last value of the time slice is taken
        to close any open transaction

        """
        pred_cnt = len(pred)
        # begin = env.timestr(skl_tics[0])
        for sample in range(pred_cnt):
            if (sample + 1) >= pred_cnt:
                self.add_signal(1, skl_close[sample], ct.TARGETS[ct.SELL], ct.TARGETS[ct.SELL])
                # end = env.timestr(skl_tics[sample])
                # print("assessment between {} and {}".format(begin, end))
            elif (skl_tics[sample+1] - skl_tics[sample]) > np.timedelta64(1, "m"):
                self.add_signal(1, skl_close[sample], ct.TARGETS[ct.SELL], ct.TARGETS[ct.SELL])
                # end = env.timestr(skl_tics[sample])
                # print("assessment between {} and {}".format(begin, end))
                # begin = env.timestr(skl_tics[sample+1])
            else:
                high_prob_cl = 0

                for cl in range(len(pred[0])):
                    if pred[sample, high_prob_cl] < pred[sample, cl]:
                        high_prob_cl = cl
                self.add_signal(pred[sample, high_prob_cl], skl_close[sample],
                                high_prob_cl, skl_target[sample])
        self.descr.append(skl_descr)

    def conf(self, estimate_class, target_class):
        elem = self.confusion[estimate_class, target_class]
        targets = 0
        estimates = 0
        for i in ct.TARGETS:
            targets += self.confusion[estimate_class, ct.TARGETS[i]]
            estimates += self.confusion[ct.TARGETS[i], target_class]
        return (elem, elem/estimates, elem/targets)

    def report_assessment(self):
        self.end_ts = timeit.default_timer()
        tdiff = (self.end_ts - self.start_ts) / 60
        print("")
        print(f"{env.timestr()} {self.set_type} performance assessment time: {tdiff:.1f}min")

        def pt(bp, sp): return (self.pix(bp, sp).performance, self.pix(bp, sp).transactions)

        print(self)
        print("target:    {: >7}/est%/tgt% {: >7}/est%/tgt% {: >7}/est%/tgt%".format(
                ct.HOLD, ct.BUY, ct.SELL))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ct.HOLD,
              *self.conf(ct.TARGETS[ct.HOLD], ct.TARGETS[ct.HOLD]),
              *self.conf(ct.TARGETS[ct.HOLD], ct.TARGETS[ct.BUY]),
              *self.conf(ct.TARGETS[ct.HOLD], ct.TARGETS[ct.SELL])))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ct.BUY,
              *self.conf(ct.TARGETS[ct.BUY], ct.TARGETS[ct.HOLD]),
              *self.conf(ct.TARGETS[ct.BUY], ct.TARGETS[ct.BUY]),
              *self.conf(ct.TARGETS[ct.BUY], ct.TARGETS[ct.SELL])))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ct.SELL,
              *self.conf(ct.TARGETS[ct.SELL], ct.TARGETS[ct.HOLD]),
              *self.conf(ct.TARGETS[ct.SELL], ct.TARGETS[ct.BUY]),
              *self.conf(ct.TARGETS[ct.SELL], ct.TARGETS[ct.SELL])))

        # self.print_result_distribution()
        print("performance matrix: estimated probability/number of buy+sell trades")
        print("     {: ^10} {: ^10} {: ^10} {: ^10} < spt".format(0.6, 0.7, 0.8, 0.9))
        print("0.6  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(6, 6), *pt(6, 7), *pt(6, 8), *pt(6, 9)))
        print("0.7  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(7, 6), *pt(7, 7), *pt(7, 8), *pt(7, 9)))
        print("0.8  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(8, 6), *pt(8, 7), *pt(8, 8), *pt(8, 9)))
        print("0.9  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(9, 6), *pt(9, 7), *pt(9, 8), *pt(9, 9)))
        print("^bpt")


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


class Classifier(Estimator):
    """Provides methods to adapt single currency performance classifiers
    """

    def __init__(self, ohlcv: ccd.Ohlcv, features: ccd.Features, targets: ct.Targets):
        super().__init__(ohlcv, features, targets)
        self.params["epoch"] = 0
        self.params["talos_iter"] = 0

    def load(self, classifier_file: str):
        """ loads a classifier and the corresoonding scaler.
            'classifier_file' shall be provided without path and suffix
            because the path is requested as model_path from Env and
            scaler and classifier are stored in different files with
            different suffix.
        """
        mpath = Env.model_path
        if classifier_file is None:
            classifier_file = self.fname()

        fname = str("{}{}{}".format(mpath, classifier_file, ".scaler_pydata"))
        try:
            with open(fname, "rb") as df_f:
                self.scaler = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"scaler loaded from {fname}")
        except IOError:
            print(f"IO-error when loading scaler from {fname}")

        fname = str("{}{}{}".format(mpath, classifier_file, ".params_pydata"))
        try:
            with open(fname, "rb") as df_f:
                self.params = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                print(f"{len(self.params)} params loaded from {fname}")
        except IOError:
            print(f"IO-error when loading params from {fname}")

        fname = str("{}{}{}".format(mpath, classifier_file, ".keras_h5"))
        try:
            with open(fname, "rb") as df_f:
                df_f.close()

                self.classifier = keras.models.load_model(fname, custom_objects=None, compile=True)
                print(f"classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

    def fname(self):
        fname = self.mnemonic() + "__" + self.features.mnemonic() + "__" + self.targets.mnemonic()
        return fname

    def save(self):
        """Saves classifier with scaler and params in seperate files.
        """
        mpath = Env.model_path

        if not os.path.isdir(mpath):
            try:
                os.mkdir(mpath)
                print(f"created {mpath}")
            except OSError:
                print(f"Creation of the directory {mpath} failed")
            else:
                print(f"Successfully created the directory {mpath}")

        if self.classifier is not None:
            fname = str("{}{}{}".format(mpath, self.fname(), ".keras_h5"))
            # Save entire tensorflow keras model to a HDF5 file
            self.classifier.save(fname)
            # keras.models.save_model(classifier, fname, overwrite=True, include_optimizer=True)
            # , save_format="tf", signatures=None)

            if self.scaler is not None:
                fname = str("{}{}{}".format(mpath, self.fname(), ".scaler_pydata"))
                df_f = open(fname, "wb")
                pickle.dump(self.scaler, df_f, pickle.HIGHEST_PROTOCOL)
                df_f.close()

            fname = str("{}{}{}".format(mpath, self.fname(), ".params_pydata"))
            df_f = open(fname, "wb")
            pickle.dump(self.params, df_f, pickle.HIGHEST_PROTOCOL)
            df_f.close()

            print(f"{env.timestr()} classifier saved in {mpath + self.fname()}")
        else:
            print(f"{env.timestr()} WARNING: missing classifier - cannot save it")

    def performance_assessment(self, set_type, epoch=0):
        """Evaluates the performance on the given set and prints the confusion and
        performance matrix.

        Returns a tuple of (best-performance-factor, at-buy-probability-threshold,
        at-sell-probability-threshold, with-number-of-transactions)
        """
        pm = PerfMatrix(epoch, set_type)
        assessment = ad.AssessmentGenerator(set_type, self.scaler, self.ohlcv, self.features, self.targets)
        for odf, fdf, tdf in assessment:
            if self.scaler is not None:
                fdf_scaled = self.scaler.transform(fdf.values)
            pred = self.classifier.predict_on_batch(fdf_scaled)
            pm.assess_prediction(pred, odf.close.values, tdf.target.values, odf.index.values, self.mnemonic())
        pm.report_assessment()
        return pm.best()

    def adapt_keras(self):

        def MLP1(x, y, x_val, y_val, params):
            keras.backend.clear_session()  # done by switch in talos Scan command
            self.params["talos_iter"] += 1
            print(f"talos iteration: {self.params['talos_iter']}")
            model = keras.models.Sequential()
            model.add(keras.layers.Dense(
                params["l1_neurons"], input_dim=samples.shape[1],
                kernel_initializer=params["kernel_initializer"], activation=params["activation"]))
            # model.add(keras.layers.Dropout(params["dropout"]))
            model.add(keras.layers.Dense(
                int(params["l2_neurons"]),
                kernel_initializer=params["kernel_initializer"], activation=params["activation"]))
            if params["use_l3"]:
                # model.add(keras.layers.Dropout(params["dropout"]))
                model.add(keras.layers.Dense(
                    int(params["l3_neurons"]),
                    kernel_initializer=params["kernel_initializer"],
                    activation=params["activation"]))
            model.add(keras.layers.Dense(3, activation=params["last_activation"]))

            model.compile(optimizer=params["optimizer"],
                          loss="categorical_crossentropy",
                          metrics=["accuracy", km.Precision()])
            self.classifier = model
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
            out = self.classifier.fit_generator(
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
        if True:
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
                "use_l3": [False],
                "l3_neurons": [max(1*tc, int(0.3*fc))],
                "kernel_initializer": ["he_uniform"],
                "dropout": [0.2],  # , 0.45, switched off anyhow 0.6,
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
    classifier.classifier = cpc.classifier
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
        classifier.load("MLP2_epoch=0_talos_iter=0_l1=16_do=0.2_h=19_no=l3_opt=optAdam__F2cond20__T10up5low30min_0")
        classifier.performance_assessment(ad.VAL)
        # classifier.performance_assessment(ad.TEST)
    tee.close()
