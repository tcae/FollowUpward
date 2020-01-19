"""
================================
Recognizing crypto samples
================================

An example showing how the scikit-learn can be used
to classify crypto sell/buy actions.


"""
import os
import pandas as pd
import numpy as np
import timeit
import itertools
# import math
import pickle
# import h5py

# Import datasets, classifiers and performance metrics
from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import tensorflow.keras as krs
import tensorflow.keras.metrics as km
# import tensorflow.compat.v1 as tf1
import talos as ta

import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

import env_config as env
from env_config import Env
import crypto_targets as ct
import crypto_features as cf
import crypto_history_sets as chs


print(f"Tensorflow version: {tf.version.VERSION}")
print(f"Keras version: {krs.__version__}")
print(__doc__)


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

    def plot_result_distribution(self):
        dtype = [("bpt", float), ("spt", float), ("performance", float), ("transactions", int)]
        values = list()
        for bp in self.p_range:
            for sp in self.p_range:
                values.append((float((bp)/10), float((sp)/10),
                               self.pix(bp, sp).performance, self.pix(bp, sp).transactions))
                perf_result = np.array(values, dtype)
                perf_result = np.sort(perf_result, order="performance")
        plt.figure()
        plt.title("transactions over performance")
        maxt = np.amax(perf_result["transactions"])
        maxp = np.amax(perf_result["performance"])
        plt.ylim((0., maxp))
        plt.ylabel("performance")
        plt.xlabel("transactions")
        plt.grid()
        xaxis = np.arange(0, maxt, maxt*0.1)
        plt.plot(xaxis, perf_result["performance"], "o-", color="g",
                 label="performance")
        plt.legend(loc="best")
        plt.show()

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
        print(f"{env.timestr()} {self.set_type} performace assessment time: {tdiff:.1f}min")

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


class EpochPerformance(tf.keras.callbacks.Callback):
    def __init__(self, cpc, patience_mistake_focus=6, patience_stop=12):
        self.cpc = cpc
        self.missing_improvements = 0
        self.best_perf = 0
        self.patience_mistake_focus = patience_mistake_focus
        self.patience_stop = patience_stop
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.cpc.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        print("{} epoch end ==> talos iteration: {} epoch: {} classifier config: {}".format(
                env.timestr(), self.cpc.talos_iter, epoch, self.cpc.save_classifier))
        (best, bpt, spt, transactions) = self.cpc.performance_assessment(chs.TRAIN, epoch)
        (best, bpt, spt, transactions) = self.cpc.performance_assessment(chs.VAL, epoch)
        if best > self.best_perf:
            self.best_perf = best
            self.missing_improvements = 0
            cpc.save()
        else:
            self.missing_improvements += 1
#        if self.missing_improvements >= self.patience_mistake_focus:
#            self.cpc.hs.use_mistakes(chs.TRAIN)
#            print("Using training mistakes")
        if self.missing_improvements >= self.patience_stop:
            self.cpc.classifier.stop_training = True
            print("Stop training due to missing val_perf improvement since {} epochs".format(
                  self.missing_improvements))
        print(f"on_epoch_end logs:{logs}")
#        test_loss, test_acc = self.cpc.classifier.evaluate_generator(
#            base_generator(self.cpc, self.cpc.hs, chs.VAL, 2),
#            steps=len(self.cpc.hs.bases),
#            max_queue_size=10,
#            workers=1,
#            use_multiprocessing=False,
#            verbose=1 )
#        print(f"validation loss: {test_loss} validation acc: {test_acc}")


class Cpc:
    """Provides methods to adapt single currency performance classifiers
    """
    def __init__(self, load_classifier, save_classifier):
        self.load_classifier = load_classifier
        self.save_classifier = save_classifier
        self.model_path = Env.model_path
        self.scaler = None
        self.classifier = None
        self.pmlist = list()
        self.step = 0
        self.epoch = 0
        self.hs = None  # only hs_name is saved not the whole hs object
        self.hs_name = None
        self.talos_iter = 0
        if load_classifier is not None:
            self.load()
        self.__prep_classifier_log(load_classifier)

    # def __del__(self):
    #     if (self.class_log is not None) and (self.class_log_fname is not None):
    #         fname = f"{Env.model_path}{self.class_log_fname}.h5"
    #         self.class_log.to_hdf(fname, self.class_log_fname, mode="w")

    def __prep_classifier_log(self, classifier_name):
        return
        if classifier_name is None:
            self.class_log = self.class_log_fname = None
        else:
            self.class_log = pd.DataFrame(
                columns=["base", "timestamp", "target", "buy_prob", "sell_prob", "hold_prob"])
            self.class_log_fname = f"{env.timestr()}_Results_{classifier_name}"

    def __log_predict_results(self, tfv, base, pred):
        return
        df = pd.DataFrame(
            columns=["hold_prob", "buy_prob", "sell_prob"],
            data=pred[:, [ct.TARGETS[ct.HOLD], ct.TARGETS[ct.BUY], ct.TARGETS[ct.SELL]]])
        df["timestamp"] = tfv.index
        df["target"] = tfv["target"]
        df["base"] = base
        self.class_log = pd.concat([self.class_log, df], ignore_index=True)

    def load(self):
        load_clname = self.load_classifier
        save_clname = self.save_classifier
        mpath = self.model_path
        step = self.step
        epoch = self.epoch
        talos_iter = self.talos_iter
        fname = str("{}{}{}".format(self.model_path, self.load_classifier, env.PICKLE_EXT))
        try:
            with open(fname, "rb") as df_f:
                tmp_dict = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()

                self.__dict__.update(tmp_dict)
                print(f"classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

        self.load_classifier = load_clname
        self.save_classifier = save_clname
        self.model_path = mpath
        self.step = step
        self.epoch = epoch
        self.talos_iter = talos_iter
        fname = str("{}{}{}".format(self.model_path, self.load_classifier, ".h5"))
        try:
            with open(fname, "rb") as df_f:
                df_f.close()

                self.classifier = tf.keras.models.load_model(fname, custom_objects=None, compile=True)
                print(f"classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

        # if self.hs_name is not None:
        #     self.hs = chs.CryptoHistorySets(self.hs_name)

    def save(self):
        """Saves the Cpc object without hs. The classifier is stored in a seperate file
        """
        classifier = self.classifier
        self.classifier = None  # because TF classifier is stored in seperate file
        hs = self.hs
        self.hs = None

        if not os.path.isdir(self.model_path):
            try:
                os.mkdir(self.model_path)
                print(f"created {self.model_path}")
            except OSError:
                print(f"Creation of the directory {self.model_path} failed")
            else:
                print(f"Successfully created the directory {self.model_path}")
        if classifier is not None:
            fname = str("{}{}_{}{}".format(self.model_path, self.save_classifier,
                        self.epoch, ".h5"))
            # Save entire tensorflow keras model to a HDF5 file
            classifier.save(fname)
            # tf.keras.models.save_model(classifier, fname, overwrite=True, include_optimizer=True)
            # , save_format="tf", signatures=None)

        fname = str("{}{}_{}{}".format(self.model_path, self.save_classifier,
                    self.epoch, env.PICKLE_EXT))
        df_f = open(fname, "wb")
        pickle.dump(self.__dict__, df_f, pickle.HIGHEST_PROTOCOL)
        df_f.close()

        self.classifier = classifier
        self.hs = hs
        print(f"{env.timestr()} classifier saved in {fname}")

    def class_predict_of_features(self, tfv, base):
        """Classifies the tfv features.
        'base' is a string that identifies the crypto used for logging purposes.

        It returns an array of probabilities that can be indexed with ct.TARGETS[x]
        for each sample.
        """
        if tfv.empty:
            print("class_predict_of_features: empty feature vector ==> 0 probs")
            return None
        sample = cf.to_scikitlearn(tfv, np_data=None, descr=base)
        if self.scaler is not None:
            sample.data = self.scaler.transform(sample.data)
        pred = self.classifier.predict_on_batch(sample.data)
        self.__log_predict_results(tfv, base, pred)
        return pred

    def class_of_features(self, tfv, buy_trshld, sell_trshld, base):
        """Ignores targets of tfv but classifies the tfv features.
        'base' is a string that identifies the crypto used for logging purposes.

        It returns the classified targets_features.TARGETS[class]
        if a buy or sell signal meets or exceeds the given threshold otherwise
        targets_features.TARGETS[HOLD] is returned.
        """
        if tfv.empty:
            print("class_of_features: empty feature vector ==> HOLD signal")
            return ct.TARGETS[ct.HOLD]
        pred = self.class_predict_of_features(tfv, base)
        probs = pred[len(pred) - 1]
        high_prob_cl = ct.TARGETS[ct.HOLD]
        if probs[ct.TARGETS[ct.BUY]] > probs[ct.TARGETS[ct.SELL]]:
            if probs[ct.TARGETS[ct.BUY]] > probs[ct.TARGETS[ct.HOLD]]:
                if probs[ct.TARGETS[ct.BUY]] >= buy_trshld:
                    high_prob_cl = ct.TARGETS[ct.BUY]
        else:
            if probs[ct.TARGETS[ct.SELL]] > probs[ct.TARGETS[ct.HOLD]]:
                if probs[ct.TARGETS[ct.SELL]] >= sell_trshld:
                    high_prob_cl = ct.TARGETS[ct.SELL]
        return high_prob_cl

    def performance_assessment(self, set_type, epoch):
        """Evaluates the performance on the given set and prints the confusion and
        performance matrix.

        Returns a tuple of (best-performance-factor, at-buy-probability-threshold,
        at-sell-probability-threshold, with-number-of-transactions)
        """
        pm = PerfMatrix(epoch, set_type)
        for bix, base in enumerate(self.hs.bases):
            df = self.hs.set_of_type(base, set_type)
            if (df is None) or (len(df) == 0):
                continue
            tfv = self.hs.features_from_targets(df)
            descr = "{} {} {} set step {}: {}".format(env.timestr(), base, set_type,
                                                      bix, cf.str_setsize(tfv))
            # print(descr)
            samples = cf.to_scikitlearn(tfv, np_data=None, descr=descr)
            if self.scaler is not None:
                samples.data = self.scaler.transform(samples.data)
            pred = self.classifier.predict_on_batch(samples.data)
            self.__log_predict_results(tfv, base, pred)
            pm.assess_prediction(pred, df.close, samples.target, samples.tics, samples.descr)
            self.hs.register_probabilties(base, set_type, pred, df)
        self.pmlist.append(pm)
        pm.report_assessment()
        return pm.best()

    def iteration_generator(self, hs, epochs):
        "Generate one batch of data"
        for e in range(epochs):  # due to prefetch of max_queue_size
            for base in hs.bases:
                bix = list(hs.bases.keys()).index(base)
                for bstep in range(hs.max_steps[base]["max"]):
                    df = hs.trainset_step(base, bstep)
                    tfv = hs.features_from_targets(df)
                    descr = "{} {} {} set step {}: {}".format(env.timestr(), base, chs.TRAIN,
                                                              bix, cf.str_setsize(tfv))
                    # print(descr)
                    samples = cf.to_scikitlearn(tfv, np_data=None, descr=descr)
                    del tfv
                    # print(f">>> getitem: {samples.descr}", flush=True)
                    if samples is None:
                        return None, None
                    if self.scaler is not None:
                        samples.data = self.scaler.transform(samples.data)
                    # print(f"iteration_gen {base}({bix}) {bstep}(of {hs.max_steps[base]["max"]})")
                    targets = tf.keras.utils.to_categorical(samples.target,
                                                            num_classes=ct.TARGET_CLASS_COUNT)
                    yield samples.data, targets

    def base_generator(self, hs, set_type, epochs):
        "Generate one batch of data per base"
        for e in range(epochs):
            for base in hs.bases:
                bix = list(hs.bases.keys()).index(base)
                df = hs.set_of_type(base, set_type)
                tfv = hs.features_from_targets(df)
                descr = "{} {} {} set step {}: {}".format(env.timestr(), base, set_type,
                                                          bix, cf.str_setsize(tfv))
                # print(descr)
                samples = cf.to_scikitlearn(tfv, np_data=None, descr=descr)
                del tfv
                # print(f">>> getitem: {samples.descr}", flush=True)
                if samples is None:
                    return None, None
                if self.scaler is not None:
                    samples.data = self.scaler.transform(samples.data)
                # print(f"base_gen {base}({bix}) {set_type}")
                targets = tf.keras.utils.to_categorical(samples.target,
                                                        num_classes=ct.TARGET_CLASS_COUNT)
                yield samples.data, targets

    def adapt_keras(self):

        def MLP1(x, y, x_val, y_val, params):
            krs.backend.clear_session()  # done by switch in talos Scan command
            self.talos_iter += 1
            print(f"talos iteration: {self.talos_iter}")
            model = krs.models.Sequential()
            model.add(krs.layers.Dense(params["l1_neurons"],
                                       input_dim=samples.shape[1],
                                       kernel_initializer=params["kernel_initializer"],
                                       activation=params["activation"]))
            model.add(krs.layers.Dropout(params["dropout"]))
            model.add(krs.layers.Dense(int(params["l1_neurons"]*params["h_neuron_var"]),
                                       kernel_initializer=params["kernel_initializer"],
                                       activation=params["activation"]))
            if params["use_l3"]:
                model.add(krs.layers.Dropout(params["dropout"]))
                model.add(krs.layers.Dense(
                    int(params["l1_neurons"]*params["h_neuron_var"]*params["h_neuron_var"]),
                    kernel_initializer=params["kernel_initializer"],
                    activation=params["activation"]))
            model.add(krs.layers.Dense(3,
                                       activation=params["last_activation"]))

            model.compile(optimizer=params["optimizer"],
                          loss="categorical_crossentropy",
                          metrics=["accuracy", km.Precision()])
            self.classifier = model
            self.save_classifier = "MLP_l1-{}_do-{}_h-{}_l3-{}_opt{}".format(
                params["l1_neurons"], params["dropout"], params["h_neuron_var"],
                params["use_l3"], params["optimizer"])
            self.__prep_classifier_log(self.save_classifier)

            tensorboardpath = Env.tensorboardpath()
            tensorfile = "{}epoch{}.hdf5".format(tensorboardpath, "{epoch}")
            callbacks = [
                EpochPerformance(self, patience_mistake_focus=5, patience_stop=10),
                # Interrupt training if `val_loss` stops improving for over 2 epochs
                # krs.callbacks.EarlyStopping(patience=10),
                # krs.callbacks.ModelCheckpoint(tensorfile, verbose=1),
                krs.callbacks.TensorBoard(log_dir=tensorfile)]

            steps_per_epoch = self.hs.label_steps()
            epochs = params["epochs"]
            gen_epochs = epochs * 2  # due to max_queue_size more step data is requested than needed

            # print(model.summary())
            out = self.classifier.fit_generator(
                    self.iteration_generator(self.hs, gen_epochs),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=self.base_generator(self.hs, chs.VAL, gen_epochs),
                    validation_steps=len(self.hs.bases),
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=False,
                    initial_epoch=0)
            assert out is not None

            return out, model

        start_time = timeit.default_timer()
        self.hs = chs.CryptoHistorySets(env.sets_config_fname())
        self.talos_iter = 0

        scaler = preprocessing.StandardScaler(copy=False)
        for (samples, targets) in self.base_generator(self.hs, chs.TRAIN, 1):
            # print("scaler fit")
            scaler.partial_fit(samples)
        self.scaler = scaler
        print(f"{env.timestr()} scaler adapted")

        dummy_x = np.empty((1, samples.shape[1]))
        dummy_y = np.empty((1, targets.shape[1]))

        params = {"l1_neurons": [60, 80, 100],
                  "h_neuron_var": [0.8],  # 0.4, 0.6,
                  "epochs": [50],
                  "use_l3": [False],  # True
                  "kernel_initializer": ["he_uniform"],
                  "dropout": [0.4],  # 0.6,
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

    def classify_batch(self):
        start_time = timeit.default_timer()
        self.hs = chs.CryptoHistorySets(env.sets_config_fname())
        self.talos_iter = 0

        self.performance_assessment(chs.VAL, 0)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{env.timestr()} MLP performance assessment bulk time: {tdiff:.0f} min")

        start_time = timeit.default_timer()
        pm = PerfMatrix(0, chs.VAL)
        for bix, base in enumerate(self.hs.bases):
            df = self.hs.set_of_type(base, chs.VAL)
            tfv = self.hs.features_from_targets(df)
            samples = cf.to_scikitlearn(tfv, np_data=None, descr=f"{base}")
            if (samples is None) or (samples.data is None) or (len(samples.data) == 0):
                print(
                    "skipping {} len(VAL): {} len(tfv): {} len(subset): {} len(samples)"
                    .format(base, len(df), len(tfv), len(tfv), (len(samples.data))))
                continue
            if self.scaler is not None:
                samples.data = self.scaler.transform(samples.data)
            pred1 = self.classifier.predict_on_batch(samples.data)

            pm.assess_prediction(pred1, df.close, samples.target, samples.tics, samples.descr)
        pm.report_assessment()

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{env.timestr()} MLP performance assessment bulk split time: {tdiff:.0f} min")

    # def classify_per_sample(self):
    #     start_time = timeit.default_timer()
    #     # pm = PerfMatrix(0, chs.VAL)
    #     perf = 0
    #     buy_cl = 0
    #     last_fvec = None
    #     for bix, base in enumerate(self.hs.bases):
    #         df = self.hs.set_of_type(base, chs.VAL)
    #         tfv = self.hs.get_targets_features_of_base(base)
    #         subset_df = cf.targets_to_features(tfv, df)
    #         subset_len = len(subset_df.index)
    #         for ix in range(subset_len):
    #             fvec = subset_df.iloc[[ix]]  # just row ix
    #             force_sell = False
    #             if (ix > 0) and (buy_cl > 0):
    #                 ts1 = last_fvec.index[0].to_pydatetime()
    #                 ts2 = fvec.index[0].to_pydatetime()
    #                 if (ts2 - ts1) != np.timedelta64(1, "m"):
    #                     if buy_cl > 0:  # forced sell of last_vec but fvec may already be a buy
    #                         close = last_fvec.at[last_fvec.index[0], "close"]
    #                         perf += (close * (1 - ct.FEE) - buy_cl) / buy_cl
    #                         buy_cl = 0
    #                         print(f"forced step sell {base} on {fvec.index[0]} at {close}")
    #                     print("force_sell: diff({} - {}) {} min > 1 min".format(
    #                             ts2.strftime(Env.dt_format),
    #                             ts1.strftime(Env.dt_format),
    #                             (ts2 - ts1)))
    #                 if ix == (subset_len - 1):
    #                     print(f"force_sell: due to end of {base} samples")
    #                     force_sell = True  # of fvec

    #             close = fvec.at[fvec.index[0], "close"]
    #             cl = self.class_of_features(fvec, 0.7, 0.7, base)
    #             if (cl != ct.TARGETS[ct.HOLD]) or force_sell:
    #                 if buy_cl > 0:
    #                     if (cl == ct.TARGETS[ct.SELL]) or force_sell:
    #                         perf += (close * (1 - ct.FEE) - buy_cl) / buy_cl
    #                         buy_cl = 0
    #                         print(f"step sell {base} on {fvec.index[0]} at {close}")
    #                 else:
    #                     if cl == ct.TARGETS[ct.BUY]:
    #                         buy_cl = close * (1 + ct.FEE)
    #                         print(f"step buy {base} on {fvec.index[0]} at {close}")
    #             last_fvec = fvec

#                pm.assess_sample_prediction(pred2, sample.close, sample.target, sample.tics,
#                                            sample.descr)
#        pm.report_assessment()
        # print(f"performance: {perf:6.0%}")
        # tdiff = (timeit.default_timer() - start_time) / 60
        # print(f"{env.timestr()} MLP performance assessment samplewise time: {tdiff:.0f} min")

    def use_keras(self):
        self.classify_batch()
        # self.classify_per_sample()


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

# def log_confusion_matrix(epoch, logs):
#  # Use the model to predict the values from the validation dataset.
#  test_pred_raw = model.predict(test_images)
#  test_pred = np.argmax(test_pred_raw, axis=1)
#
#  # Calculate the confusion matrix.
#  cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
#  # Log the confusion matrix as an image summary.
#  figure = plot_confusion_matrix(cm, class_names=class_names)
#  cm_image = plot_to_image(figure)
#
#  # Log the confusion matrix as an image summary.
#  with file_writer_cm.as_default():
#    tf.summary.image("Confusion Matrix", cm_image, step=epoch)


if __name__ == "__main__":
    env.test_mode()
    tee = env.Tee()
    load_classifier = None  # "MLP-ti1-l160-h0.8-l3False-optAdam_9"  # "MLP-ti1-l160-h0.8-l3False-do0.8-optadam_21-v2"
    save_classifier = None
    #     load_classifier = str("{}{}".format(BASE, target_key))
    cpc = Cpc(load_classifier, save_classifier)
    if True:
        cpc.adapt_keras()
    else:
        # cpc.save()
        cpc.use_keras()
    tee.close()
