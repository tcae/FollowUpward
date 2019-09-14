"""
================================
Recognizing crypto samples
================================

An example showing how the scikit-learn can be used
to classify crypto sell/buy actions.


"""
import os
# import pandas as pd
import numpy as np
import timeit
import itertools
# import math
import pickle

# Import datasets, classifiers and performance metrics
from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import tensorflow.keras as krs
import tensorflow.keras.metrics as km
import talos as ta

import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

import crypto_targets_features as ctf

print(f"Tensorflow version: {tf.VERSION}")
print(f"Keras version: {krs.__version__}")
print(__doc__)

MODEL_PATH = f"{ctf.OTHER_PATH_PREFIX}classifier/"
TFBLOG_PATH = f"{ctf.OTHER_PATH_PREFIX}tensorflowlog/"


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
        if signal == ctf.TARGETS[ctf.BUY]:
            if not self.open_transaction:
                if prob >= self.bpt:
                    self.open_transaction = True
                    self.open_buy = close_price * (1 + ctf.FEE)
                    self.highest = close_price
        elif signal == ctf.TARGETS[ctf.SELL]:
            if self.open_transaction:
                if prob >= self.spt:
                    self.open_transaction = False
                    gain = (close_price * (1 - ctf.FEE) - self.open_buy) / self.open_buy
                    self.performance += gain
                    self.transactions += 1
        elif signal == ctf.TARGETS[ctf.HOLD]:
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
        self.confusion = np.zeros((len(ctf.TARGETS), len(ctf.TARGETS)), dtype=int)
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
        if signal in ctf.TARGETS.values():
            for bp in self.p_range:
                for sp in self.p_range:
                    self.pix(bp, sp).add_trade_signal(prob, close_price, signal)
            if target not in ctf.TARGETS.values():
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
        # begin = ctf.timestr(skl_tics[0])
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
        # begin = ctf.timestr(skl_tics[0])
        for sample in range(pred_cnt):
            if (sample + 1) >= pred_cnt:
                self.add_signal(1, skl_close[sample], ctf.TARGETS[ctf.SELL], ctf.TARGETS[ctf.SELL])
                # end = ctf.timestr(skl_tics[sample])
                # print("assessment between {} and {}".format(begin, end))
            elif (skl_tics[sample+1] - skl_tics[sample]) > np.timedelta64(1, "m"):
                self.add_signal(1, skl_close[sample], ctf.TARGETS[ctf.SELL], ctf.TARGETS[ctf.SELL])
                # end = ctf.timestr(skl_tics[sample])
                # print("assessment between {} and {}".format(begin, end))
                # begin = ctf.timestr(skl_tics[sample+1])
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
        for i in ctf.TARGETS:
            targets += self.confusion[estimate_class, ctf.TARGETS[i]]
            estimates += self.confusion[ctf.TARGETS[i], target_class]
        return (elem, elem/estimates, elem/targets)

    def report_assessment(self):
        self.end_ts = timeit.default_timer()
        tdiff = (self.end_ts - self.start_ts) / 60
        print("")
        print(f"{ctf.timestr()} {self.set_type} performace assessment time: {tdiff:.1f}min")

        def pt(bp, sp): return (self.pix(bp, sp).performance, self.pix(bp, sp).transactions)

        print(self)
        print("target:    {: >7}/est%/tgt% {: >7}/est%/tgt% {: >7}/est%/tgt%".format(
                ctf.HOLD, ctf.BUY, ctf.SELL))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ctf.HOLD,
              *self.conf(ctf.TARGETS[ctf.HOLD], ctf.TARGETS[ctf.HOLD]),
              *self.conf(ctf.TARGETS[ctf.HOLD], ctf.TARGETS[ctf.BUY]),
              *self.conf(ctf.TARGETS[ctf.HOLD], ctf.TARGETS[ctf.SELL])))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ctf.BUY,
              *self.conf(ctf.TARGETS[ctf.BUY], ctf.TARGETS[ctf.HOLD]),
              *self.conf(ctf.TARGETS[ctf.BUY], ctf.TARGETS[ctf.BUY]),
              *self.conf(ctf.TARGETS[ctf.BUY], ctf.TARGETS[ctf.SELL])))
        print("est. {: >4}: {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%} {: >7}/{:4.0%}/{:4.0%}".format(
              ctf.SELL,
              *self.conf(ctf.TARGETS[ctf.SELL], ctf.TARGETS[ctf.HOLD]),
              *self.conf(ctf.TARGETS[ctf.SELL], ctf.TARGETS[ctf.BUY]),
              *self.conf(ctf.TARGETS[ctf.SELL], ctf.TARGETS[ctf.SELL])))

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
                ctf.timestr(), self.cpc.talos_iter, epoch, self.cpc.save_classifier))
        (best, bpt, spt, transactions) = self.cpc.performance_assessment(ctf.TRAIN, epoch)
        (best, bpt, spt, transactions) = self.cpc.performance_assessment(ctf.VAL, epoch)
        if best > self.best_perf:
            self.best_perf = best
            self.missing_improvements = 0
            cpc.save()
        else:
            self.missing_improvements += 1
#        if self.missing_improvements >= self.patience_mistake_focus:
#            self.cpc.hs.use_mistakes(ctf.TRAIN)
#            print("Using training mistakes")
        if self.missing_improvements >= self.patience_stop:
            self.cpc.classifier.stop_training = True
            print("Stop training due to missing val_perf improvement since {} epochs".format(
                  self.missing_improvements))
        print(f"on_epoch_end logs:{logs}")
#        test_loss, test_acc = self.cpc.classifier.evaluate_generator(
#            base_generator(self.cpc, self.cpc.hs, ctf.VAL, 2),
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
        self.model_path = MODEL_PATH
        self.scaler = None
        self.classifier = None
        self.pmlist = list()
        self.step = 0
        self.epoch = 0
        self.hs = None  # only hs_name is saved not the whole hs object
        self.hs_name = None
        self.talos_iter = 0

    def load(self):
        load_clname = self.load_classifier
        save_clname = self.save_classifier
        mpath = self.model_path
        step = self.step
        epoch = self.epoch
        talos_iter = self.talos_iter
        fname = str("{}{}{}".format(self.model_path, self.load_classifier, ctf.PICKLE_EXT))
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

                self.classifier = tf.keras.models.load_model(fname)
                print(f"classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

        # if self.hs_name is not None:
        #     self.hs = ctf.HistorySets(self.hs_name)

    def save(self):
        """Saves the Cpc object without hs. The classifier is stored in a seperate file
        """
        classifier = self.classifier
        self.classifier = None
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
        fname = str("{}{}_{}{}".format(self.model_path, self.save_classifier,
                    self.epoch, ".h5"))
        # Save entire tensorflow keras model to a HDF5 file
        classifier.save(fname)

        fname = str("{}{}_{}{}".format(self.model_path, self.save_classifier,
                    self.epoch, ctf.PICKLE_EXT))
        df_f = open(fname, "wb")
        pickle.dump(self.__dict__, df_f, pickle.HIGHEST_PROTOCOL)
        df_f.close()

        self.classifier = classifier
        self.hs = hs
        print(f"{ctf.timestr()} classifier saved in {fname}")

    def performance_with_features(self, tfv, buy_trshld, sell_trshld):
        """Ignores targets of tfv and just evaluates the tfv features in respect
        to their close price performance.

        It returns the classified targets_features.TARGETS[class]
        if a buy or sell signal meets or exceeds the given threshold otherwise
        targets_features.TARGETS[HOLD] is returned.
        """
        if tfv.empty:
            print("performance_with_features: empty feature vector ==> HOLD signal")
            return ctf.TARGETS[ctf.HOLD]
        sample = ctf.to_scikitlearn(tfv, np_data=None, descr="?")
        if self.scaler is not None:
            sample.data = self.scaler.transform(sample.data)
        pred = self.classifier.predict_on_batch(sample.data)

        ls = len(pred) - 1
        high_prob_cl = ctf.TARGETS[ctf.HOLD]
        if pred[ls, ctf.TARGETS[ctf.BUY]] > pred[ls, ctf.TARGETS[ctf.SELL]]:
            if pred[ls, ctf.TARGETS[ctf.BUY]] > pred[ls, ctf.TARGETS[ctf.HOLD]]:
                if pred[ls, ctf.TARGETS[ctf.BUY]] >= buy_trshld:
                    high_prob_cl = ctf.TARGETS[ctf.BUY]
        else:
            if pred[ls, ctf.TARGETS[ctf.SELL]] > pred[ls, ctf.TARGETS[ctf.HOLD]]:
                if pred[ls, ctf.TARGETS[ctf.SELL]] >= sell_trshld:
                    high_prob_cl = ctf.TARGETS[ctf.SELL]
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
            samples = self.hs.features_from_targets(df, base, set_type, bix)
            if self.scaler is not None:
                samples.data = self.scaler.transform(samples.data)
            pred = self.classifier.predict_on_batch(samples.data)
            pm.assess_prediction(pred, samples.close, samples.target, samples.tics, samples.descr)
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
                    samples = hs.features_from_targets(df, base, ctf.TRAIN, bix)
                    # print(f">>> getitem: {samples.descr}", flush=True)
                    if samples is None:
                        return None, None
                    if self.scaler is not None:
                        samples.data = self.scaler.transform(samples.data)
                    # print(f"iteration_gen {base}({bix}) {bstep}(of {hs.max_steps[base]["max"]})")
                    targets = tf.keras.utils.to_categorical(samples.target,
                                                            num_classes=len(ctf.TARGETS))
                    yield samples.data, targets

    def base_generator(self, hs, set_type, epochs):
        "Generate one batch of data per base"
        for e in range(epochs):
            for base in hs.bases:
                bix = list(hs.bases.keys()).index(base)
                df = hs.set_of_type(base, set_type)
                samples = hs.features_from_targets(df, base, set_type, bix)
                # print(f">>> getitem: {samples.descr}", flush=True)
                if samples is None:
                    return None, None
                if self.scaler is not None:
                    samples.data = self.scaler.transform(samples.data)
                # print(f"base_gen {base}({bix}) {set_type}")
                targets = tf.keras.utils.to_categorical(samples.target,
                                                        num_classes=len(ctf.TARGETS))
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
            self.classifier = model
            assert model is not None

            self.classifier.compile(optimizer=params["optimizer"],
                                    loss="categorical_crossentropy",
                                    metrics=["accuracy", km.Precision()])
            self.save_classifier = "MLP-ti{}-l1{}-h{}-l3{}-do{}-opt{}".format(
                self.talos_iter, params["l1_neurons"], params["h_neuron_var"],
                params["use_l3"], params["dropout"], params["optimizer"])

            tensorboardpath = f"{TFBLOG_PATH}{ctf.timestr()}talos{self.talos_iter}-"
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
                    validation_data=self.base_generator(self.hs, ctf.VAL, gen_epochs),
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
        self.hs = ctf.HistorySets(ctf.sets_config_fname())
        self.talos_iter = 0

        scaler = preprocessing.StandardScaler(copy=False)
        for (samples, targets) in self.base_generator(self.hs, ctf.TRAIN, 1):
            # print("scaler fit")
            scaler.partial_fit(samples)
        self.scaler = scaler
        print(f"{ctf.timestr()} scaler adapted")

        dummy_x = np.empty((1, samples.shape[1]))
        dummy_y = np.empty((1, targets.shape[1]))

        params = {"l1_neurons": [60, 80, 100],
                  "h_neuron_var": [0.4, 0.6, 0.8],
                  "epochs": [50],
                  "use_l3": [True, False],
                  "dropout": [0.6, 0.8],
                  "kernel_initializer": ["he_uniform"],
                  "optimizer": ["adam", "adadelta"],
                  "losses": ["categorical_crossentropy"],
                  "activation": ["relu"],
                  "last_activation": ["softmax"]}

        ta.Scan(x=dummy_x,  # real data comes from generator
                y=dummy_y,  # real data comes from generator
                model=MLP1,
                debug=True,
                print_params=True,
                clear_tf_session=True,
                params=params,
                dataset_name="xrp-eos-bnb-btc-eth-neo-ltc-trx",
                # dataset_name="xrp_eos",
                grid_downsample=1,
                experiment_no=f"talos_{ctf.timestr()}")

        # ta.Deploy(scan, "talos_lstm_x", metric="val_loss", asc=True)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{ctf.timestr()} MLP adaptation time: {tdiff:.0f} min")

    def use_keras(self):
        start_time = timeit.default_timer()
        self.hs = ctf.HistorySets(ctf.sets_config_fname())
        self.talos_iter = 0

        self.performance_assessment(ctf.VAL, 0)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{ctf.timestr()} MLP performance assessment bulk time: {tdiff:.0f} min")

        start_time = timeit.default_timer()
        pm = PerfMatrix(0, ctf.VAL)
        for bix, base in enumerate(self.hs.bases):
            df = self.hs.set_of_type(base, ctf.VAL)
            tfv = self.hs.get_targets_features_of_base(base)
            subset_df = ctf.targets_to_features(tfv.vec, df)
            samples = ctf.to_scikitlearn(subset_df, np_data=None, descr=f"{base}")
            if self.scaler is not None:
                samples.data = self.scaler.transform(samples.data)
            pred1 = self.classifier.predict_on_batch(samples.data)

            pm.assess_prediction(pred1, samples.close, samples.target, samples.tics, samples.descr)
        pm.report_assessment()

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{ctf.timestr()} MLP performance assessment bulk split time: {tdiff:.0f} min")

        start_time = timeit.default_timer()
        pm = PerfMatrix(0, ctf.VAL)
        perf = 0
        buy_cl = 0
        last_fvec = None
        for bix, base in enumerate(self.hs.bases):
            df = self.hs.set_of_type(base, ctf.VAL)
            tfv = self.hs.get_targets_features_of_base(base)
            subset_df = ctf.targets_to_features(tfv.vec, df)
            subset_len = len(subset_df.index)
            for ix in range(subset_len):
                fvec = subset_df.iloc[[ix]]  # just row ix
                force_sell = False
                if (ix > 0) and (buy_cl > 0):
                    ts1 = last_fvec.index[0].to_pydatetime()
                    ts2 = fvec.index[0].to_pydatetime()
                    if (ts2 - ts1) != np.timedelta64(1, "m"):
                        if buy_cl > 0:  # forced sell of last_vec but fvec may already be a buy
                            close = last_fvec.at[last_fvec.index[0], "close"]
                            perf += (close * (1 - ctf.FEE) - buy_cl) / buy_cl
                            buy_cl = 0
                            print(f"forced step sell {base} on {fvec.index[0]} at {close}")
                        print("force_sell: diff({} - {}) {} min > 1 min".format(
                                ts2.strftime(ctf.DT_FORMAT),
                                ts1.strftime(ctf.DT_FORMAT),
                                (ts2 - ts1)))
                    if ix == (subset_len - 1):
                        print(f"force_sell: due to end of {base} samples")
                        force_sell = True  # of fvec

                close = fvec.at[fvec.index[0], "close"]
                cl = self.performance_with_features(fvec, 0.7, 0.7)
                if (cl != ctf.TARGETS[ctf.HOLD]) or force_sell:
                    if buy_cl > 0:
                        if (cl == ctf.TARGETS[ctf.SELL]) or force_sell:
                            perf += (close * (1 - ctf.FEE) - buy_cl) / buy_cl
                            buy_cl = 0
                            print(f"step sell {base} on {fvec.index[0]} at {close}")
                    else:
                        if cl == ctf.TARGETS[ctf.BUY]:
                            buy_cl = close * (1 + ctf.FEE)
                            print(f"step buy {base} on {fvec.index[0]} at {close}")
                last_fvec = fvec


#                assert np.array_equal(fvec.values, subset_df.iloc[[ix]].values)
#                sample = ctf.to_scikitlearn(fvec, np_data=None, descr="single sample")
#                if self.scaler is not None:
#                    sample.data = self.scaler.transform(sample.data)
#                assert np.array_equal(sample.data, samples.data[ix:ix+1])
#                pred2 = self.classifier.predict_on_batch(sample.data)
#                assert np.allclose(pred2, pred1[ix:ix+1])
#                if (ix == 0) or (ix == subset_len-1):
#                    print(f"ix {ix}, len(fvec)={len(fvec)}, len(pred2)={len(pred2)}", fvec)
#                pm.assess_sample_prediction(pred2, sample.close, sample.target, sample.tics,
#                                            sample.descr)
#        pm.report_assessment()
        print(f"performance: {perf:6.0%}")
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{ctf.timestr()} MLP performance assessment samplewise time: {tdiff:.0f} min")


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
    tee = ctf.Tee(f"{MODEL_PATH}Log_{ctf.timestr()}.txt")
    load_classifier = "MLP-ti1-l160-h0.8-l3False-do0.8-optadam_21"
    save_classifier = None  # "MLP-110-80relu-40relu-3softmax"
    #     load_classifier = str("{}{}".format(BASE, target_key))
    cpc = Cpc(load_classifier, save_classifier)
    if False:
        cpc.adapt_keras()
    else: 
        cpc.load()
        cpc.use_keras()
    tee.close()
