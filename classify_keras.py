"""
================================
Recognizing crypto samples
================================

An example showing how the scikit-learn can be used
to classify crypto sell/buy actions.


"""
import numpy as np
import timeit
# import math
import pickle

# Import datasets, classifiers and performance metrics
from sklearn import svm, metrics, preprocessing
# from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import tensorflow.keras as krs
import talos as ta

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

import crypto_targets_features as ctf

print(f"Tensorflow version: {tf.VERSION}")
print(f"Keras version: {krs.__version__}")
print(__doc__)

MODEL_PATH = '/Users/tc/crypto/classifier'
LOG_PATH = '/Users/tc/crypto/tensorflowlog'


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, pre_dispatch="2*n_jobs", train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


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
                    gain = (close_price * (1 - ctf.FEE) - self.open_buy) /self.open_buy
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
        self.perf = np.zeros((len(self.p_range),len(self.p_range)), dtype=EvalPerf)
        for bp in self.p_range:
            for sp in self.p_range:
                self.perf[bp - self.p_range[0], sp - self.p_range[0]] = \
                    EvalPerf(float((bp)/10), float((sp)/10))
        self.confusion = np.zeros((len(ctf.TARGETS),len(ctf.TARGETS)), dtype=int)
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
        dtype = [('bpt', float), ('spt', float), ('performance', float), ('transactions', int)]
        values = list()
        for bp in self.p_range:
            for sp in self.p_range:
                values.append((float((bp)/10), float((sp)/10), \
                               self.pix(bp, sp).performance, self.pix(bp, sp).transactions))
                perf_result = np.array(values, dtype)
                perf_result = np.sort(perf_result, order='performance')
        plt.figure()
        plt.title("transactions over performance")
        maxt = np.amax(perf_result['transactions'])
        maxp = np.amax(perf_result['performance'])
        plt.ylim((0., maxp))
        plt.ylabel("performance")
        plt.xlabel("transactions")
        plt.grid()
        xaxis = np.arange(0, maxt, maxt*0.1)
        plt.plot(xaxis, perf_result['performance'], 'o-', color="g",
                 label="performance")
        plt.legend(loc="best")
        plt.show()

    def __str__(self):
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
        return "epoch {}, best performance {:6.0%} with buy threshold {:.1f} / sell threshold {:.1f} at {} transactions".format(self.epoch, best, bpt, spt, t)

    def add_signal(self, prob, close_price, signal, target):
        assert (prob >=0) and (prob <= 1), \
                print(f"PerfMatrix add_signal: unexpected probability {prob}")
        if signal in ctf.TARGETS.values():
            for bp in self.p_range:
                for sp in self.p_range:
                    self.pix(bp, sp).add_trade_signal(prob, close_price, signal)
            if target not in ctf.TARGETS.values():
                print(f"PerfMatrix add_signal: unexpected target result {target}")
                return
            self.confusion[signal, target] +=1
        else:
            raise ValueError(f"PerfMatrix add_signal: unexpected class result {signal}")


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
            elif (skl_tics[sample+1] - skl_tics[sample]) > np.timedelta64(1, 'm'):
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

        pt = lambda bp, sp: (self.pix(bp, sp).performance, self.pix(bp, sp).transactions)
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
                *pt(6,6), *pt(6,7), *pt(6,8), *pt(6,9)))
        print("0.7  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(7,6), *pt(7,7), *pt(7,8), *pt(7,9)))
        print("0.8  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(8,6), *pt(8,7), *pt(8,8), *pt(8,9)))
        print("0.9  {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4} {:>5.0%}/{:<4}".format(
                *pt(9,6), *pt(9,7), *pt(9,8), *pt(9,9)))
        print("^bpt")


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, cpc, set_type, single_step=True):
        #this line is just to make the generator infinite, keras needs that
        self.single_step = single_step
        self.set_type = set_type
        self.cpc = cpc
        self.steps_per_epoch = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.single_step:
            self.steps_per_epoch = len(self.cpc.hs.bases)
        else:
            self.steps_per_epoch = self.cpc.hs.max_steps['total']
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        assert index >= 0
        if self.single_step:
            assert index < len(ctf.BASES)
            base = ctf.BASES[index]
            df = self.cpc.hs.set_of_type(base, self.set_type)
        else:
            bix = 0
            base = ctf.BASES[bix]
            bstep = 0  # within a base
            step = 0  # steps from base to base
            myix = step + bstep
            while myix < index:
                base_steps = self.cpc.hs.max_steps[base]['max']
                if (step + base_steps) < index:
                    bix += 1
                    assert bix < len(ctf.BASES)
                    base = ctf.BASES[bix]
                    step += base_steps
                else:
                    bstep = index - step
                myix = step + bstep
            assert myix == index
            df = self.cpc.hs.trainset_step(base, bstep)
        samples = self.cpc.hs.features_from_targets(df, base, self.set_type, index)
        # print(f">>> getitem: {samples.descr}", flush=True)
        if samples is None:
            return None, None
        if self.cpc.scaler is not None:
            samples.data = self.cpc.scaler.transform(samples.data)
        targets = tf.keras.utils.to_categorical(samples.target, num_classes=len(ctf.TARGETS))
        return samples.data, targets


class EpochPerformance(tf.keras.callbacks.Callback):
    def __init__(self, cpc):
        self.cpc = cpc
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.cpc.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.cpc.performance_assessment(ctf.TRAIN, epoch)
        self.cpc.performance_assessment(ctf.VAL, epoch)


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


    def load(self):
        fname = str("{}/{}{}".format(self.model_path, self.load_classifier, ctf.PICKLE_EXT))
        try:
            with open(fname, 'rb') as df_f:
                tmp_dict = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()

                self.__dict__.update(tmp_dict)
                print(f"classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

        fname = str("{}/{}{}".format(self.model_path, self.load_classifier, '.h5'))
        try:
            with open(fname, 'rb') as df_f:
                df_f.close()

                self.classifier = tf.keras.models.load_model(fname)
                print(f"classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")


        if self.hs_name is not None:
            self.hs = ctf.HistorySets(self.hs_name)


    def save(self):
        """Saves the Cpc object without hs. The classifier is stored in a seperate file
        """
        classifier = self.classifier
        self.classifier = None
        hs = self.hs
        self.hs = None

        fname = str("{}/{}_{}{}".format(self.model_path, self.save_classifier,
                    self.epoch, '.h5'))
        # Save entire tensorflow keras model to a HDF5 file
        classifier.save(fname)

        #for ta in self.cpc:  # strip pandas due to pickle incompatibility between versions
        #    self.cpc[ta].prep_pickle()
        fname = str("{}/{}_{}{}".format(self.model_path, self.save_classifier,
                    self.epoch, ctf.PICKLE_EXT))
        df_f = open(fname, 'wb')
        pickle.dump(self.__dict__, df_f, pickle.HIGHEST_PROTOCOL)
        df_f.close()

        self.classifier = classifier
        self.hs = hs
        print(f"{ctf.timestr()} classifier saved in {fname}")

    def eval_classifier_with_data(self, val_data):
        """
        """
        val_cnt = len(val_data.data)
        print(f"{val_data.descr}: # samples validation: {val_cnt}")

        start_time = timeit.default_timer()
        # Now predict the action corresponding to samples on the second half:
        expected = val_data.target
        val_data.data = self.scaler.transform(val_data.data)
        predicted = self.classifier.predict(val_data.data)
#         predicted = self.predict_probabilities(val_data.data)
#         does not work because of ValueError: Classification metrics can't handle a mix of multiclass and continuous-multioutput targets
        tdiff = timeit.default_timer() - start_time
        print(f"{val_data.descr} SVM evaluation time: {tdiff:.0f}s")

        print(f"Classification report for {val_data.descr} classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(expected, predicted)))
        print(f"{val_data.descr} Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    def predict_probabilities(self, features):
        features = self.scaler.transform(features)
        pred = self.classifier.predict_proba(features)
        return pred

    def predict_probabilities_with_targetsubset(self, tfv, target_df):
        tfv_ta = tfv.vec
        subset_df = ctf.targets_to_features(tfv_ta, target_df)
        descr = f"{tfv.cur_pair()}"
        skldata = ctf.to_scikitlearn(subset_df, np_data=None, descr=descr)
        pred = self.predict_probabilities(skldata.data)
        return pred

    def adapt_with_targetsubset(self, tfv, targetsubset, setname):
        """
        """
        start_time = timeit.default_timer()
        subset_df = ctf.targets_to_features(tfv.vec, targetsubset)
        descr = f"{tfv.cur_pair()} {setname}"
        ctf.report_setsize(descr, subset_df)
        skldata = ctf.to_scikitlearn(subset_df, np_data=None, descr=descr)
        self.adapt_classifier_with_data(skldata)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"adapt_with_targetsubset time: {tdiff:.1f} min")

    def eval_with_targetsubset(self, tfv, targetsubset, setname):
        """
        """
        start_time = timeit.default_timer()
        subset_df = ctf.targets_to_features(tfv.vec, targetsubset)
        descr = f"{tfv.cur_pair()} {setname}"
        ctf.report_setsize(descr, subset_df)
        skldata = ctf.to_scikitlearn(subset_df, np_data=None, descr=descr)
        self.eval_classifier_with_data(skldata)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"eval_with_targetsubset time: {tdiff:.1f} min")


    def performance_with_features(self, tfv, buy_trshld, sell_trshld):
        """Ignores targets of tfv and just evaluates the tfv features in respect
        to their close price performance.

        It returns the classified targets_features.TARGETS[class]
        """
        if tfv.vec.empty:
            return ctf.TARGETS[ctf.HOLD]
        descr = f"{tfv.cur_pair()}"
        skldata = ctf.to_scikitlearn(tfv.vec, np_data=None, descr=descr)
        pred = self.predict_probabilities(skldata.data)
        ls = len(pred) - 1
        high_prob_cl = ctf.TARGETS[ctf.HOLD]
        if pred[ls, ctf.TARGETS[ctf.BUY]] > pred[ls, ctf.TARGETS[ctf.SELL]]:
            if pred[ls, ctf.TARGETS[ctf.BUY]] > pred[ls, ctf.TARGETS[ctf.HOLD]]:
                if pred[ls, ctf.TARGETS[ctf.BUY]] >= buy_trshld:
                    high_prob_cl = ctf.TARGETS[ctf.BUY]
        else:
            if pred[ls, ctf.TARGETS[ctf.SELL]] > pred[ls, ctf.TARGETS[ctf.HOLD]]:
                if pred[ls, ctf.TARGETS[ctf.BUY]] >= sell_trshld:
                    high_prob_cl = ctf.TARGETS[ctf.SELL]
        return high_prob_cl

    def performance_with_targetsubset(self, tfv, subset_df, pm, setname):
        """Take target subset of tfv and just evaluates the tfv features in respect
        to their close price performance
        """
        start_time = timeit.default_timer()
        tfv_ta = tfv.vec
        if tfv_ta.empty:
            print(f"{ctf.timestr()} performance_with_targetsubset ERROR: empty df")
            return
        tfv_ta_subset = ctf.targets_to_features(tfv_ta, subset_df)
        descr = f"{tfv.cur_pair()} {setname}"
        ctf.report_setsize(descr, tfv_ta_subset)
        skldata = ctf.to_scikitlearn(tfv_ta_subset, np_data=None, descr=descr)
        pred = self.predict_probabilities(skldata.data)
        print(f"assessing {descr}")
        pm.assess_prediction(pred, skldata.close, skldata.target, skldata.tics, skldata.descr)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"performance_with_targetsubset time: {tdiff:.1f} min")

    def adapt_with_history_sets(self):
        start_time = timeit.default_timer()
        self.trainpm = None
        self.valpm = None
        self.hs_name = ctf.sets_config_fname()
        hs = ctf.HistorySets(self.hs_name)
        for self.epoch in range(2):
            hs.label_steps()
            train_samples = 0
            for self.step in range(max(hs.max_step.values())):
                self.trainpm = PerfMatrix(self.epoch, ctf.TRAIN)
                self.valpm = PerfMatrix(self.epoch, ctf.VAL)
                start_time2 = timeit.default_timer()
                start_time3 = timeit.default_timer()
                for base in hs.bases:
                    traindf = hs.trainset_step(base, self.step)
                    train_samples = 0
                    if traindf.empty:
                        print(f"empty training subset for {base} in step {self.step}")
                    else:
                        # print("training subset: ", df.head(), df.tail())
                        tfv = hs.get_targets_features_of_base(base)
                        try:
                            cpc.adapt_with_targetsubset(tfv, traindf, ctf.TRAIN)
                        except ctf.NoSubsetWarning as msg:
                            print("adapt_with_history_sets adapt train error {}/{}/{}: {}".format(
                                    self.epoch, self.step, base, msg))
                tdiff = (timeit.default_timer() - start_time3) / (60*60)
                print(f"adaptation step {self.step} total time: {tdiff:.2f} hours")
                start_time3 = timeit.default_timer()
                train_samples2 = 0
                for base in hs.bases:
                    traindf = hs.set_of_type(base, ctf.TRAIN)
                    train_samples2 += len(traindf)
                    if  traindf.empty:
                        print(f"empty training subset for {base}")
                    else:
                        tfv = hs.get_targets_features_of_base(base)
                        # cpc.eval_with_targetsubset(tfv, traindf, ctf.TRAIN)
                        try:
                            cpc.performance_with_targetsubset(tfv, traindf, self.trainpm, ctf.TRAIN)
                        except ctf.NoSubsetWarning as msg:
                            print("adapt_with_history_sets perf train error {}/{}/{}: {}".format(
                                    self.epoch, self.step, base, msg))
                print("performance {} after step {}".format(ctf.TRAIN, self.step))
                self.trainpm.report_assessment()
                for base in hs.bases:
                    valdf = hs.set_of_type(base, ctf.VAL)
                    if valdf.empty:
                        print(f"empty validation subset for {base}")
                    else:
                        tfv = hs.get_targets_features_of_base(base)
                        try:
                            cpc.performance_with_targetsubset(tfv, valdf, self.valpm, ctf.VAL)
                        except ctf.NoSubsetWarning as msg:
                            print("adapt_with_history_sets perf val error {}/{}/{}: {}".format(
                                    self.epoch, self.step, base, msg))
                print("performance {} after step {}".format(ctf.VAL, self.step))
                self.valpm.report_assessment()
                tdiff = (timeit.default_timer() - start_time3) / (60*60)
                print(f"performance step {self.step} total time: {tdiff:.2f} hours")
                tdiff = (timeit.default_timer() - start_time2) / (60*60)
                print(f"total step {self.step} time: {tdiff:.2f} hours")
                self.save()
            assert train_samples == train_samples2, "train_samples mismatch: ".format(
                    train_samples, train_samples2)
            # evaluate wrong classifications and reinforce those in the next round
            hs.use_training_mistakes()
        testpm = PerfMatrix(self.epoch, ctf.TEST)
        for base in hs.bases:
            testdf = hs.set_of_type(base, ctf.TEST)
            tfv = hs.get_targets_features_of_base(base)
            try:
                cpc.performance_with_targetsubset(tfv, testdf, testpm, ctf.TEST)
            except ctf.NoSubsetWarning as msg:
                print(f"adapt_with_history_sets perf test error {base}: {msg}")
        print("performance {} ".format(ctf.TEST))
        testpm.report_assessment()

        tdiff = (timeit.default_timer() - start_time) / (60*60)
        print(f"total adaptation and performance evaluation time: {tdiff:.2f} hours")

    def adapt_classifier_with_data(self, train_data):
        """
        TODO: warm start
        """
        train_cnt = len(train_data.data)
        print(f"{train_data.descr}: # samples training: {train_cnt}")

        self.scaler = preprocessing.StandardScaler()
        train_data.data = self.scaler.fit_transform(train_data.data)
        # sample_sets['validation'].data = scaler.transform(sample_sets['validation'].data)

        start_time = timeit.default_timer()

        # Create a classifier: a support vector classifier
        # changed gamma from 0.001 to 1/#features = 0.02 for less generalization
        # changed gamma from 0.02 to 0.1 for less generalization
        self.classifier = svm.SVC(kernel='rbf', degree=3, gamma=0.001, probability=True,
                                  cache_size=400, class_weight='balanced', shrinking=True,
                                  decision_function_shape='ovo')
        # classifier = MLPClassifier(solver='adam',
        #                    hidden_layer_sizes=(20, 10, 3), random_state=1)

        # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV steps:
        # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # X, y = sample_sets['training'].data, sample_sets['training'].target
        # plot_learning_curve(classifier, title, X, y, (0.2, 1.01), cv=cv, n_jobs=1)

        self.classifier.fit(train_data.data, train_data.target)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{train_data.descr} SVM adaptation time: {tdiff:.0f} min")

    def performance_assessment(self, set_type, epoch):
        pm = PerfMatrix(epoch, set_type)
        for bix, base in enumerate(self.hs.bases):
            df = self.hs.set_of_type(base, set_type)
            samples = self.hs.features_from_targets(df, base, set_type, bix)
            if self.scaler is not None:
                samples.data = self.scaler.transform(samples.data)
            pred = self.classifier.predict_on_batch(samples.data)
            pm.assess_prediction(pred, samples.close, samples.target, samples.tics, samples.descr)
        self.pmlist.append(pm)
        pm.report_assessment()

    def adapt_keras_with_history_sets(self):
        start_time = timeit.default_timer()
        self.hs = ctf.HistorySets(ctf.sets_config_fname())

        batch_gen = BatchGenerator(self, ctf.TRAIN, single_step=True)
        scaler = preprocessing.StandardScaler(copy=False)
        for (samples, targets) in batch_gen:
            # print("scaler fit")
            scaler.partial_fit(samples)
        self.scaler = scaler

        inputs = krs.Input(shape=(samples.shape[1],))  # (110,))  # Returns a placeholder tensor
        x = krs.layers.Dense(80, kernel_initializer='he_uniform', activation='relu')(inputs)
        x = krs.layers.Dense(40, kernel_initializer='he_uniform', activation='relu')(x)
        predictions = krs.layers.Dense(3, activation='softmax')(x)
        self.classifier = krs.Model(inputs=inputs, outputs=predictions)

        self.classifier.compile(optimizer=krs.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        callbacks = [
          EpochPerformance(self),
          # Interrupt training if `val_loss` stops improving for over 2 epochs
          krs.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
          # Write TensorBoard logs to `./logs` directory
          krs.callbacks.TensorBoard(log_dir=LOG_PATH)
        ]

        self.hs.label_steps()
        self.classifier.fit_generator(
                BatchGenerator(self, ctf.TRAIN, single_step=False),
                steps_per_epoch=None, epochs=3, callbacks=callbacks, verbose=2,
                validation_data=BatchGenerator(self, ctf.VAL, single_step=True),
                validation_steps=len(self.hs.bases),
                class_weight=None, max_queue_size=10, workers=1,
                use_multiprocessing=False, shuffle=False, initial_epoch=0)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"MLP adaptation time: {tdiff:.0f} min")
        self.save()



def adapt_keras(cpc, talos=False):

    def iteration_generator(cpc, hs, epochs):
        'Generate one batch of data'
        for e in range(epochs):  # due to prefetch of max_queue_size
            for base in hs.bases:
                bix = list(hs.bases.keys()).index(base)
                for bstep in range(hs.max_steps[base]['max']):
                    df = hs.trainset_step(base, bstep)
                    samples = hs.features_from_targets(df, base, ctf.TRAIN, bix)
                    # print(f">>> getitem: {samples.descr}", flush=True)
                    if samples is None:
                        return None, None
                    if cpc.scaler is not None:
                        samples.data = cpc.scaler.transform(samples.data)
                    # print(f"iteration_gen {base}({bix}) {bstep}(of {hs.max_steps[base]['max']})")
                    targets = tf.keras.utils.to_categorical(samples.target,
                                                            num_classes=len(ctf.TARGETS))
                    yield samples.data, targets

    def base_generator(cpc, hs, set_type, epochs):
        'Generate one batch of data per base'
        for e in range(epochs):
            for base in hs.bases:
                bix = list(hs.bases.keys()).index(base)
                df = hs.set_of_type(base, set_type)
                samples = hs.features_from_targets(df, base, set_type, bix)
                # print(f">>> getitem: {samples.descr}", flush=True)
                if samples is None:
                    return None, None
                if cpc.scaler is not None:
                    samples.data = cpc.scaler.transform(samples.data)
                # print(f"base_gen {base}({bix}) {set_type}")
                targets = tf.keras.utils.to_categorical(samples.target,
                                                        num_classes=len(ctf.TARGETS))
                yield samples.data, targets

    def MLP1(x, y, x_val, y_val, params):
        krs.backend.clear_session()
        model = krs.models.Sequential()
        model.add(krs.layers.Dense(params['first_layer_neurons'],
                                   input_dim=samples.shape[1],
                                   kernel_initializer=params['kernel_initializer'],
                                   activation=params['activation']))
        model.add(krs.layers.Dropout(params['dropout']))
        model.add(krs.layers.Dense(params['second_layer_neurons'],
                                   kernel_initializer=params['kernel_initializer'],
                                   activation=params['activation']))
        model.add(krs.layers.Dense(params['last_layer_neurons'],
                                   activation=params['last_activation']))
        cpc.classifier = model
        assert model is not None

        cpc.classifier.compile(optimizer=params['optimizer'],
                               loss=params['losses'],
                               metrics=['accuracy'])

        callbacks = [
          EpochPerformance(cpc),
          # Interrupt training if `val_loss` stops improving for over 2 epochs
          # krs.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
          # WARNING:tensorflow:Early stopping conditioned on metric `val_loss`
          #     which is not available. Available metrics are: loss,acc
          # Write TensorBoard logs to `./logs` directory
          krs.callbacks.TensorBoard(log_dir=LOG_PATH)
        ]

        steps_per_epoch = cpc.hs.label_steps()
        epochs = params['epochs']
        gen_epochs = epochs * 2  # due to max_queue_size more step data is requested than needed

        print(model.summary())
        out = cpc.classifier.fit_generator(
                iteration_generator(cpc, cpc.hs, gen_epochs),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=callbacks,
                verbose=2,
                validation_data=base_generator(cpc, cpc.hs, ctf.VAL, gen_epochs),
                validation_steps=len(cpc.hs.bases),
                class_weight=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                shuffle=False,
                initial_epoch=0)
        assert out is not None

        return out,model

    start_time = timeit.default_timer()
    cpc.hs = ctf.HistorySets(ctf.sets_config_fname())

    scaler = preprocessing.StandardScaler(copy=False)
    for (samples, targets) in base_generator(cpc, cpc.hs, ctf.TRAIN, 1):
        # print("scaler fit")
        scaler.partial_fit(samples)
    cpc.scaler = scaler
    print(f"{ctf.timestr()} scaler adapted")
    print(f"Talos active:{talos}")

    dummy_x = np.empty((1, samples.shape[1]))
    dummy_y = np.empty((1, targets.shape[1]))
    if talos:
        params = {'first_layer_neurons': [80],
                  'second_layer_neurons': [40],
                  'last_layer_neurons': [3],
                  'epochs': [2],
                  'dropout': (0, 0.40, 3),
                  'kernel_initializer': ['he_uniform'],
                  'optimizer': ['adam'],
                  'losses': ['categorical_crossentropy'],
                  'activation': ['relu', 'elu'],
                  'last_activation': ['softmax']}

        ta.Scan(x=dummy_x,  # real data comes from generator
                       y=dummy_y,  # real data comes from generator
                       model=MLP1,
                       debug=True,
                       print_params=True,
                       params=params,
                       dataset_name='xrp_eos',  # 'xrp-eos-bnb-btc-eth-neo-ltc-trx',
                       grid_downsample=1,
                       experiment_no='talos_test_001')

        # ta.Deploy(scan, 'talos_lstm_x', metric='val_loss', asc=True)
    else:
        MLP1(dummy_x, dummy_y, dummy_x, dummy_y, None)

    tdiff = (timeit.default_timer() - start_time) / 60
    print(f"{ctf.timestr()} MLP adaptation time: {tdiff:.0f} min")
    cpc.save()


def adapt_keras_without_talos(cpc):

    def iteration_generator(cpc, hs, epochs):
        'Generate one batch of data'
        for e in range(epochs):  # due to prefetch of max_queue_size
            for base in hs.bases:
                bix = list(hs.bases.keys()).index(base)
                for bstep in range(hs.max_steps[base]['max']):
                    df = hs.trainset_step(base, bstep)
                    samples = hs.features_from_targets(df, base, ctf.TRAIN, bix)
                    # print(f">>> getitem: {samples.descr}", flush=True)
                    if samples is None:
                        return None, None
                    if cpc.scaler is not None:
                        samples.data = cpc.scaler.transform(samples.data)
                    # print(f"iteration_gen {base}({bix}) {bstep}(of {hs.max_steps[base]['max']})")
                    targets = tf.keras.utils.to_categorical(samples.target,
                                                            num_classes=len(ctf.TARGETS))
                    yield samples.data, targets

    def base_generator(cpc, hs, set_type, epochs):
        'Generate one batch of data per base'
        for e in range(epochs):
            for base in hs.bases:
                bix = list(hs.bases.keys()).index(base)
                df = hs.set_of_type(base, set_type)
                samples = hs.features_from_targets(df, base, set_type, bix)
                # print(f">>> getitem: {samples.descr}", flush=True)
                if samples is None:
                    return None, None
                if cpc.scaler is not None:
                    samples.data = cpc.scaler.transform(samples.data)
                # print(f"base_gen {base}({bix}) {set_type}")
                targets = tf.keras.utils.to_categorical(samples.target,
                                                        num_classes=len(ctf.TARGETS))
                yield samples.data, targets

    def MLP1(x, y, x_val, y_val, params):
        model = krs.models.Sequential()
        model.add(krs.layers.Dense(80,
                                   input_dim=samples.shape[1],
                                   kernel_initializer='he_uniform',
                                   activation='relu'))
        model.add(krs.layers.Dropout(0.2))
        model.add(krs.layers.Dense(40,
                                   kernel_initializer='he_uniform',
                                   activation='relu'))
        model.add(krs.layers.Dense(3,
                                   activation='softmax'))
        cpc.classifier = model
        assert model is not None

        cpc.classifier.compile(optimizer=krs.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

        callbacks = [
          EpochPerformance(cpc),
          # Interrupt training if `val_loss` stops improving for over 2 epochs
          krs.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
          # Write TensorBoard logs to `./logs` directory
          krs.callbacks.TensorBoard(log_dir=LOG_PATH)
        ]

        steps_per_epoch = cpc.hs.label_steps()
        epochs = 2
        gen_epochs = epochs * 2  # due to max_queue_size more step data is requested than needed
        out = cpc.classifier.fit_generator(
                iteration_generator(cpc, cpc.hs, gen_epochs),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                callbacks=callbacks,
                verbose=2,
                validation_data=base_generator(cpc, cpc.hs, ctf.VAL, gen_epochs),
                validation_steps=len(cpc.hs.bases),
                class_weight=None,
                max_queue_size=10, workers=1, use_multiprocessing=False,
                shuffle=False, initial_epoch=0)
        assert out is not None
        return out,model

    start_time = timeit.default_timer()
    cpc.hs = ctf.HistorySets(ctf.sets_config_fname())

    scaler = preprocessing.StandardScaler(copy=False)
    for (samples, targets) in base_generator(cpc, cpc.hs, ctf.TRAIN, 1):
        # print("scaler fit")
        scaler.partial_fit(samples)
    cpc.scaler = scaler
    print(f"{ctf.timestr()} scaler adapted")

    MLP1(samples, targets, samples, targets, None)
    tdiff = (timeit.default_timer() - start_time) / 60
    print(f"{ctf.timestr()} MLP adaptation time: {tdiff:.0f} min")
    cpc.save()



if __name__ == "__main__":
    if True:
        load_classifier = None  # "2019-04-18_Full-SVM-orig-buy-hold-sell_gamma_0.01_extended_features_smooth+xrp_usdt5+D10"
        save_classifier = str("MLP-110-80relu-40relu-3softmax")
        #     load_classifier = str("{}{}".format(BASE, target_key))
        unit_test = False
        cpc = Cpc(load_classifier, save_classifier)
        # cpc.adapt_keras_with_history_sets()
        adapt_keras(cpc, talos=True)

