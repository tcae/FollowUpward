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

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

import targets_features as t_f

print(__doc__)
start_time = timeit.default_timer()

BASE = 'xrp'
AGGREGATION = 5
MODEL_PATH = '/Users/tc/crypto/classifier'


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

    def buy_with_probability(self, buy_prob, close_price):
        if self.open_transaction:
            if self.highest < close_price:
                self.highest = close_price
        else:
            if buy_prob >= self.bpt:
                self.open_transaction = True
                self.open_buy = close_price * (1 + t_f.FEE / 1000)
                self.highest = close_price

    def hold_with_probability(self, sell_prob, close_price):
        return # no action

    def sell_with_probability(self, sell_prob, close_price):
        if self.open_transaction:
            if sell_prob >= self.spt:
                self.open_transaction = False
                gain = (close_price * (1 - t_f.FEE / 1000) - self.open_buy) /self.open_buy * 100
                self.performance += gain # in %
                self.transactions += 1
                self.highest = 0


class PerfMatrix:
    """Evaluates the performance across a range of buy/sell thresholds
    """

    def __init__(self):
        self.perf = np.zeros((5,5), dtype=EvalPerf)
        for bp in range(5):
            for sp in range(5):
                self.perf[bp,sp] = EvalPerf(float((bp+5)/10), float((sp+5)/10))

    def print_result_distribution(self):
        for bp in range(len(self.perf)):
            for sp in range(len(self.perf[0])):
                print("bpt: {:<5.2} spt: {:<5.2} perf: {:<8.2F}% #trans.: {:<4}".format(\
                      float((bp+5)/10), float((sp+5)/10), \
                      float(self.perf[bp, sp].performance), self.perf[bp, sp].transactions))

    def plot_result_distribution(self):
        dtype = [('bpt', float), ('spt', float), ('performance', float), ('transactions', int)]
        values = list()
        for bp in range(len(self.perf)):
            for sp in range(len(self.perf[0])):
                values.append((float((bp+5)/10), float((sp+5)/10), \
                               self.perf[bp, sp].performance, self.perf[bp, sp].transactions))
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
        hbp = hsp = 0
        for bp in range(len(self.perf)):
            for sp in range(len(self.perf[0])):
                if best < self.perf[bp, sp].performance:
                    best = self.perf[bp, sp].performance
                    hbp = bp
                    hsp = sp
        bpt = float((hbp+5)/10)
        spt = float((hsp+5)/10)
        t = self.perf[hbp, hsp].transactions
        return "best performance {:<8.2F}% with buy threshold {:.2f} / sell threshold {:.2f} at {} transactions".format(best, bpt, spt, t)

    def add_buy_signal(self, prob, close_price):
        assert (prob >=0) and (prob <= 1), \
                print(f"unexpected probability {prob}")
        for bp in range(5):
            for sp in range(5):
                self.perf[bp, sp].buy_with_probability(prob, close_price)

    def add_hold_signal(self, prob, close_price):
        assert (prob >=0) and (prob <= 1), \
                print(f"unexpected probability {prob}")
        for bp in range(5):
            for sp in range(5):
                self.perf[bp, sp].hold_with_probability(prob, close_price)

    def add_sell_signal(self, prob, close_price):
        assert (prob >=0) and (prob <= 1), \
                print(f"unexpected probability {prob}")
        for bp in range(5):
            for sp in range(5):
                self.perf[bp, sp].sell_with_probability(prob, close_price)

    def assess_prediction(self, pred, skl_close, skl_tics):
        """Assess the highest probability of a class prediction
        to find the optimum buy/sell threshold.

        to do:
        ======

        handle time gaps: in case of a time gap the last value of the time slice is taken
        to close any open transaction

        """
        pred_cnt = len(pred)
        print(f"performance assessment starts: {skl_tics[0]}")
        lowcnt = 0
        for sample in range(pred_cnt):
            if (sample + 1) >= pred_cnt:
                self.add_sell_signal(1, skl_close[sample])
                print(f"performance assessment ends: {skl_tics[sample]}")
            elif (skl_tics[sample+1] - skl_tics[sample]) > np.timedelta64(1, 'm'):
                self.add_sell_signal(1, skl_close[sample])
                print(f"assessment gap between {skl_tics[sample]} and {skl_tics[sample+1]}")
            else:
                high_prob_cl = 0
                for cl in range(len(pred[0])):
                    if pred[sample, high_prob_cl] < pred[sample, cl]:
                        high_prob_cl = cl
                if pred[sample, high_prob_cl] < 0.5:
                    self.add_hold_signal(pred[sample, high_prob_cl], skl_close[sample])
                    lowcnt += 1
                elif high_prob_cl == t_f.TARGETS[t_f.BUY]:
                    self.add_buy_signal(pred[sample, high_prob_cl], skl_close[sample])
                elif high_prob_cl == t_f.TARGETS[t_f.SELL]:
                    self.add_sell_signal(pred[sample, high_prob_cl], skl_close[sample])
                elif high_prob_cl == t_f.TARGETS[t_f.HOLD]:
                    self.add_hold_signal(pred[sample, high_prob_cl], skl_close[sample])
                else:
                    print("unexpected missing classification result")
        print(self)
        if lowcnt > 0:
            print(f"{lowcnt} predictions are 'hold' due to probability lower/equal 0.5")
        self.print_result_distribution()

class Cpc:
    """Provides methods to adapt single currency performance classifiers
    """
    def __init__(self, target_key, load_classifier, save_classifier):
        self.load_classifier = load_classifier
        self.save_classifier = save_classifier
        self.model_path = MODEL_PATH
        self.target_key = target_key
        self.scaler = None
        self.classifier = None


    def load(self):
        fname = str("{}/{}{}".format(self.model_path, self.load_classifier, t_f.PICKLE_EXT))
        try:
            with open(fname, 'rb') as df_f:
                tmp_dict = pickle.load(df_f)  # requires import * to resolve Cpc attribute
                df_f.close()
                self.__dict__.update(tmp_dict)
                print(f"classifier loaded from {fname}")
        except IOError:
            print(f"IO-error when loading classifier from {fname}")

    def save(self):
        #for ta in self.cpc:  # strip pandas due to pickle incompatibility between versions
        #    self.cpc[ta].prep_pickle()
        fname = str("{}/{}{}".format(self.model_path, self.save_classifier, t_f.PICKLE_EXT))
        df_f = open(fname, 'wb')
        pickle.dump(self.__dict__, df_f, pickle.HIGHEST_PROTOCOL)
        df_f.close()
        print(f"classifier saved in {fname}")

    def adapt_classifier_with_data(self, train_data):
        """
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
                                  cache_size=400, class_weight='balanced', shrinking=True)
        # classifier = MLPClassifier(solver='adam',
        #                    hidden_layer_sizes=(20, 10, 3), random_state=1)

        # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # X, y = sample_sets['training'].data, sample_sets['training'].target
        # plot_learning_curve(classifier, title, X, y, (0.2, 1.01), cv=cv, n_jobs=1)

        self.classifier.fit(train_data.data, train_data.target)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"{train_data.descr} SVM adaptation time: {tdiff:.0f} min")

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
        tfv_ta = tfv.vec(target_key)
        subset_df = t_f.targets_to_features(tfv_ta, target_df)
        descr = f"{tfv.cur_pair} {target_key}"
        skldata = tfv.to_scikitlearn(subset_df, np_data=None, descr=descr)
        pred = self.predict_probabilities(skldata.data)
        return pred

    def adapt_with_targetsubset(self, tfv, targetsubset, setname):
        """
        """
        start_time = timeit.default_timer()
        tfv_ta = tfv.vec(self.target_key)
        subset_df = t_f.targets_to_features(tfv_ta, targetsubset)
        descr = f"{tfv.cur_pair} {self.target_key} {setname}"
        t_f.report_setsize(descr, subset_df)
        skldata = tfv.to_scikitlearn(subset_df, np_data=None, descr=descr)
        self.adapt_classifier_with_data(skldata)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"adapt_with_targetsubset time: {tdiff:.1f} min")
        self.save()

    def eval_with_targetsubset(self, tfv, targetsubset, setname):
        """
        """
        start_time = timeit.default_timer()
        tfv_ta = tfv.vec(self.target_key)
        subset_df = t_f.targets_to_features(tfv_ta, targetsubset)
        descr = f"{tfv.cur_pair} {self.target_key} {setname}"
        t_f.report_setsize(descr, subset_df)
        skldata = tfv.to_scikitlearn(subset_df, np_data=None, descr=descr)
        self.eval_classifier_with_data(skldata)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"eval_with_targetsubset time: {tdiff:.1f} min")


    def performance_with_features(self, tfv, buy_trshld, sell_trshld):
        """Ignores targets of tfv and just evaluates the tfv features in respect
        to their close price performance.

        It returns the classified targets_features.TARGETS[class]
        """
        tfv_ta = tfv.vec(self.target_key)
        if tfv_ta.empty:
            return t_f.TARGETS[t_f.HOLD]
        descr = f"{tfv.cur_pair} {self.target_key}"
        skldata = tfv.to_scikitlearn(tfv_ta, np_data=None, descr=descr)
        pred = self.predict_probabilities(skldata.data)
        ls = len(pred) - 1
        high_prob_cl = t_f.TARGETS[t_f.HOLD]
        if pred[ls, t_f.TARGETS[t_f.BUY]] > pred[ls, t_f.TARGETS[t_f.SELL]]:
            if pred[ls, t_f.TARGETS[t_f.BUY]] > pred[ls, t_f.TARGETS[t_f.HOLD]]:
                if pred[ls, t_f.TARGETS[t_f.BUY]] >= buy_trshld:
                    high_prob_cl = t_f.TARGETS[t_f.BUY]
        else:
            if pred[ls, t_f.TARGETS[t_f.SELL]] > pred[ls, t_f.TARGETS[t_f.HOLD]]:
                if pred[ls, t_f.TARGETS[t_f.BUY]] >= sell_trshld:
                    high_prob_cl = t_f.TARGETS[t_f.SELL]
        return high_prob_cl

    def performance_with_targetsubset(self, tfv, subset_df, setname):
        """Take target subset of tfv and just evaluates the tfv features in respect
        to their close price performance
        """
        start_time = timeit.default_timer()
        pm = PerfMatrix()
        tfv_ta = tfv.vec(self.target_key)
        if tfv_ta.empty:
            print(f"{t_f.timestr()} performance_with_targetsubset ERROR: empty df")
            return
        start_time2 = timeit.default_timer()
        tfv_ta_subset = t_f.targets_to_features(tfv_ta, subset_df)
        descr = f"{tfv.cur_pair} {self.target_key} {setname}"
        t_f.report_setsize(descr, tfv_ta_subset)
        skldata = tfv.to_scikitlearn(tfv_ta_subset, np_data=None, descr=descr)
        pred = self.predict_probabilities(skldata.data)
        print(f"assessing {descr}")
        pm.assess_prediction(pred, skldata.close, skldata.tics)

        tdiff = (timeit.default_timer() - start_time2) / 60
        print(f"single classifier performance_with_targetsubset time: {tdiff:.1f} min")
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"performance_with_targetsubset time: {tdiff:.1f} min")

if __name__ == "__main__":
    start_time = timeit.default_timer()
    target_key = 5
    time_aggs = {1: 10, 5: 10, 15: 10, 60: 10, 4*60: 10, 24*60: 10}
#     load_classifier = str("{}{}".format(BASE, target_key))
    load_classifier = None  # "2019-04-18_Full-SVM-orig-buy-hold-sell_gamma_0.01_extended_features_smooth+xrp_usdt5+D10"
    save_classifier = str("{}{}_SVM_0.01gamma+D10_newCpcSet".format(BASE, target_key))
    unit_test = False
    if True:  # new wow with history sets
        hs = t_f.HistorySets(target_key, time_aggs)
        trainsets = t_f.TrainingSet(hs)
        cpc = Cpc(target_key, load_classifier, save_classifier)
        tfv = t_f.load_classifier_features(time_aggs, target_key, BASE)
        for df in trainsets:
            if df.empty:
                print(f"empty training subset")
            else:
                print("training subset: ", df.head(), df.tail())
                tf = t_f.load_classifier_features(time_aggs, target_key, trainsets.base)
                tfv = tf.tf_vectors
                cpc.adapt_with_targetsubset(tfv, df, t_f.TRAIN)
                # cpc.eval_with_targetsubset(tfv, df, t_f.TRAIN)
                print(f"performance with hold or sell as sell {tfv.cur_pair} {t_f.TRAIN}")
                cpc.performance_with_targetsubset(tfv, df, t_f.TRAIN)
    else:  # old wow
        cpc = Cpc(target_key, load_classifier, save_classifier)
        tfv = t_f.load_classifier_features(time_aggs, target_key, BASE)
        tfv = tfv.tf_vectors
        cfname = "/Users/tc/crypto/classifier/sample_set_split.config"
        seq = tfv.timeslice_targets_as_configured(tfv.any_key(), cfname)
        # cpc.adapt_with_targetsubset(tfv, seq[t_f.TRAIN], t_f.TRAIN)
        # cpc.eval_with_targetsubset(tfv, seq[t_f.TRAIN], t_f.TRAIN)
        print(f"performance with hold or sell as sell {tfv.cur_pair} {t_f.TRAIN}")
        cpc.performance_with_targetsubset(tfv, seq[t_f.TRAIN], t_f.TRAIN)
        print(f"performance with hold as sell at highest buy {tfv.cur_pair} {t_f.VAL}")
        if False:
            cpc.performance_with_targetsubset(tfv, seq[t_f.VAL], t_f.VAL)
            print(f"performance {tfv.cur_pair} {t_f.TEST} using a {BASE} SVM 0.01gamma classifier")
            cpc.performance_with_targetsubset(tfv, seq[t_f.TEST], t_f.TEST)

            tfv = t_f.load_classifier_features(time_aggs, target_key, 'bnb')
            tfv = tfv.tf_vectors
            seq = tfv.timeslice_targets_as_configured(tfv.any_key(), cfname)
            print(f"performance {tfv.cur_pair} {t_f.TEST} using a {BASE} SVM 0.01gamma classifier")
            cpc.performance_with_targetsubset(tfv, seq[t_f.TEST], t_f.TEST)

    tdiff = (timeit.default_timer() - start_time) / (60*60)
    print(f"total time: {tdiff:.2f} hours")
