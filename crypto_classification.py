"""
================================
Recognizing crypto samples
================================

An example showing how the scikit-learn can be used
to classify crypto sell/buy actions.


"""
import numpy as np
import timeit
import math
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

PAIR = 'xrp_usdt'
AGGREGATION = 5


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

def eval_probs(classifier, val_data):
    """ cross_val is a 3 dimenional array with the following dimensions:
        ================================================================

        classes: as defined in t_f.TARGETS = {HOLD: 0, BUY: 1, SELL: 2, NA: 11}

        histogram with 10 probability elements to accumulated the evaluation results

        result dimension:
            0 == nbr of wrong classifications for that class and probability;

            1 == nbr of correct classifications for that class and probability;

            2 == either zero of no result available of that class probability or
            calculated probability for the class

            3 == number observations for that probability (is only valid for class == 0)
        """
    start_time = timeit.default_timer()
    cross_val = np.zeros((3, 10, 4))
    # Now predict the action corresponding to samples on the second half:
    expected = val_data.target
    predicted = classifier.predict_proba(val_data.data)
    pred_cnt = len(predicted)
    class_samples = dict()
    for cl in range(len(predicted[0])):
        class_samples[cl] = 0
    for sample in range(pred_cnt):
        t = expected[sample]
        p = 0
        for cl in range(len(predicted[0])):
            if predicted[sample, p] < predicted[sample, cl]:
                p = cl # make p the class index with the highest prediction
        for cl in range(len(predicted[0])):
            hist_ix = int(math.floor(predicted[sample, cl]*10))
            if (p == cl) or (t == cl):
                correct_ix = int(t == cl)
                cross_val[cl, hist_ix, correct_ix] += 1
                cross_val[cl, hist_ix, 2] += 1 # all observations
                if p == cl:
                    cross_val[cl, hist_ix, 3] += 1 # all highest probability observations
                class_samples[cl] += 1
    for hist_ix in range(len(cross_val[0])):
        all_observ = 0
        for cl in range(len(cross_val)):
            if cross_val[cl, hist_ix, 2] != 0: # any observation?
                cross_val[cl, hist_ix, 2] = cross_val[cl, hist_ix, 1] / \
                                            cross_val[cl, hist_ix, 2]
            all_observ += cross_val[cl, hist_ix, 3] # for that histogram entry
        cross_val[0, hist_ix, 3] = all_observ / pred_cnt # only for (arbitrary) class 0

    plt.figure()
    plt.title("prob cross validation of SVM "+val_data.descr)
    plt.ylim((0., 1.0))
    plt.xlabel("prob estimate")
    plt.ylabel("prob validate")
#        train_sizes, train_scores, test_scores = learning_curve(
#            estimator, X, y, cv=cv, n_jobs=n_jobs, pre_dispatch="2*n_jobs", train_sizes=train_sizes)
#        train_scores_mean = np.mean(train_scores, axis=1)
#        train_scores_std = np.std(train_scores, axis=1)
#        test_scores_mean = np.mean(test_scores, axis=1)
#        test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    xaxis = np.arange(0, 1, 0.1)
    plt.plot(xaxis, cross_val[t_f.TARGETS[t_f.BUY], :, 2], 'o-', color="g",
             label="BUY prob")
    plt.plot(xaxis, cross_val[t_f.TARGETS[t_f.SELL], :, 2], 'o-', color="r",
             label="SELL prob")
    plt.plot(xaxis, cross_val[t_f.TARGETS[t_f.HOLD], :, 2], 'o-', color="b",
             label="HOLD prob")
    plt.plot(xaxis, cross_val[t_f.TARGETS[t_f.HOLD], :, 3], 'o-', color="c",
             label="% samples") # take an arbitrary class as it is for all the same

    plt.legend(loc="best")
    plt.show()

    # proba = classifier.predict_proba(sample_sets['validation'].data)
    print(f"SVM evaluation: {timeit.default_timer() - start_time}")

def targets_to_features(tfv_ta_df, target_df):
    """Extracts a sample subset with targets and features of a specific time aggregation
    based on given targets. target_df and tfv_ta_df both have to share the same index basis.
    The index of target_df shall be a subset of tfv_ta_df.
    """
    return tfv_ta_df[tfv_ta_df.index.isin(target_df.index)]


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
                if high_prob_cl == t_f.TARGETS[t_f.BUY]:
                    self.add_buy_signal(pred[sample, high_prob_cl], skl_close[sample])
                elif high_prob_cl == t_f.TARGETS[t_f.SELL]:
                    self.add_sell_signal(pred[sample, high_prob_cl], skl_close[sample])
                elif high_prob_cl == t_f.TARGETS[t_f.HOLD]:
                    self.add_hold_signal(pred[sample, high_prob_cl], skl_close[sample])
                else:
                    print("unexpected missing classification result")
        print(self)
        self.print_result_distribution()


class Cpc:
    """Provides methods to adapt single currency performance classifiers
    """

    def __init__(self, aggregation):
        self.key = aggregation

    def adapt_classifier_with_data(self, train_data):
        """
        """
        train_cnt = len(train_data.data)
        print(f"{train_data.descr}: # samples training: {train_cnt}")

        # scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0), copy=True)
        self.scaler = None
        # sample_sets['training'].data = scaler.fit_transform(sample_sets['training'].data)
        # sample_sets['validation'].data = scaler.transform(sample_sets['validation'].data)

        start_time = timeit.default_timer()
        # Create a classifier: a support vector classifier
        # changed gamma from 0.001 to 1/#features = 0.02 for less generalization
        self.classifier = svm.SVC(kernel='rbf', degree=3, gamma=0.02, probability=True,
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
        # start_time = timeit.default_timer()
        # eval_probs(self.classifier, val_data)
        # print(f"{self.descr} probability evaluation: {timeit.default_timer() - start_time}")

        start_time = timeit.default_timer()
        # Now predict the action corresponding to samples on the second half:
        expected = val_data.target
        predicted = self.classifier.predict(val_data.data)
        # proba = classifier.predict_proba(sample_sets['validation'].data)
        tdiff = timeit.default_timer() - start_time
        print(f"{val_data.descr} SVM evaluation time: {tdiff:.0f}s")

        print(f"Classification report for {val_data.descr} classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(expected, predicted)))
        print(f"{val_data.descr} Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


class CpcSet:
    """Provides methods to adapt an orchestrated set of currency performance classifiers
    """
    def __init__(self, currency_pair, data_path, model_path):
        self.currency_pair = currency_pair
        self.data_path = data_path
        self.model_path = model_path
        fname = self.model_path + '/' + self.currency_pair + t_f.PICKLE_EXT
        try:
            with open(fname, 'rb') as df_f:
                tmp_dict = pickle.load(df_f)
                df_f.close()
                self.__dict__.update(tmp_dict)
                print(f"classifier loaded from {fname}")
        except IOError:
            pass

    def save(self, as_ext=t_f.PICKLE_EXT):
        #for ta in self.cpc:  # strip pandas due to pickle incompatibility between versions
        #    self.cpc[ta].prep_pickle()
        fname = self.model_path + '/' + self.currency_pair + as_ext
        df_f = open(fname, 'wb')
        pickle.dump(self.__dict__, df_f, pickle.HIGHEST_PROTOCOL)
        df_f.close()
        print(f"classifier saved in {fname}")

    def combined_probs(self, tfv, target_df):
        predictions = None
        for ta in self.cpc:
            if isinstance(ta, int):
                tfv_ta = tfv.vec(ta)
                subset_df = targets_to_features(tfv_ta, target_df)
                descr = f"{self.currency_pair} {ta}"
                skldata = tfv.to_scikitlearn(subset_df, np_data=None, descr=descr)
                pred = self.cpc[ta].classifier.predict_proba(skldata.data)
                if predictions is None:
                    predictions = pred
                else:
                    predictions = np.concatenate((predictions, pred), axis=1)
        return predictions

    def adapt_ensemble_with_targetsubset(self, tfv, targetsubset, setname, balance):
        """
        """
        start_time = timeit.default_timer()
        self.cpc = dict(t_f.TIME_AGGS) # use a ta dict to manage the cpc set
        for ta in self.cpc:
            self.cpc[ta] = Cpc(ta)
            if isinstance(ta, int):
                tfv_ta = tfv.vec(ta)
                if balance:
                    subset_df = tfv.reduce_sequences_balance_targets(ta, targetsubset, balance)
                else:
                    subset_df = targetsubset
                subset_df = targets_to_features(tfv_ta, subset_df)
                descr = f"{self.currency_pair} {ta} {setname}"
                tfv.report_setsize(descr, subset_df)
                skldata = tfv.to_scikitlearn(subset_df, np_data=None, descr=descr)
                self.cpc[ta].adapt_classifier_with_data(skldata)

        if t_f.CPC in self.cpc:
            combo_cpc = self.cpc[t_f.CPC]
            if balance:
                subset_df = tfv.reduce_sequences_balance_targets(t_f.CPC, targetsubset, balance)
            else:
                subset_df = targetsubset
            predictions = self.combined_probs(tfv, subset_df)
            tfv_ta = tfv.vec(t_f.CPC)
            subset_df = targets_to_features(tfv_ta, subset_df)
            descr = f"{self.currency_pair} {t_f.CPC} {setname}"
            tfv.report_setsize(descr, subset_df)
            skldata = tfv.to_scikitlearn(subset_df, np_data=predictions, descr=descr)
            combo_cpc.adapt_classifier_with_data(skldata)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"adapt_ensemble time: {tdiff:.1f} min")
        self.save()

    def adapt_ensemble(self, train_ratio, val_ratio, balance, days):
        """
        """
        # The crypto dataset
        fname = self.data_path + '/' + self.currency_pair + t_f.MSG_EXT
        tfv = t_f.TfVectors(filename=fname)
        for ta in t_f.TIME_AGGS:
            assert ta in tfv.aggregations, print(f"missing {ta} feature set")
        self.cpc = dict(t_f.TIME_AGGS) # use a ta dict to manage the cpc set
        seq = tfv.timeslice_targets(t_f.ALL_SAMPLES, train_ratio, val_ratio, days)
        self.adapt_ensemble_with_targetsubset(tfv, seq[t_f.TRAIN], t_f.TRAIN, balance)

    def eval_ensemble_with_targetsubset(self, tfv, targetsubset, setname, balance):
        """
        """
        start_time = timeit.default_timer()
        for ta in self.cpc:
            if isinstance(ta, int):
                tfv_ta = tfv.vec(ta)
                if balance:
                    subset_df = tfv.reduce_sequences_balance_targets(ta, targetsubset, balance)
                else:
                    subset_df = targetsubset
                subset_df = targets_to_features(tfv_ta, subset_df)
                descr = f"{self.currency_pair} {ta} {setname}"
                tfv.report_setsize(descr, subset_df)
                skldata = tfv.to_scikitlearn(subset_df, np_data=None, descr=descr)
                self.cpc[ta].eval_classifier_with_data(skldata)

        if t_f.CPC in self.cpc:
            combo_cpc = self.cpc[t_f.CPC]
            tfv_ta = tfv.vec(t_f.CPC)
            if balance:
                subset_df = tfv.reduce_sequences_balance_targets(t_f.CPC, targetsubset, balance)
            else:
                subset_df = targetsubset
            predictions = self.combined_probs(tfv, subset_df)
            subset_df = targets_to_features(tfv_ta, subset_df)
            descr = f"{self.currency_pair} {t_f.CPC} {setname}"
            tfv.report_setsize(descr, subset_df)
            skldata = tfv.to_scikitlearn(subset_df, np_data=predictions, descr=descr)
            combo_cpc.eval_classifier_with_data(skldata)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"eval_ensemble time: {tdiff:.1f} min")

    def eval_ensemble(self, train_ratio, val_ratio, balance, days):
        """
        """
        # The crypto dataset
        fname = self.data_path + '/' + self.currency_pair + t_f.MSG_EXT
        tfv = t_f.TfVectors(filename=fname)
        seq = tfv.timeslice_targets(t_f.ALL_SAMPLES, train_ratio, val_ratio, days)
        self.eval_ensemble_with_targetsubset(tfv, seq[t_f.VAL], t_f.VAL, balance)

    def ensemble_performance_with_features(self, tfv):
        """Ignores targets of tfv and just evaluates the tfv features in respect
        to their close price performance
        """
        start_time = timeit.default_timer()
        pm = dict(self.cpc) # use a ta dict to manage the PerfMatrix of each cpc
        predictions = None
        for ta in self.cpc:
            pm[ta] = PerfMatrix()
            if isinstance(ta, int):
                tfv_ta = tfv.vec(ta)
                descr = f"{self.currency_pair} {ta}"
                skldata = tfv.to_scikitlearn(tfv_ta, np_data=None, descr=descr)
                pred = self.cpc[ta].classifier.predict_proba(skldata.data)
                pm[ta].assess_prediction(pred, skldata.close, skldata.tics)
                if predictions is None:
                    predictions = pred
                else:
                    predictions = np.concatenate((predictions, pred), axis=1)

        if t_f.CPC in self.cpc:
            combo_cpc = self.cpc[t_f.CPC]
            tfv_ta = tfv.vec(t_f.CPC)
            descr = f"{self.currency_pair} {t_f.CPC}"
            skldata = tfv.to_scikitlearn(tfv_ta, np_data=predictions, descr=descr)
            pred = combo_cpc.classifier.predict_proba(skldata.data)
            pm[t_f.CPC].assess_prediction(pred, skldata.close, skldata.tics)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"ensemble_performance_with_features time: {tdiff:.1f} min")

    def ensemble_performance_with_targetsubset(self, tfv, subset_df, setname):
        """Take target subset of tfv and just evaluates the tfv features in respect
        to their close price performance
        """
        start_time = timeit.default_timer()
        pm = dict(self.cpc) # use a ta dict to manage the PerfMatrix of each cpc
        predictions = None
        for ta in self.cpc:
            pm[ta] = PerfMatrix()
            if isinstance(ta, int):
                tfv_ta = tfv.vec(ta)
                start_time2 = timeit.default_timer()
                tfv_ta_subset = targets_to_features(tfv_ta, subset_df)
                descr = f"{self.currency_pair} {ta} {setname}"
                tfv.report_setsize(descr, tfv_ta_subset)
                skldata = tfv.to_scikitlearn(tfv_ta_subset, np_data=None, descr=descr)
                pred = self.cpc[ta].classifier.predict_proba(skldata.data)
                print(f"assessing {descr}")
                pm[ta].assess_prediction(pred, skldata.close, skldata.tics)
                tdiff = (timeit.default_timer() - start_time2) / 60
                print(f"single classifier performance_with_targetsubset time: {tdiff:.1f} min")
                if predictions is None:
                    predictions = pred
                else:
                    predictions = np.concatenate((predictions, pred), axis=1)

        if t_f.CPC in self.cpc:
            combo_cpc = self.cpc[t_f.CPC]
            tfv_ta = tfv.vec(t_f.CPC)
            start_time2 = timeit.default_timer()
            tfv_ta_subset = targets_to_features(tfv_ta, subset_df)
            descr = f"{self.currency_pair} {t_f.CPC} {setname}"
            tfv.report_setsize(descr, tfv_ta_subset)
            skldata = tfv.to_scikitlearn(tfv_ta_subset, np_data=predictions, descr=descr)
            pred = combo_cpc.classifier.predict_proba(skldata.data)
            print(f"assessing {descr}")
            pm[t_f.CPC].assess_prediction(pred, skldata.close, skldata.tics)
            tdiff = (timeit.default_timer() - start_time2) / 60
            print(f"single classifier performance_with_targetsubset time: {tdiff:.1f} min")
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"ensemble_performance_with_targetsubset time: {tdiff:.1f} min")

    def ensemble_performance(self, train_ratio, val_ratio, days):
        """
        """
        start_time = timeit.default_timer()
        # The crypto dataset
        fname = self.data_path + '/' + self.currency_pair + t_f.MSG_EXT
        tfv = t_f.TfVectors(filename=fname)
        seq = tfv.timeslice_targets(t_f.ALL_SAMPLES, train_ratio, val_ratio, days)
        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"ensemble_performance time: loaded data in {tdiff:.1f} min")
        print(f"performance {tfv.cur_pair} {t_f.TRAIN}")
        self.ensemble_performance_with_targetsubset(tfv, seq[t_f.TRAIN])
        print(f"performance {tfv.cur_pair} {t_f.VAL}")
        self.ensemble_performance_with_targetsubset(tfv, seq[t_f.VAL])
        print(f"performance {tfv.cur_pair} {t_f.TEST}")
        self.ensemble_performance_with_targetsubset(tfv, seq[t_f.TEST])


    def eval_combo_with_features(self, tfv):
        """not yet ready
        """
        if t_f.CPC in self.cpc:
            start_time = timeit.default_timer()
            combo_cpc = self.cpc[t_f.CPC]
            tfv_ta = tfv.vec(t_f.CPC)
            predictions = self.combined_probs(tfv, tfv_ta)
            descr = f"{self.currency_pair} {t_f.CPC}"
            skldata = tfv.to_scikitlearn(tfv_ta, np_data=predictions, descr=descr)
            combo_cpc.eval_classifier_with_data(skldata)
            tdiff = (timeit.default_timer() - start_time) / 60
            print(f"eval_combo_with_features time: {tdiff:.1f} min")
        else:
            print("missing combo classifier - cannot eval_combo_with_features")

def load_classifier_features(cur_pair):
    df = t_f.load_asset_dataframe(cur_pair)
    tf = t_f.TargetsFeatures(cur_pair=cur_pair)
    tf.calc_features_and_targets(df)
    tf.calc_performances()
    test = tf.performance
    for p in test:
        print(f"performance potential for aggregation {p}: {test[p]:%}")
    return tf.tf_vectors

start_time = timeit.default_timer()
unit_test = False
if not unit_test:
    t_f.TIME_AGGS = {5: 10}  # {1: 10, 5: 10, 15: 10, 60: 10}
    cpcs = CpcSet(PAIR, t_f.DATA_PATH, '/Users/tc/tf_models/crypto')
    # The crypto dataset
    # fname = cpcs.data_path + '/' + cpcs.currency_pair + t_f.MSG_EXT
    # fname = cpcs.data_path + '/' + 'bnb_usdt' + t_f.MSG_EXT
    # tfv = t_f.TfVectors(filename=fname)
    tfv = load_classifier_features('bnb_usdt')
    cfname = "/Users/tc/tf_models/crypto/sample_set_split.config"
    seq = tfv.timeslice_targets_as_configured(t_f.ALL_SAMPLES, cfname)
    # seq = tfv.timeslice_targets(t_f.ALL_SAMPLES, train_ratio=0.6, val_ratio=0.4, days=30)
    # cpcs.adapt_ensemble_with_targetsubset(tfv, seq[t_f.TRAIN], t_f.TRAIN, balance=False)
    # cpcs.eval_ensemble_with_targetsubset(tfv, seq[t_f.VAL], t_f.VAL, balance=False)
    # print(f"performance with hold or sell as sell {tfv.cur_pair} {t_f.TRAIN}")
    # cpcs.ensemble_performance_with_targetsubset(tfv, seq[t_f.TRAIN], t_f.TRAIN)
    # print(f"performance with hold as sell at highest buy {tfv.cur_pair} {t_f.VAL}")
    # cpcs.ensemble_performance_with_targetsubset(tfv, seq[t_f.VAL], t_f.VAL)
    print(f"performance {tfv.cur_pair} {t_f.TEST} using a {PAIR} SVM 0.02gamma classifier")
    cpcs.ensemble_performance_with_targetsubset(tfv, seq[t_f.TEST], t_f.TEST)
else:
    t_f.TIME_AGGS = {1: 10, 5: 10}
    cpcs = CpcSet(PAIR, t_f.DATA_PATH, '/Users/tc/tf_models/crypto')
    # The crypto dataset
    # fname = cpcs.data_path + '/' + cpcs.currency_pair + t_f.MSG_EXT
    # tfv = t_f.TfVectors(filename=fname)
    tfv = load_classifier_features(cpcs.currency_pair)
    seq = tfv.timeslice_targets(t_f.ALL_SAMPLES, train_ratio=0.4, val_ratio=0.4, days=30)
    cpcs.adapt_ensemble_with_targetsubset(tfv, seq[t_f.TRAIN], t_f.TRAIN, balance=True)
    cpcs.eval_ensemble_with_targetsubset(tfv, seq[t_f.VAL], t_f.VAL, balance=True)
    print(f"performance {tfv.cur_pair} {t_f.TEST}")
    cpcs.ensemble_performance_with_targetsubset(tfv, seq[t_f.TEST], t_f.TEST)
tdiff = (timeit.default_timer() - start_time) / (60*60)
print(f"total time: {tdiff:.2f} hours")
