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

import crypto_targets_features as ctf

print(__doc__)
start_time = timeit.default_timer()

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

    def __init__(self, epoch, iteration, hs_name="no history_sets_config"):

        self.p_range = range(6, 10)
        self.perf = np.zeros((len(self.p_range),len(self.p_range)), dtype=EvalPerf)
        for bp in self.p_range:
            for sp in self.p_range:
                self.perf[bp - self.p_range[0], sp - self.p_range[0]] = \
                    EvalPerf(float((bp)/10), float((sp)/10))
        self.confusion = np.zeros((len(ctf.TARGETS),len(ctf.TARGETS)), dtype=int)
        self.iteration = iteration
        self.epoch = epoch
        self.history_sets_config = hs_name

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
        return "epoch {}, iteration {}: best performance {:6.0%} with buy threshold {:.1f} / sell threshold {:.1f} at {} transactions".format(self.epoch, self.iteration, best, bpt, spt, t)

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


    def assess_prediction(self, pred, skl_close, skl_target, skl_tics):
        """Assess the highest probability of a class prediction
        to find the optimum buy/sell threshold.

        handling of time gaps: in case of a time gap the last value of the time slice is taken
        to close any open transaction

        """
        pred_cnt = len(pred)
        begin = ctf.timestr(skl_tics[0])
        for sample in range(pred_cnt):
            if (sample + 1) >= pred_cnt:
                self.add_signal(1, skl_close[sample], ctf.TARGETS[ctf.SELL], ctf.TARGETS[ctf.SELL])
                end = ctf.timestr(skl_tics[sample])
                print("assessment between {} and {}".format(begin, end))
            elif (skl_tics[sample+1] - skl_tics[sample]) > np.timedelta64(1, 'm'):
                self.add_signal(1, skl_close[sample], ctf.TARGETS[ctf.SELL], ctf.TARGETS[ctf.SELL])
                end = ctf.timestr(skl_tics[sample])
                print("assessment between {} and {}".format(begin, end))
                begin = ctf.timestr(skl_tics[sample+1])
            else:
                high_prob_cl = 0
                for cl in range(len(pred[0])):
                    if pred[sample, high_prob_cl] < pred[sample, cl]:
                        high_prob_cl = cl
                self.add_signal(pred[sample, high_prob_cl], skl_close[sample],
                                high_prob_cl, skl_target[sample])

    def conf(self, estimate_class, target_class):
        elem = self.confusion[estimate_class, target_class]
        targets = 0
        estimates = 0
        for i in ctf.TARGETS:
            targets += self.confusion[estimate_class, ctf.TARGETS[i]]
            estimates += self.confusion[ctf.TARGETS[i], target_class]
        return (elem, elem/estimates, elem/targets)

    def report_assessment(self):
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
        print("")
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

class Cpc:
    """Provides methods to adapt single currency performance classifiers
    """
    def __init__(self, load_classifier, save_classifier):
        self.load_classifier = load_classifier
        self.save_classifier = save_classifier
        self.model_path = MODEL_PATH
        self.scaler = None
        self.classifier = None
        self.pm = None
        self.iteration = 0


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

    def save(self):
        #for ta in self.cpc:  # strip pandas due to pickle incompatibility between versions
        #    self.cpc[ta].prep_pickle()
        fname = str("{}/{}_{}-{}{}".format(self.model_path, self.save_classifier,
                    self.epoch, self.iteration, ctf.PICKLE_EXT))
        df_f = open(fname, 'wb')
        pickle.dump(self.__dict__, df_f, pickle.HIGHEST_PROTOCOL)
        df_f.close()
        print(f"{ctf.timestr()} classifier saved in {fname}")

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
        pm.assess_prediction(pred, skldata.close, skldata.target, skldata.tics)

        tdiff = (timeit.default_timer() - start_time) / 60
        print(f"performance_with_targetsubset time: {tdiff:.1f} min")

    def adapt_with_history_sets(self):
        start_time = timeit.default_timer()
        hs = ctf.HistorySets()
        ctfd = dict()
        for self.epoch in range(2):
            hs.label_iterations()
            train_samples = 0
            for self.iteration in range(max(hs.max_iter.values())):
                self.trainpm = PerfMatrix(self.epoch, self.iteration, hs.hs_name)
                self.valpm = PerfMatrix(self.epoch, self.iteration, hs.hs_name)
                start_time2 = timeit.default_timer()
                start_time3 = timeit.default_timer()
                for base in hs.baselist:
                    traindf = hs.trainset_iteration(base, self.iteration)
                    train_samples = 0
                    if traindf.empty:
                        print(f"empty training subset for {base} in iteration {self.iteration}")
                    else:
                        # print("training subset: ", df.head(), df.tail())
                        if base in ctfd:
                            tfv = ctfd[base]
                        else:
                            tfv = ctf.TargetsFeatures(base, ctf.QUOTE)
                            try:
                                tfv.load_classifier_features()
                            except ctf.MissingHistoryData as msg:
                                print(f"adapt_with_history_sets error: {msg}")
                                continue
                            ctfd[base] = tfv
                        try:
                            cpc.adapt_with_targetsubset(tfv, traindf, ctf.TRAIN)
                        except ctf.NoSubsetWarning as msg:
                            print("adapt_with_history_sets adapt train error {}/{}/{}: {}".format(
                                    self.epoch, self.iteration, base, msg))
                tdiff = (timeit.default_timer() - start_time3) / (60*60)
                print(f"adaptation iteration {self.iteration} total time: {tdiff:.2f} hours")
                start_time3 = timeit.default_timer()
                train_samples2 = 0
                for base in hs.baselist:
                    traindf = hs.set_of_type(base, ctf.TRAIN)
                    train_samples2 += len(traindf)
                    if  traindf.empty:
                        print(f"empty training subset for {base}")
                    else:
                        tfv = ctfd[base]
                        if tfv is None:
                            continue
                        # cpc.eval_with_targetsubset(tfv, traindf, ctf.TRAIN)
                        try:
                            cpc.performance_with_targetsubset(tfv, traindf, self.trainpm, ctf.TRAIN)
                        except ctf.NoSubsetWarning as msg:
                            print("adapt_with_history_sets perf train error {}/{}/{}: {}".format(
                                    self.epoch, self.iteration, base, msg))
                print("performance {} after iteration {}".format(ctf.TRAIN, self.iteration))
                self.trainpm.report_assessment()
                for base in hs.baselist:
                    valdf = hs.set_of_type(base, ctf.VAL)
                    if valdf.empty:
                        print(f"empty validation subset for {base}")
                    else:
                        tfv = ctfd[base]
                        if tfv is None:
                            continue
                        try:
                            cpc.performance_with_targetsubset(tfv, valdf, self.valpm, ctf.VAL)
                        except ctf.NoSubsetWarning as msg:
                            print("adapt_with_history_sets perf val error {}/{}/{}: {}".format(
                                    self.epoch, self.iteration, base, msg))
                print("performance {} after iteration {}".format(ctf.VAL, self.iteration))
                self.valpm.report_assessment()
                tdiff = (timeit.default_timer() - start_time3) / (60*60)
                print(f"performance iteration {self.iteration} total time: {tdiff:.2f} hours")
                tdiff = (timeit.default_timer() - start_time2) / (60*60)
                print(f"total iteration {self.iteration} time: {tdiff:.2f} hours")
                self.save()
            assert train_samples == train_samples2, "train_samples mismatch: ".format(
                    train_samples, train_samples2)
            # evaluate wrong classifications and reinforce those in the next round
            hs.use_training_mistakes()
        testpm = PerfMatrix(self.epoch, self.iteration, hs.hs_name)
        for base in hs.baselist:
            testdf = hs.set_of_type(base, ctf.TEST)
            tfv = ctfd[base]
            if tfv is None:
                continue
            try:
                cpc.performance_with_targetsubset(tfv, testdf, testpm, ctf.TEST)
            except ctf.NoSubsetWarning as msg:
                print(f"adapt_with_history_sets perf test error {base}: {msg}")
        print("performance {} ".format(ctf.TEST))
        testpm.report_assessment()

        tdiff = (timeit.default_timer() - start_time) / (60*60)
        print(f"total adaptation and performance evaluation time: {tdiff:.2f} hours")

if __name__ == "__main__":
    if True:
        load_classifier = None  # "2019-04-18_Full-SVM-orig-buy-hold-sell_gamma_0.01_extended_features_smooth+xrp_usdt5+D10"
        save_classifier = str("SVM_0.001gamma+D10_newCpc")
        #     load_classifier = str("{}{}".format(BASE, target_key))
        unit_test = False
        cpc = Cpc(load_classifier, save_classifier)
        cpc.adapt_with_history_sets()

