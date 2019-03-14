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
    return plt


class Cpc:
    """Provides methods to adapt single currency performance classifiers
    """

    def __init__(self, aggregation):
        self.key = aggregation

    def timeslice_targets(self, tfv, train_ratio, val_ratio, balance, days):
        self.tfv = tfv
        self.descr = tfv.cur_pair+' '+str(self.key)
        self.data_df = tfv.vec(self.key)
        seq = self.tfv.timeslice_and_select_targets(self.key, train_ratio, val_ratio, balance, days)
        self.train_targets = seq[t_f.TRAIN]
        self.val_targets = seq[t_f.VAL]
        self.test_targets = seq[t_f.TEST]

    def sklearn_data(self, target_df, np_data=None):
        # h=a[a.index.isin(f.index)] use only the data that is also in the target df
        skldata_df = self.data_df[self.data_df.index.isin(target_df.index)]
        skldata = self.tfv.to_sklearn(skldata_df, np_data)
        return skldata

    def sklearn_train_data(self):
        return self.sklearn_data(self.train_targets)

    def sklearn_validation_data(self):
        return self.sklearn_data(self.val_targets)

    def sklearn_test_data(self):
        return self.sklearn_data(self.test_targets)

    def predict_probs(self, target_df):
        skldata = self.sklearn_data(target_df)
        predicted = self.classifier.predict_proba(skldata.data)
        return predicted

    def eval_probs(self, val_data):
        start_time = timeit.default_timer()
        cross_val = np.zeros((3, 10, 4))
        # Now predict the action corresponding to samples on the second half:
        expected = val_data.target
        predicted = self.classifier.predict_proba(val_data.data)
        pred_cnt = len(predicted)
        for sample in range(pred_cnt):
            t = expected[sample]
            for cl in range(len(predicted[0])):
                hist_ix = int(math.floor(predicted[sample, cl]*10))
                correct_ix = int(t == cl)
                cross_val[cl, hist_ix, correct_ix] += 1
        for cl in range(len(cross_val)):
            for hist_ix in range(len(cross_val[0])):
                all_observ = (cross_val[cl, hist_ix, 1] + cross_val[cl, hist_ix, 0])
                cross_val[cl, hist_ix, 3] = all_observ / pred_cnt
                if all_observ != 0:
                    cross_val[cl, hist_ix, 2] = cross_val[cl, hist_ix, 1] / all_observ
        plt.figure()
        plt.title("prob cross validation of SVM "+self.descr)
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
                 label="% samples")

        plt.legend(loc="best")

        # proba = classifier.predict_proba(sample_sets['validation'].data)
        print(f"SVM evaluation: {timeit.default_timer() - start_time}")

    def prep_pickle(self):
        self.tfv = None
        self.data_df = None
        self.train_targets = None
        self.val_targets = None
        self.test_targets = None

    def adapt_classifier(self):
        """
        """
        train_data = self.sklearn_train_data()
        self.adapt_classifier_with_data(train_data)

    def adapt_classifier_with_data(self, train_data):
        """
        """
        train_cnt = len(train_data.data)
        print(f"{self.descr}: # samples training: {train_cnt}")

        # scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0), copy=True)
        self.scaler = None
        # sample_sets['training'].data = scaler.fit_transform(sample_sets['training'].data)
        # sample_sets['validation'].data = scaler.transform(sample_sets['validation'].data)

        start_time = timeit.default_timer()
        # Create a classifier: a support vector classifier
        self.classifier = svm.SVC(kernel='rbf', degree=3, gamma=0.001, probability=True,
                                  cache_size=400, class_weight='balanced', shrinking=True)
        # classifier = MLPClassifier(solver='adam',
        #                    hidden_layer_sizes=(20, 10, 3), random_state=1)

        # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # X, y = sample_sets['training'].data, sample_sets['training'].target
        # plot_learning_curve(classifier, title, X, y, (0.2, 1.01), cv=cv, n_jobs=1)
        #
        # plt.show()

        self.classifier.fit(train_data.data, train_data.target)
        print(f"{self.descr} SVM adaptation: {timeit.default_timer() - start_time}")

    def eval_classifier(self):
        """
        """
        val_data = self.sklearn_validation_data()
        self.eval_classifier_with_data(val_data)

    def eval_classifier_with_data(self, val_data):
        """
        """
        val_cnt = len(val_data.data)
        print(f"{self.descr}: # samples validation: {val_cnt}")
        start_time = timeit.default_timer()
        self.eval_probs(val_data)
        print(f"{self.descr} probability evaluation: {timeit.default_timer() - start_time}")

        start_time = timeit.default_timer()
        # Now predict the action corresponding to samples on the second half:
        expected = val_data.target
        predicted = self.classifier.predict(val_data.data)
        # proba = classifier.predict_proba(sample_sets['validation'].data)
        print(f"{self.descr} SVM evaluation: {timeit.default_timer() - start_time}")

        print("Classification report for {self.descr} classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(expected, predicted)))
        print("{self.descr} Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


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
        for ta in self.cpc_set:  # strip pandas due to pickle incompatibility between versions
            self.cpc_set[ta].prep_pickle()
        self.tfv = None
        fname = self.model_path + '/' + self.currency_pair + as_ext
        df_f = open(fname, 'wb')
        pickle.dump(self.__dict__, df_f, pickle.HIGHEST_PROTOCOL)
        df_f.close()
        print(f"classifier saved in {fname}")

    def combined_probs(self, target_df):
        predictions = None
        for ta in self.cpc_set:
            if isinstance(ta, int):
                pred = self.cpc_set[ta].predict_probs(target_df)
                if predictions is None:
                    predictions = pred
                else:
                    predictions = np.concatenate((predictions, pred), axis=1)
        return predictions

    def adapt_combo(self):
        """
        """
        combo_cpc = self.cpc_set[t_f.CPC]
        predictions = self.combined_probs(combo_cpc.train_targets)
        train_data = combo_cpc.sklearn_data(combo_cpc.train_targets, predictions)
        combo_cpc.adapt_classifier_with_data(train_data)

    def eval_combo(self):
        """
        """
        combo_cpc = self.cpc_set[t_f.CPC]
        predictions = self.combined_probs(combo_cpc.val_targets)
        val_data = combo_cpc.sklearn_data(combo_cpc.val_targets, predictions)
        combo_cpc.eval_classifier_with_data(val_data)

    def adapt_ensemble(self, train_ratio, val_ratio, balance, days):
        """
        """
        start_time = timeit.default_timer()
        # The crypto dataset
        fname = self.data_path + '/' + self.currency_pair + t_f.MSG_EXT
        self.tfv = t_f.TfVectors(filename=fname)
        self.cpc_set = self.tfv.aggregations
        for ta in self.cpc_set:
            cpc = Cpc(ta)
            cpc.timeslice_targets(self.tfv, train_ratio, val_ratio, balance, days)
            if isinstance(ta, int):
                cpc.adapt_classifier()
                cpc.eval_classifier()
            self.cpc_set[ta] = cpc
        self.adapt_combo()
        self.eval_combo()
        print(f"adapt_ensemble: {timeit.default_timer() - start_time}")
        self.save()

    def eval_ensemble(self, train_ratio, val_ratio, balance, days):
        """
        """
        start_time = timeit.default_timer()
        # The crypto dataset
        fname = self.data_path + '/' + self.currency_pair + t_f.MSG_EXT
        self.tfv = t_f.TfVectors(filename=fname)
        for ta in self.cpc_set:
            self.cpc_set[ta].timeslice_targets(self.tfv, train_ratio, val_ratio, balance, days)
            if isinstance(ta, int):
                self.cpc_set[ta].eval_classifier()
        self.eval_combo()
        print(f"eval_ensemble: {timeit.default_timer() - start_time}")


    def eval_combo_with_features(self, tf_vecs):
        """not yet ready
        """
        start_time = timeit.default_timer()
        combo_cpc = self.cpc_set[t_f.CPC]
        predictions = self.combined_probs(combo_cpc.val_targets)
        val_data = combo_cpc.sklearn_data(combo_cpc.val_targets, predictions)
        combo_cpc.eval_classifier_with_data(val_data)
        print(f"eval_combo_with_features: {timeit.default_timer() - start_time}")


cpcs = CpcSet(PAIR, t_f.DATA_PATH, '/Users/tc/tf_models/crypto')
cpcs.adapt_ensemble(train_ratio=0.6, val_ratio=0.3, balance=True, days=30)
cpcs.save()
cpcs.eval_ensemble(train_ratio=0.6, val_ratio=0.3, balance=True, days=30)
