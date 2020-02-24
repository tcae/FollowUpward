import pandas as pd
# import numpy as np
# import timeit
# import itertools
# import math
# import pickle
# import h5py

# Import datasets, classifiers and performance metrics
# from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

import tensorflow as tf
import tensorflow.keras as keras
# import tensorflow.keras.metrics as km
# import keras
# import keras.metrics as km
# import tensorflow.compat.v1 as tf1
# import talos as ta

# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

# import env_config as env
# from env_config import Env
import crypto_targets as ct
import crypto_features as cf
# import crypto_history_sets as chs
import cached_crypto_data as ccd
# import condensed_features as cof
# import aggregated_features as agf
# import classify_keras as ck
import adaptation_data as ad


print(f"Tensorflow version: {tf.version.VERSION}")
print(f"Keras version: {keras.__version__}")
print(__doc__)


class Estimator:

    def __init__(self, scaler, target_class: ct.Targets, feature_class: cf.Features):
        self.tcls = target_class
        self.fcls = feature_class
        self.scaler = scaler

    @staticmethod
    def mnemonic():
        "returns a string that represent the estimator class as mnemonic, e.g. to use it in file names"
        assert False, "to be implemented by subclass"
        return None


class PredictionData(ccd.CryptoData):

    def __init__(self, estimator_obj: Estimator, ohlcv_class: ccd.OhlcvData):
        self.ocls = ohlcv_class  # ohlcv class is required to calculate actual performance
        self.est = estimator_obj

    def history(self):
        "no history minutes are requires to calculate prediction data"
        return 0

    def keys(self):
        "returns the list of element keys"
        return self.est.tcls.target_dict().keys()

    def mnemonic(self):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        mem = self.est.mnemonic() + "_" + self.est.tcls.mnemonic() + "_" + self.est.fcls.mnemonic()
        return mem

    def new_data(self, base, lastdatetime, minutes):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        fdf = self.est.fcls.get_data(base, lastdatetime, minutes)
        tdf = self.est.tcls.get_data(base, lastdatetime, minutes)
        df = pd.concat([fdf, tdf], axis=1, join="inner", keys=["features", "targets"])
        for tk in self.est.tcls.target_dict():
            df[("pred", tk)] = 0.0  # prepare for predictioins with same index
        fdf = df["features"]  # inner join ensures shared samples
        tdf = df["targets"]
        data, target = ad.prepare4keras(self.est.scaler, fdf, tdf, self.est.tcls)
        pred = self.est.predict_on_batch(data)
        df["pred"] = pred
        return self._check_timerange(df, lastdatetime, minutes)

        # df_ages["age_by_decade"] = pd.cut(x=df_ages["age"], bins=[20, 29, 39, 49], labels=["20s", "30s", "40s"])
        # ! introduce quantiles in another code area beside pred
