""" main docstring
"""
import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)  # before importing tensorflow.

import env_config as env
from env_config import Env

import os
# import pandas as pd
# import numpy as np
import timeit
# import itertools
# import math
# import pickle

# from concurrent.futures import ProcessPoolExecutor
# from multiprocessing import Process

# Import datasets, classifiers and performance metrics
# from sklearn import preprocessing
# from sklearn.neural_network import MLPClassifier

# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.metrics as km
# import keras
# import keras.metrics as km
# import tensorflow.compat.v1 as tf1

# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.model_selection import ShuffleSplit
# from sklearn.utils import Bunch

import crypto_targets as ct
import cached_crypto_data as ccd
import condensed_features as cof
import aggregated_features as agf
# import classify_keras as ck
import adaptation_data as ad
import classifier_predictor as class_pred

"""
NUM_PARALLEL_EXEC_UNITS = 6
config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
    inter_op_parallelism_threads=2,
    allow_soft_placement=True,
    device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.compat.v1.Session(config=config)
# keras.backend.set_session(session)
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
"""
logger = logging.getLogger(__name__)


def pinfo(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

    # def convert_cpc_classifier():
    #     load_classifier = "MLP2_epoch=15_talos_iter=3_l1=14_do=0.2_l2=16_l3=no_opt=Adam__F2cond20__T10up5low30min"
    #     # "MLP_l1-16_do-0.2_h-19_no-l3_optAdam_F2cond24_0"
    #     cpc = ck.Cpc(load_classifier, None)
    #     ohlcv = ccd.Ohlcv()
    #     targets = ct.T10up5low30min(ohlcv)
    #     if True:
    #         features = cof.F2cond20(ohlcv)
    #     else:
    #         features = agf.AggregatedFeatures(ohlcv)
    #     clsfer = Classifier(ohlcv, features, targets)
    #     clsfer.kerasmodel = cpc.classifier
    #     clsfer.scaler = cpc.scaler
    #     clsfer.params["l1"] = 14
    #     clsfer.params["do"] = 0.2
    #     clsfer.params["l2"] = 16
    #     clsfer.params["l3"] = "no"
    #     clsfer.params["opt"] = "Adam"
    #     clsfer.epoch = 15
    #     clsfer.save()
    #     return clsfer


if __name__ == "__main__":
    env.test_mode()
    start_time = timeit.default_timer()
    ohlcv = ccd.Ohlcv()
    targets = ct.T10up5low30min(ohlcv)
    try:

        if True:
            features = cof.F3cond14(ohlcv)
        else:
            features = agf.AggregatedFeatures(ohlcv)
        classifier = class_pred.Classifier(ohlcv, features, targets)
        if True:
            classifier.adapt_keras()
        else:
            # classifier = convert_cpc_classifier()
            classifier.load_classifier_file_collection(
                "MLP2_epoch=15_talos_iter=3_l1=14_do=0.2_l2=16_l3=no_opt=Adam__F2cond20__T10up5low30min")
            # classifier.load("MLP2_talos_iter-3_l1-14_do-0.2_l2-16_l3-no_opt-Adam__F2cond20__T10up5low30min", epoch=15)
            env.Tee.set_path(classifier.path_without_epoch(), log_prefix="TrainEval")
            classifier.save()
            # MLP2_epoch=0_talos_iter=0_l1=16_do=0.2_h=19_no=l3_opt=optAdam__F2cond20__T10up5low30min_0")
            # perf = PerformanceData(PredictionData(classifier))
            # for base in Env.bases:
            #     features_df = features.load_data(base)
            #     print(features_df.index[0], features_df.index[-1])
            #     perf_df = perf.get_data(base, features_df.index[0], features_df.index[-1], use_cache=False)
            #     perf.save_data(base, perf_df)

            print("return new: ", classifier.assess_performance(Env.bases, ad.VAL))
            # print("return new: ", classifier.assess_performance(["xrp"], ad.VAL))
            env.Tee.reset_path()

    except KeyboardInterrupt:
        logger.info("keyboard interrupt")
    tdiff = (timeit.default_timer() - start_time) / 60
    logger.info(f"total script time: {tdiff:.0f} min")
