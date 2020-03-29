import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)  # before importing tensorflow.

# import env_config as env
# from env_config import Env

# import os
import pandas as pd
import numpy as np
# import timeit
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
# import condensed_features as cof
# import aggregated_features as agf
# import classify_keras as ck
# import adaptation_data as ad
import prediction_data as preddat


logger = logging.getLogger(__name__)


class PerformanceData(ccd.CryptoData):

    def __init__(self, prediction_obj: preddat.PredictionData):
        super().__init__()
        self.predictions = prediction_obj
        self.path = self.predictions.predictor.path_with_epoch()
        self.btl = [0.3, 0.4]  # buy threshold list
        self.stl = [0.5, 0.6]  # sell threshold list
        self.set_type = None

    def history(self):
        "no history minutes are requires to calculate prediction data"
        return 0

    def keyiter(self):
        for bt in self.btl:  # bt = lower bound buy signal bucket
            for st in self.stl:  # st = lower bound sell signal bucket
                yield (f"bt{bt:1.1f}/st{st:1.1f}", (round(bt, 1), round(st, 1)))

    def keys(self):
        "returns the list of element keys"
        keyl = list()
        [keyl.append((bt, st)) for (lbl, (bt, st)) in self.keyiter()]
        return keyl

    def indexlist(self):
        "returns the list of element keys"
        ixl = list()
        [ixl.append(ix) for (_, ix) in self.keyiter()]
        return ixl

    def mnemonic(self):
        "returns a string that represent the PredictionData class as mnemonic, e.g. to use it in file names"
        mem = self.predictions.predictor.mnemonic_with_epoch() + "_performances"
        return mem

    def performance_calculation(self, in_df, bt: float, st: float, fee_factor: float, verbose=False):
        """ Expect a DataFrame with columns 'close', 'buy', 'sell' and a Timerange index.
            Returns a DataFrame with a % performance column 'perf' in %,
            buy_tic, buy_price, buy prediction, buy_threshold, sell_tic, sell_price, sell prediction, sell_threshold.
            The returned DataFrame has an int range index not compatible with in_df.
        """
        in_df.index.rename("tic", inplace=True)  # debug candy
        df = in_df.reset_index().copy()  # preserve tics as data but use numbers as index
        df.index.rename("ix", inplace=True)
        df.at[df.index[-1], "buy"] = 0  # no buy at last tic
        df.at[df.index[-1], "sell"] = 1  # forced sell at last tic
        df["sell_ix"] = pd.Series(dtype=pd.Int64Dtype())
        df["sell_tic"] = pd.Series(dtype='datetime64[ns, UTC]')
        df["sell_price"] = pd.Series(dtype=float)
        df["sell_trhld"] = st
        df["buy_trhld"] = bt
        df = df.loc[(df.buy >= bt) | (df.sell >= st)]
        # ccd.show_verbose(in_df, verbose)
        df = df.reset_index()  # preserve ix line numbers as unique reference of sell lines
        if len(df) > 1:
            # ccd.show_verbose(df, verbose)
            df["sell_ix"] = pd.Series(dtype=pd.Int64Dtype())
            df["sell_tic"] = pd.Series(dtype='datetime64[ns, UTC]')
            df.loc[(df.sell >= st), "sell_price"] = df.close
            df.loc[(df.sell >= st), "sell_ix"] = df["ix"]
            df.loc[(df.sell < st), "sell"] = np.nan  # debug candy
            df.loc[(df.sell >= st), "sell_tic"] = df["tic"]  # debug candy
            df.loc[:, "sell_ix"] = df.sell_ix.fillna(method='backfill')
            df.loc[:, "sell_price"] = df.sell_price.fillna(method='backfill')
            df.loc[:, "sell"] = df.sell.fillna(method='backfill')  # debug candy
            df.loc[:, "sell_tic"] = df.sell_tic.fillna(method='backfill')  # debug candy
            df = df.rename(columns={"close": "buy_price", "tic": "buy_tic", "ix": "buy_ix"})
            df = df.loc[(df.buy >= bt)]
            df = df.drop_duplicates("sell_ix", keep="first")
            df["perf"] = (df.sell_price * (1 - fee_factor) - df.buy_price * (1 + fee_factor)) / df.buy_price * 100
            df = df.reset_index(drop=True)  # eye candy
            # ccd.show_verbose(df, verbose)
        else:
            df = df.rename(columns={"close": "buy_price", "tic": "buy_tic", "ix": "buy_ix"})
            df = df.drop(df.index[-1])
        df = df.set_index("buy_tic")
        return df

    def new_data_deprecated(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        ohlcv_df = self.predictions.predictor.ohlcv.get_data(base, first, last, use_cache=True)
        pred_df = self.predictions.get_data(base, first, last, use_cache=False)  # enforce recalculation
        if (ohlcv_df is None) or ohlcv_df.empty or (pred_df is None) or pred_df.empty:
            return None
        cdf = pd.concat([ohlcv_df.close, pred_df.buy, pred_df.sell], axis=1, join="inner")  # combi df
        # mix = pd.MultiIndex.from_tuples(self.keys(), names=["bt", "st"])
        # perf_df = pd.DataFrame(data=float(0.0), index=cdf.index, columns=mix)
        ccd.show_verbose(cdf)
        rdf_list = list()
        for (lbl, (bt, st)) in self.keyiter():  # (bt, st) = signal thresholds
            logger.debug(f"performance_calculation with {base} {lbl} from {first} to {last}")
            rdf = self.performance_calculation(cdf, bt, st, ct.FEE, verbose=False)
            # perf_df.loc[[tic for tic in rdf.index], (round(rdf.buy_trhld, 1), round(rdf.sell_trhld, 1))] = rdf.perf
            if not rdf.empty:
                rdf_list.append(rdf)
        if len(rdf_list) > 0:
            rdf = pd.concat(rdf_list, join="outer", axis=0, keys=self.keys(), sort=True)
        ccd.show_verbose(rdf)
        return rdf

    def perfcalc(self, in_df, btl: list, stl: list, fee_factor: float):
        """ Expect a DataFrame with columns 'close', 'buy', 'sell' and a Timerange index.
            'buy', 'sell' are predictions likelihoods of these trade signals.
            'bt' and 'st' are lists of buy / sell threshold values.
            Returns a DataFrame with a % performance column 'perf', buy_tic and sell_tic.
            The returned DataFrame has an int range index not compatible with in_df.
        """

        in_df.index.rename("btic", inplace=True)  # debug candy
        df = in_df.reset_index().copy()  # preserve tics as data but use one numbers as index
        df.index.rename("bix", inplace=True)
        df = df.reset_index()  # preserve ix line numbers as unique reference of buy lines
        df = df.rename(columns={"close": "bprice"})
        df.at[df.index[-1], "buy"] = 0  # no buy at last tic
        df.at[df.index[-1], "sell"] = 1  # forced sell at last tic
        # show_verbose(df, lines=9)
        mix = pd.MultiIndex.from_product([btl, stl, ["sell", "six", "sprice", "stic", "perf"]])
        rdf = pd.DataFrame(index=df.index, columns=mix)
        tmix = pd.MultiIndex.from_product([btl, stl, ["perf", "count"]], names=['bt', 'st', 'KPI'])
        tix = in_df.index[:1]
        tdf = pd.DataFrame(index=tix, columns=tmix)
        tdf.loc[in_df.index[0], :] = 0
        # show_verbose(rdf, lines=9)
        for bt in btl:
            for st in stl:
                rdf[bt, st, "six"] = pd.Series(dtype=pd.Int32Dtype())
                rdf[bt, st, "stic"] = pd.Series(dtype='datetime64[ns, UTC]')
                rdf[bt, st, "sell"] = pd.Series(dtype=float)
                rdf[bt, st, "sprice"] = pd.Series(dtype=float)
                rdf[bt, st, "perf"] = pd.Series(dtype=float)
                # rdf.loc[df.sell >= st, (bt, st, ["six", "sprice", "sell", "stic"])] = \
                #     df.loc[:, ["bix", "bprice", "sell", "btic"]]
                rdf.loc[df.sell >= st, (bt, st, "six")] = df["bix"]
                rdf.loc[df.sell >= st, (bt, st, "sprice")] = df["bprice"]
                rdf.loc[df.sell >= st, (bt, st, "sell")] = df["sell"]
                rdf.loc[df.sell >= st, (bt, st, "stic")] = df["btic"]
                rdf.loc[df.sell >= st, (bt, st, "perf")] = np.nan
        # show_verbose(rdf, lines=9)
        rdf = rdf.fillna(axis=0, method="backfill")
        # show_verbose(rdf, lines=9)
        for bt in btl:
            for st in stl:
                df_mask = (df.buy >= bt) & (rdf[bt, st, "sell"] >= st)
                # print(f"df_mask threshold reduced\n{df_mask}")
                df_mask = ~rdf.loc[df_mask].duplicated((bt, st, "six"), keep="first")
                df_mask = df_mask.index[df_mask]
                # print(f"df_mask duplication reduced\n{df_mask}")
                # show_verbose(rdf.loc[df_mask], lines=9)
                rdf.loc[df_mask, (bt, st, "perf")] = \
                    rdf[bt, st, "sprice"] * (1 - fee_factor) - df["bprice"] * (1 + fee_factor)
                tdf.loc[tdf.index[0], (bt, st, "count")] = rdf[bt, st, "perf"].count()
                tdf.loc[tdf.index[0], (bt, st, "perf")] = rdf[bt, st, "perf"].sum()
        # show_verbose(rdf, lines=9)
        # df = pd.concat([df, rdf], axis=1, join="inner")  # combi df
        # show_verbose(df, lines=9)
        # print(tdf)
        return tdf

    def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=True):
        """ Predicts all samples and returns the result.
            set_type specific evaluations can be done using the saved prediction data.
        """
        logger.debug(f"start performance assessment of {base} from {first} to {last}")
        ohlcv_df = self.predictions.predictor.ohlcv.get_data(base, first, last, use_cache=True)
        pred_df = self.predictions.get_data(base, first, last, use_cache=False)  # enforce recalculation
        if (ohlcv_df is None) or ohlcv_df.empty or (pred_df is None) or pred_df.empty:
            return None
        cdf = pd.concat([ohlcv_df.close, pred_df.buy, pred_df.sell], axis=1, join="inner")  # combi df
        rdf = self.perfcalc(cdf, self.btl, self.stl, ct.FEE)
        logger.debug(f"end performance assessment of {base} from {first} to {last}\n{rdf}")
        return rdf

    def get_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp, use_cache=False):
        return self.new_data(base, first, last, use_cache)
