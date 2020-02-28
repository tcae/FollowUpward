import pandas as pd
# import numpy as np
# import math
# import indicators as ind
import env_config as env
# from env_config import Env
import cached_crypto_data as ccd
import crypto_targets as ct

import classify_keras as ck
from classify_keras import PerfMatrix, EvalPerf  # required for pickle  # noqa
import condensed_features as cof
import aggregated_features as agf
# import crypto_features as cf
# import adaptation_data as ad


class CategoryPredictions(ccd.CryptoData):

    def __init__(self, classifier, ohlcv: ccd.Ohlcv, targets: ct.Targets, features: ccd.Features):
        # self.ohlcv = ohlcv  # required for performance calculation
        self.targets = targets
        self.features = features
        self.classifier = classifier
        super().__init__()

    def history(self):
        """ No history for features or targets required.
        """
        return 0

    def keys(self):
        "returns the list of element keys"
        keylist = list(self.targets.target_dict().keys())
        return keylist

    def mnemonic(self, base: str):
        "returns a string that represents this class/base as mnemonic, e.g. to use it in file names"
        return "CategoryPred{}".format(len(self.keys()))

    def new_data(self, base: str, last: pd.Timestamp, minutes: int):
        """ Downloads or calculates new data for 'minutes' samples up to and including last.
        """
        # tdf = self.targets.new_data(base, last, minutes + self.history())
        fdf = self.features.get_data(base, last, minutes + self.history())
        # prepare4keras(self.scaler, fdf, tdf, self.targets)
        pred = self.classifier.class_predict_of_features(fdf, base)
        fdf = fdf.iloc[-minutes:]
        pred_df = pd.DataFrame(data=pred, index=fdf.index, columns=self.targets.target_dict().keys())
        return pred_df


if __name__ == "__main__":
    # print([(a, b) for a in range(3) for b in range(2)])
    env.test_mode()
    tee = env.Tee()
    ohlcv = ccd.Ohlcv()
    base = "xrp"
    df = ohlcv.load_data(base)
    ohlcv.no_index_gaps(df)
    # ohlcv.save_data(base, df)
    df = ohlcv.get_data(base, pd.Timestamp("2019-02-28 01:00:00+00:00"), 1000)
    print(f"{ohlcv.mnemonic(base)} history: {ohlcv.history()}, {len(ohlcv.keys())} keys: {ohlcv.keys()}, fname: {ohlcv.fname(base)}")
    print(f"ohlcv got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}\n")

    cond = cof.F2cond24(ohlcv)
    df = cond.get_data(base, pd.Timestamp("2019-02-28 01:00:00+00:00"), 1000)
    print(f"{cond.mnemonic(base)} history: {cond.history()}, {len(cond.keys())} keys: {cond.keys()}, fname: {cond.fname(base)}")
    print(f"cond got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}\n")

    agg = agf.F1agg110(ohlcv)
    df = agg.get_data(base, pd.Timestamp("2019-02-28 01:00:00+00:00"), 1000)
    print(f"{agg.mnemonic(base)} history: {agg.history()}, {len(agg.keys())} keys: {agg.keys()}, fname: {agg.fname(base)}")
    print(f"agg got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}\n")

    trgt = ct.T10up5low30min(ohlcv)
    df = trgt.get_data(base, pd.Timestamp("2019-02-28 01:00:00+00:00"), 1000)
    print(f"{trgt.mnemonic(base)} history: {trgt.history()}, {len(trgt.keys())} keys: {trgt.keys()}, fname: {trgt.fname(base)}")
    print(f"trgt got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}\n")

    cpc = ck.Cpc(load_classifier="MLP_l1-16_do-0.2_h-19_no-l3_optAdam_F2cond24_0", save_classifier=None)
    pred = CategoryPredictions(cpc, ohlcv, trgt, cond)
    df = pred.get_data(base, pd.Timestamp("2019-02-28 01:00:00+00:00"), 1000)
    print(f"{pred.mnemonic(base)} history: {pred.history()}, {len(pred.keys())} keys: {pred.keys()}, fname: {pred.fname(base)}")
    print(f"pred got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}\n")
    print(df.head(4))
    print(df.tail(4))

    # df = ohlcv.load_data(base)
    # print(f"got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}")
    df = ohlcv.set_type_data(base, "training")
    print(f"ohlcv training got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}")
    tee.close()
