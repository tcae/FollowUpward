import pandas as pd
import logging
# import numpy as np
# import math
# import indicators as ind
import env_config as env
from env_config import Env
import cached_crypto_data as ccd
import crypto_targets as ct

import classify_keras as ck
from classify_keras import PerfMatrix, EvalPerf  # required for pickle  # noqa
import condensed_features as cof
import aggregated_features as agf
# import crypto_features as cf
# import adaptation_data as ad

logger = logging.getLogger(__name__)


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

    def mnemonic(self):
        "returns a string that represents this class as mnemonic, e.g. to use it in file names"
        mem = "CategoryPred{}__".format(len(self.keys())) + self.classifier.load_classifier + "__" \
            + self.features.mnemonic() + "_" + self.targets.mnemonic()
        return mem

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


def test_data_handling():
    ohlcv = ccd.Ohlcv()
    cond = cof.F2cond20(ohlcv)
    # agg = agf.F1agg110(ohlcv)
    trgt = ct.T10up5low30min(ohlcv)
    cpc = ck.Cpc(load_classifier="MLP_l1-16_do-0.2_h-19_no-l3_optAdam_F2cond24_0", save_classifier=None)
    pred = CategoryPredictions(cpc, ohlcv, trgt, cond)
    for base in Env.bases:
        ohlcv.check_report(base)
        cond.check_report(base)
        # agg.check_report(base)
        trgt.check_report(base)

        pred.check_report(base)
        # logger.debug(df.head(4))
        # logger.debug(df.tail(4))

    # df = ohlcv.load_data(base)
    # logger.debug(f"got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}")
    # df = ohlcv.set_type_data(base, "training")
    # logger.debug(f"ohlcv training got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}")


def classify_saved_history(classifier, targets: ct.Targets, features: ccd.Features):
    ohlcv = ccd.Ohlcv()
    feat = features(ohlcv)
    trgt = targets(ohlcv)
    cpc = ck.Cpc(load_classifier=classifier, save_classifier=None)
    pred = CategoryPredictions(cpc, ohlcv, trgt, feat)
    for base in Env.bases:
        fdf = feat.load_data(base)
        pdf = pred.get_data(base, last=fdf.index[-1], minutes=len(fdf), use_cache=False)  # enforce classification
        pred.save_data(base, pdf)

        if False:  # print excerpt to check that results are there and distinct from each other
            pdf = pred.load_data(base)
            logger.debug(f"{base} results")
            logger.debug(pdf.iloc[[0, 1, -2, -1]])


def convert_cache_files():
    for base in Env.bases:
        df = ccd.load_asset_dataframe(base, path=Env.data_path)
        ccd.dfdescribe(f"ohlcv: {base}", df)
        if (df is not None) and ccd.no_index_gaps(df):
            ohlcv = ccd.Ohlcv()
            ohlcv.save_data(base, df)

        cond = cof.CondensedFeatures(base, path=Env.data_path)
        cond.load_cache_ok()
        fdf = cond.vec
        # fdf = fdf.iloc[fdf.index != fdf.index[-2]]  # provoke index gap
        # ccd.dfdescribe(f"CondensedFeatures as loaded: {base}", df)
        # logger.debug(f"cond Diff: {len(fdf) + cond.history() - len(df)}: len(cond) + history =
        # {len(fdf)} + {cond.history()} = {len(fdf) + cond.history()} != {len(df)} = len(ohlcv)")

        tdf = fdf["target"]
        ccd.dfdescribe(f"CondensedFeatures only target: {base}", tdf)

        fdf = fdf.drop(columns=["target"])
        ccd.dfdescribe(f"CondensedFeatures without target: {base}", fdf)

        if (fdf is not None) and ccd.no_index_gaps(fdf):
            cond = cof.F2cond20(ohlcv)
            cond.save_data(base, fdf)
            trgt = ct.T10up5low30min(ohlcv)
            trgt.save_data(base, tdf)

        agg = agf.AggregatedFeatures(base, path=Env.data_path)
        agg.load_cache_ok()
        fdf = agg.vec
        # ccd.dfdescribe(f"AggregatedFeatures: {base}", df)
        # logger.debug(f"agg Diff: {len(fdf) + cond.history() - len(df)}: len(cond) + history =
        # {len(fdf)} + {cond.history()} =
        # {len(fdf) + cond.history()} != {len(df)} = len(ohlcv)")

        fdf = fdf.drop(columns=["close", "target"])
        ccd.dfdescribe(f"AggregatedFeatures without close and target: {base}", fdf)

        if (fdf is not None) and ccd.no_index_gaps(fdf):
            agg = agf.F1agg110(ohlcv)
            agg.save_data(base, fdf)

        # tdf = df["target"]
        # ccd.dfdescribe(f"AggregatedFeatures only target: {base}", tdf)


if __name__ == "__main__":
    # logger.debug([(a, b) for a in range(3) for b in range(2)])
    # env.test_mode()
    tee = env.Tee()
    # test_data_handling()  # successfully tested
    # convert_cache_files()  # successfully done
    classify_saved_history("MLP_l1-16_do-0.2_h-19_no-l3_optAdam_F2cond24_0", ct.T10up5low30min, cof.F2cond20)
    tee.close()
