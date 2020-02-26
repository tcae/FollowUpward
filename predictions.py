import pandas as pd
# import numpy as np
# import math
# import indicators as ind
import env_config as env
# from env_config import Env
import cached_crypto_data as ccd
import crypto_targets as ct

# import classify_keras as ck
# import condensed_features as condensed_features
# import crypto_features as crypto_features
# import adaptation_data as ad


class CategoryPredictions(ccd.CryptoData):

    def __init__(self, classifier, ohlcv: ccd.Ohlcv, targets: ct.Targets, features: ccd.Features):
        # self.ohlcv = ohlcv  # required for performance calculation
        self.targets = targets
        self.features = features
        self.classifier = classifier

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
        fdf = self.features.new_data(base, last, minutes + self.history())
        # prepare4keras(self.scaler, fdf, tdf, self.targets)
        pred = self.classifier.class_predict_of_features(fdf, base)
        fdf = fdf.iloc[-minutes:]
        pred_df = pd.DataFrame(data=pred, index=fdf.index, columns=self.targets.target_dict().keys())
        return pred_df


if __name__ == "__main__":
    print([(a,b) for a in range(3) for b in range(2)])
    env.test_mode()
    tee = env.Tee()
    ohlcv = ccd.Ohlcv()
    # df = ohlcv.load_data("xrp")
    # ohlcv.save_data("xrp", df)
    df = ohlcv.get_data("xrp", pd.Timestamp("2019-02-12 23:59:00+00:00"), 1000)
    print(f"got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}")
    df = ohlcv.set_type_data("xrp", "validation")
    print(f"got data: {len(df)} samples from {df.index[0]} until {df.index[-1]}")
    tee.close()
