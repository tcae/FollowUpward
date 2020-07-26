# import math
import pandas as pd
import numpy as np
import scipy.signal as sig
from env_config import Env
# import env_config as env
# import cached_crypto_data as ccd
# import condensed_features as cof
# import crypto_targets as ct
# import adaptation_data as ad
# import update_crypto_history as uch


def TriangleTestdata(first: pd.Timestamp, last: pd.Timestamp):
    price = 200
    volume = 100
    amplitude = price * 0.04  # 2% of price
    variation = 0.01
    first = pd.Timestamp("2018-06-29 00:00:00+00:00", tz=Env.tz)
    # first is the reference point to reproduce the pattern
    xfirst = int((first - first) / pd.Timedelta(1, unit="T"))
    xlast = int((last - first) / pd.Timedelta(1, unit="T"))
    minutes = xlast - xfirst + 1
    # minutes are used as degrees, i.e. 1 full sinus = 360 degree = 6h
    x = np.linspace(xfirst, xlast, num=minutes) * np.pi / 180.
    y = sig.sawtooth(x, width=0.5)
    df = pd.DataFrame(
        data={
            "open":  y * amplitude / 2 + price + (np.absolute(y) - 1) * variation * price / 4,
            "high":  y * amplitude / 2 + price - (np.absolute(y) - 1) * variation * price / 2,
            "low":   y * amplitude / 2 + price + (np.absolute(y) - 1) * variation * price / 2,
            "close": y * amplitude / 2 + price - (np.absolute(y) - 1) * variation * price / 4,
            "volume": - (np.absolute(y) - 1) * volume},
        index=pd.date_range(first, periods=minutes, freq="T", tz="Europe/Amsterdam"))
    return df


def SinusTestdata(first: pd.Timestamp, last: pd.Timestamp):
    price = 200
    volume = 100
    amplitude = price * 0.04  # 2% of price
    variation = 0.01
    first = pd.Timestamp("2018-06-29 00:00:00+00:00", tz=Env.tz)
    # first is an arbitrary reference point to reproduce the pattern
    xfirst = int((first - first) / pd.Timedelta(1, unit="T"))
    xlast = int((last - first) / pd.Timedelta(1, unit="T"))
    minutes = xlast - xfirst + 1
    # minutes are used as degrees, i.e. 1 full sinus = 360 degree = 6h
    x = np.linspace(xfirst, xlast, num=minutes) * np.pi / 180.
    y = np.sin(x)
    df = pd.DataFrame(
        data={
            "open":  y * amplitude / 2 + price + (np.absolute(y) - 1) * variation * price / 4,
            "high":  y * amplitude / 2 + price - (np.absolute(y) - 1) * variation * price / 2,
            "low":   y * amplitude / 2 + price + (np.absolute(y) - 1) * variation * price / 2,
            "close": y * amplitude / 2 + price - (np.absolute(y) - 1) * variation * price / 4,
            "volume": - (np.absolute(y) - 1) * volume},
        index=pd.date_range(first, periods=minutes, freq="T", tz=Env.tz))
    return df


# class Testdata(ccd.Ohlvc):
#     """ Generates test data equal in format to crypto price data but defined pattern to check algorithms.
#         Data is generated in the timerange of ad.SplitSets.overall_timerange()
#     """

#     bases = {"sinus": SinusTestdata, "triangle": TriangleTestdata}
#     first, last = ad.SplitSets.overall_timerange()

#     def new_data(self, base: str, first: pd.Timestamp, last: pd.Timestamp):
#         pass


# if __name__ == "__main__":
#     env.test_mode()
#     tee = env.Tee(log_prefix="Testdata")
#     testdata = Testdata()
#     bases = Env.bases
#     # bases = Env.training_bases

#     first, last = ad.SplitSets.overall_timerange()
#     data_objs = [testdata, cof.F4CondAgg(testdata), ct.TargetGrad30m(testdata)]
#     uch.regenerate_dataset(bases, first, last, data_objs)
#     tee.close()
